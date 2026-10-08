use std::{
    error::Error,
    num::NonZero,
    panic::{AssertUnwindSafe, resume_unwind},
    sync::{
        Arc, Mutex, RwLock,
        atomic::{AtomicUsize, Ordering},
    },
    time::Duration,
};

use arc_swap::ArcSwap;
use futures::{FutureExt, future::BoxFuture};
use pumpkin_config::{AdvancedConfiguration, BasicConfiguration, TelemetryConfig};
use pumpkin_data::{dimension::Dimension, entity::EntityType, packet::CURRENT_MC_VERSION};
use pumpkin_protocol::{
    ClientPacket, RawPacket, ServerPacket, VarInt,
    java::{
        client::play::{CKeepAlive, CPlayerPosition, CSetCamera},
        packet_decoder::TCPNetworkDecoder,
        server::play::SPlayerInput,
    },
    packet::MultiVersionJavaPacket,
    ser::NetworkReadExt,
};
use pumpkin_util::{
    GameMode,
    math::{position::BlockPos, vector2::Vector2, vector3::Vector3},
    world_seed::Seed,
};
use pumpkin_world::cylindrical_chunk_iterator::Cylindrical;
use tempfile::TempDir;
use tokio::{
    io::BufReader,
    net::{TcpListener, TcpStream},
    time::timeout,
};
use uuid::Uuid;

use crate::{
    data::VanillaData,
    entity::{Entity, EntityBase, living::LivingEntity, player::Player},
    net::{
        ClientPlatform, GameProfile, PacketRateLimiter, PlayerConfig,
        java::{JavaClient, pending::PendingConnection},
    },
    plugin::{
        EventHandler,
        api::events::{EventPriority, player::PlayerTeleportEvent},
    },
    server::Server,
    world::World,
};

type TestResult<T = ()> = Result<T, Box<dyn Error + Send + Sync>>;

const OBSERVER_POSITION: Vector3<f64> = Vector3::new(1.25, 100.5, 2.75);
const TARGET_POSITION: Vector3<f64> = Vector3::new(4.25, 102.5, 6.75);
const OBSERVER_ROTATION: (f32, f32) = (31.25, -17.5);
const NONZERO_VELOCITY: Vector3<f64> = Vector3::new(0.75, -0.5, 0.25);
const ZERO_VELOCITY: Vector3<f64> = Vector3::new(0.0, 0.0, 0.0);
const PACKET_TIMEOUT: Duration = Duration::from_secs(5);

struct CameraFixture {
    server: Arc<Server>,
    world: Arc<World>,
    observer: Arc<Player>,
    target: Arc<LivingEntity>,
    wire: TCPNetworkDecoder<BufReader<TcpStream>>,
    barrier_id: i64,
    _directory: TempDir,
}

impl CameraFixture {
    async fn new() -> TestResult<Self> {
        let directory = TempDir::new()?;
        let listener = TcpListener::bind("127.0.0.1:0").await?;
        let peer = TcpStream::connect(listener.local_addr()?).await?;
        let (socket, address) = listener.accept().await?;

        let basic = BasicConfiguration {
            seed: Seed(0),
            default_level_name: directory.path().to_string_lossy().into_owned(),
            allow_nether: false,
            allow_end: false,
            allow_chat_reports: false,
            use_favicon: false,
            ..BasicConfiguration::default()
        };
        let mut advanced = AdvancedConfiguration::default();
        advanced.networking.java.online_mode = false;
        advanced.networking.java.encryption = false;
        advanced.networking.bedrock.online_mode = false;
        advanced.player_data.save_player_data = false;
        advanced.advancement.save_advancements = false;
        let data = VanillaData {
            banned_ip_list: RwLock::default(),
            banned_player_list: RwLock::default(),
            operator_config: RwLock::default(),
            user_cache: RwLock::default(),
            whitelist_config: RwLock::default(),
        };
        let telemetry = TelemetryConfig {
            enabled: false,
            ..TelemetryConfig::default()
        };
        let server = Server::new(basic, advanced, telemetry, data, Vec::new()).await?;
        let world = server.get_world_from_dimension(&Dimension::OVERWORLD);
        let profile = GameProfile {
            id: Uuid::new_v4(),
            name: "CameraObserver".to_string(),
            properties: ArcSwap::from_pointee(Vec::new()),
            profile_actions: None,
        };
        let config = PlayerConfig {
            view_distance: NonZero::new(2).ok_or("view distance must be nonzero")?,
            ..PlayerConfig::default()
        };
        let pending = PendingConnection::new(
            socket,
            address,
            0,
            PacketRateLimiter::from_config(&server.advanced_config.networking.java.packet_limiter),
            Arc::downgrade(&server),
        );
        pending
            .connection_state
            .store(pumpkin_protocol::ConnectionState::Play);
        let mut client = JavaClient::from_pending(pending, profile.clone(), config.clone());
        client.start_outgoing_packet_task();
        let observer = Arc::new(Player::new(
            Arc::new(ClientPlatform::Java(client)),
            profile,
            config,
            &world,
            GameMode::Spectator,
        ));
        observer.get_entity().set_pos(OBSERVER_POSITION);
        observer
            .get_entity()
            .set_rotation(OBSERVER_ROTATION.0, OBSERVER_ROTATION.1);
        observer.set_client_loaded(true);
        // All scenarios stay in this view; chunk generation is not part of the camera transaction.
        observer.watched_section.store(Cylindrical::new(
            observer.get_entity().chunk_pos.load(),
            observer.config.load().view_distance,
        ));
        world.add_player(&observer)?;
        observer
            .client
            .java()
            .ok_or("Java fixture required")?
            .set_player(observer.clone());
        let target = Arc::new(LivingEntity::new(Entity::new(
            world.clone(),
            TARGET_POSITION,
            &EntityType::PIG,
        )));
        target.entity.set_rotation(146.25, 24.5);
        target.entity.velocity.store(NONZERO_VELOCITY);
        assert!(world.spawn_entity(target.clone()));

        Ok(Self {
            server,
            world,
            observer,
            target,
            wire: TCPNetworkDecoder::new(BufReader::new(peer)),
            barrier_id: 0,
            _directory: directory,
        })
    }

    fn client(&self) -> TestResult<&JavaClient> {
        self.observer
            .client
            .java()
            .ok_or_else(|| "Java fixture required".into())
    }

    async fn packets(&mut self) -> TestResult<Vec<RawPacket>> {
        self.barrier_id += 1;
        let barrier_id = self.barrier_id;
        // A FIFO marker proves an empty batch without racing the writer or sleeping.
        self.client()?.try_send_packet(&CKeepAlive::new(barrier_id));
        timeout(PACKET_TIMEOUT, async {
            let mut packets = Vec::new();
            loop {
                let packet = self.wire.get_raw_packet().await?;
                if packet.id == CKeepAlive::to_id(CURRENT_MC_VERSION) {
                    let mut payload = &packet.payload[..];
                    let marker = CKeepAlive::read(&mut payload, &CURRENT_MC_VERSION)?;
                    assert_eq!(marker.keep_alive_id, barrier_id);
                    assert!(payload.is_empty());
                    return Ok(packets);
                }
                packets.push(packet);
                assert!(
                    packets.len() <= 512,
                    "camera fixture produced excessive packets"
                );
            }
        })
        .await?
    }

    async fn close(&self) {
        if let Some(client) = self.observer.client.java() {
            client.close();
            client.await_tasks().await;
            client.player.store(Arc::new(None));
        }
        let _ = self.world.remove_player(&self.observer, false).await;
        self.world.remove_entity(self.target.as_ref());
        self.server.shutdown().await;
        self.world.level.world_portal.store(Arc::new(None));
    }
}

async fn with_fixture(
    test: impl for<'a> FnOnce(&'a mut CameraFixture) -> BoxFuture<'a, TestResult>,
) -> TestResult {
    let mut fixture = CameraFixture::new().await?;
    let result = AssertUnwindSafe(async {
        fixture.packets().await?;
        test(&mut fixture).await
    })
    .catch_unwind()
    .await;
    // Level owns native worker threads, so release them even when an assertion fails.
    fixture.close().await;
    match result {
        Ok(result) => result,
        Err(payload) => resume_unwind(payload),
    }
}

#[derive(Debug, PartialEq)]
struct ObserverState {
    position: Vector3<f64>,
    last_position: Vector3<f64>,
    rotation: (f32, f32),
    velocity: Vector3<f64>,
    teleport_id: i32,
    epoch: u32,
    pending: Option<(VarInt, Vector3<f64>)>,
    camera_id: i32,
}

impl ObserverState {
    fn capture(player: &Player) -> Self {
        let entity = player.get_entity();
        Self {
            position: entity.pos.load(),
            last_position: entity.last_pos.load(),
            rotation: (entity.yaw.load(), entity.pitch.load()),
            velocity: entity.velocity.load(),
            teleport_id: player.teleport_id_count.load(Ordering::Relaxed),
            epoch: player.chunk_send_epoch.load(Ordering::Relaxed),
            pending: *player
                .awaiting_teleport
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner),
            camera_id: player.get_camera_entity_id(),
        }
    }
}

fn camera_packets(packets: &[RawPacket]) -> Vec<&RawPacket> {
    packets
        .iter()
        .filter(|packet| {
            packet.id == CPlayerPosition::to_id(CURRENT_MC_VERSION)
                || packet.id == CSetCamera::to_id(CURRENT_MC_VERSION)
        })
        .collect()
}

fn assert_camera_change(
    packets: &[RawPacket],
    position: Vector3<f64>,
    rotation: (f32, f32),
    teleport_id: i32,
    camera_id: i32,
) -> TestResult {
    let packets = camera_packets(packets);
    assert_eq!(
        packets.len(),
        2,
        "exactly one teleport and one camera change"
    );
    assert_eq!(packets[0].id, CPlayerPosition::to_id(CURRENT_MC_VERSION));
    assert_eq!(packets[1].id, CSetCamera::to_id(CURRENT_MC_VERSION));
    let mut payload = &packets[0].payload[..];
    let teleport = CPlayerPosition::read(&mut payload, &CURRENT_MC_VERSION)?;
    assert!(payload.is_empty());
    assert_eq!(teleport.teleport_id.0, teleport_id);
    assert_eq!(teleport.position, position);
    assert_eq!((teleport.yaw, teleport.pitch), rotation);
    assert_eq!(teleport.delta, ZERO_VELOCITY);
    assert!(teleport.relatives.is_empty());
    let mut payload = &packets[1].payload[..];
    assert_eq!(payload.get_var_int()?.0, camera_id);
    assert!(payload.is_empty());
    Ok(())
}

#[derive(Clone, Copy)]
enum TeleportAction {
    Observe,
    Cancel,
    Redirect(Vector3<f64>),
}

struct TeleportHandler {
    observer: Uuid,
    action: TeleportAction,
    calls: AtomicUsize,
    seen: Mutex<Option<(Vector3<f64>, Vector3<f64>)>>,
}

impl TeleportHandler {
    fn register(fixture: &CameraFixture, action: TeleportAction) -> Arc<Self> {
        let handler = Arc::new(Self {
            observer: fixture.observer.gameprofile.id,
            action,
            calls: AtomicUsize::new(0),
            seen: Mutex::new(None),
        });
        fixture
            .server
            .plugin_manager
            .register::<PlayerTeleportEvent, _>(handler.clone(), EventPriority::Normal, true);
        handler
    }
}

impl EventHandler<PlayerTeleportEvent> for TeleportHandler {
    fn handle_blocking<'a>(
        &'a self,
        _server: &'a Arc<Server>,
        event: &'a mut PlayerTeleportEvent,
    ) -> BoxFuture<'a, ()> {
        Box::pin(async move {
            if event.player.gameprofile.id != self.observer {
                return;
            }
            self.calls.fetch_add(1, Ordering::Relaxed);
            *self
                .seen
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner) = Some((event.from, event.to));
            match self.action {
                TeleportAction::Observe => {}
                TeleportAction::Cancel => event.cancelled = true,
                TeleportAction::Redirect(position) => event.to = position,
            }
        })
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn camera_switch_preserves_observer_rotation_and_repeating_it_is_a_noop() -> TestResult {
    with_fixture(|fixture| {
        Box::pin(async move {
            let handler = TeleportHandler::register(fixture, TeleportAction::Observe);
            fixture
                .observer
                .get_entity()
                .velocity
                .store(NONZERO_VELOCITY);
            let before = ObserverState::capture(&fixture.observer);
            let result = fixture
                .observer
                .request_spectate(Some(fixture.target.clone()));
            fixture.observer.process_camera_requests();
            assert!(result.await?);
            let changed = ObserverState::capture(&fixture.observer);
            assert_eq!(changed.position, TARGET_POSITION);
            assert_eq!(changed.last_position, TARGET_POSITION);
            assert_eq!(changed.rotation, OBSERVER_ROTATION);
            assert_eq!(changed.velocity, ZERO_VELOCITY);
            assert_eq!(changed.teleport_id, before.teleport_id + 1);
            assert_eq!(changed.epoch, before.epoch + 1);
            assert_eq!(
                changed.pending,
                Some((VarInt(changed.teleport_id), TARGET_POSITION))
            );
            assert_eq!(changed.camera_id, fixture.target.entity.entity_id);
            let packets = fixture.packets().await?;
            assert_camera_change(
                &packets,
                TARGET_POSITION,
                OBSERVER_ROTATION,
                changed.teleport_id,
                fixture.target.entity.entity_id,
            )?;

            let result = fixture
                .observer
                .request_spectate(Some(fixture.target.clone()));
            fixture.observer.process_camera_requests();
            assert!(result.await?);
            assert_eq!(ObserverState::capture(&fixture.observer), changed);
            assert_eq!(handler.calls.load(Ordering::Relaxed), 1);
            assert!(
                fixture.packets().await?.is_empty(),
                "same camera must not send any packet"
            );
            Ok(())
        })
    })
    .await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn stationary_target_rotation_follows_on_tick_and_shift_exits_after_following() -> TestResult
{
    with_fixture(|fixture| {
        Box::pin(async move {
            assert!(
                fixture
                    .observer
                    .set_camera_now(Some(fixture.target.clone()))
            );
            fixture.packets().await?;
            let attached = ObserverState::capture(&fixture.observer);
            fixture.target.entity.set_rotation(75.5, -28.25);
            fixture.observer.tick(&fixture.server);
            let followed = ObserverState::capture(&fixture.observer);
            assert_eq!(followed.position, TARGET_POSITION);
            assert_eq!(followed.rotation, (75.5, -28.25));
            assert_eq!(followed.velocity, ZERO_VELOCITY);
            assert_eq!(followed.teleport_id, attached.teleport_id);
            assert_eq!(followed.epoch, attached.epoch);
            assert_eq!(followed.camera_id, fixture.target.entity.entity_id);
            assert!(camera_packets(&fixture.packets().await?).is_empty());

            let exit_position = Vector3::new(12.25, 105.5, 7.75);
            fixture.target.entity.set_pos(exit_position);
            fixture.target.entity.set_rotation(-61.5, 32.25);
            let mut input = Vec::new();
            SPlayerInput {
                input: SPlayerInput::SNEAK,
            }
            .write_packet_data(&mut input, &CURRENT_MC_VERSION)?;
            fixture.observer.inbound_packets.push(RawPacket {
                id: SPlayerInput::to_id(CURRENT_MC_VERSION),
                payload: input.into(),
            });
            fixture.observer.tick(&fixture.server);
            let exited = ObserverState::capture(&fixture.observer);
            assert_eq!(exited.position, exit_position);
            assert_eq!(exited.rotation, (-61.5, 32.25));
            assert_eq!(exited.velocity, ZERO_VELOCITY);
            assert_eq!(exited.camera_id, fixture.observer.entity_id());
            assert_eq!(exited.teleport_id, attached.teleport_id + 1);
            assert_eq!(exited.epoch, attached.epoch + 1);
            assert_eq!(
                exited.pending,
                Some((VarInt(exited.teleport_id), exit_position))
            );
            let packets = fixture.packets().await?;
            assert_camera_change(
                &packets,
                exit_position,
                (-61.5, 32.25),
                exited.teleport_id,
                fixture.observer.entity_id(),
            )?;
            Ok(())
        })
    })
    .await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn dead_living_camera_exits_without_snapping_to_its_last_position() -> TestResult {
    with_fixture(|fixture| {
        Box::pin(async move {
            assert!(
                fixture
                    .observer
                    .set_camera_now(Some(fixture.target.clone()))
            );
            fixture.packets().await?;
            let attached = ObserverState::capture(&fixture.observer);
            // A dead living entity is still present during its death animation.
            fixture.target.health.store(0.0);
            fixture
                .target
                .entity
                .set_pos(Vector3::new(13.25, 110.5, 14.75));
            fixture.target.entity.set_rotation(-111.5, 49.25);
            assert!(!fixture.target.entity.is_removed());
            fixture.observer.tick(&fixture.server);
            let exited = ObserverState::capture(&fixture.observer);
            assert_eq!(exited.position, attached.position);
            assert_eq!(exited.rotation, attached.rotation);
            assert_eq!(exited.velocity, ZERO_VELOCITY);
            assert_eq!(exited.camera_id, fixture.observer.entity_id());
            assert_eq!(exited.teleport_id, attached.teleport_id + 1);
            assert_eq!(exited.epoch, attached.epoch + 1);
            let packets = fixture.packets().await?;
            assert_camera_change(
                &packets,
                attached.position,
                attached.rotation,
                exited.teleport_id,
                fixture.observer.entity_id(),
            )?;
            Ok(())
        })
    })
    .await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cancelled_teleport_does_not_commit_the_camera_transaction() -> TestResult {
    with_fixture(|fixture| {
        Box::pin(async move {
            fixture
                .observer
                .get_entity()
                .velocity
                .store(NONZERO_VELOCITY);
            let handler = TeleportHandler::register(fixture, TeleportAction::Cancel);
            let before = ObserverState::capture(&fixture.observer);
            let result = fixture
                .observer
                .request_spectate(Some(fixture.target.clone()));
            fixture.observer.process_camera_requests();
            assert!(!result.await?);
            assert_eq!(handler.calls.load(Ordering::Relaxed), 1);
            assert_eq!(
                *handler
                    .seen
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner),
                Some((OBSERVER_POSITION, TARGET_POSITION))
            );
            assert_eq!(ObserverState::capture(&fixture.observer), before);
            assert!(fixture.observer.camera_entity().is_none());
            assert!(fixture.packets().await?.is_empty());

            fixture
                .target
                .entity
                .set_pos(Vector3::new(14.25, 109.5, 12.75));
            fixture.target.entity.set_rotation(-121.5, 42.25);
            fixture.observer.tick(&fixture.server);
            let next_tick = ObserverState::capture(&fixture.observer);
            assert_eq!(next_tick.position, before.position);
            assert_eq!(next_tick.rotation, before.rotation);
            assert_eq!(next_tick.camera_id, before.camera_id);
            assert_eq!(next_tick.teleport_id, before.teleport_id);
            assert_eq!(next_tick.epoch, before.epoch);
            assert_eq!(next_tick.pending, before.pending);
            assert_eq!(handler.calls.load(Ordering::Relaxed), 1);
            assert!(camera_packets(&fixture.packets().await?).is_empty());
            Ok(())
        })
    })
    .await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn event_redirect_is_committed_to_position_pending_and_wire_together() -> TestResult {
    with_fixture(|fixture| {
        Box::pin(async move {
            let destination = Vector3::new(11.25, 113.5, 13.75);
            let handler = TeleportHandler::register(fixture, TeleportAction::Redirect(destination));
            let before = ObserverState::capture(&fixture.observer);
            let result = fixture
                .observer
                .request_spectate(Some(fixture.target.clone()));
            fixture.observer.process_camera_requests();
            assert!(result.await?);
            assert_eq!(handler.calls.load(Ordering::Relaxed), 1);
            assert_eq!(
                *handler
                    .seen
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner),
                Some((OBSERVER_POSITION, TARGET_POSITION))
            );
            let changed = ObserverState::capture(&fixture.observer);
            assert_eq!(changed.position, destination);
            assert_eq!(changed.last_position, destination);
            assert_eq!(
                fixture.observer.get_entity().block_pos.load(),
                BlockPos::new(11, 113, 13)
            );
            assert_eq!(
                fixture.observer.get_entity().chunk_pos.load(),
                Vector2::new(0, 0)
            );
            assert_eq!(
                changed.pending,
                Some((VarInt(changed.teleport_id), destination))
            );
            assert_eq!(changed.rotation, OBSERVER_ROTATION);
            assert_eq!(changed.velocity, ZERO_VELOCITY);
            assert_eq!(changed.teleport_id, before.teleport_id + 1);
            assert_eq!(changed.epoch, before.epoch + 1);
            assert_eq!(changed.camera_id, fixture.target.entity.entity_id);
            let packets = fixture.packets().await?;
            assert_camera_change(
                &packets,
                destination,
                OBSERVER_ROTATION,
                changed.teleport_id,
                fixture.target.entity.entity_id,
            )?;
            Ok(())
        })
    })
    .await
}
