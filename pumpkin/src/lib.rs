// Not warn event sending macros
#![allow(unused_labels)]

use crate::crash::CrashReport;
use crate::data::VanillaData;
use crate::logging::{GzipRollingLogger, PumpkinCommandCompleter, ReadlineLogWrapper};
use crate::net::bedrock::BedrockClient;
use crate::net::java::JavaClient;
use crate::net::{ClientPlatform, DisconnectReason, PacketHandlerResult};
use crate::net::{lan_broadcast::LANBroadcast, query, rcon::RCONServer};
use crate::server::{Server, ticker::Ticker};
use plugin::server::server_command::ServerCommandEvent;
use pumpkin_config::{AdvancedConfiguration, BasicConfiguration};
use pumpkin_i18n::{Locale, get_translation, resolve_server_locale};
use pumpkin_macros::send_cancellable;
use pumpkin_util::text::TextComponent;
use pumpkin_util::text::color::{Color, NamedColor};
use rustyline::Editor;
use rustyline::history::FileHistory;
use rustyline::{Config, error::ReadlineError};
use std::collections::HashMap;
use std::io::{Cursor, ErrorKind, IsTerminal, stdin};
use std::process::exit;
use std::str::FromStr;
use std::sync::atomic::{AtomicBool, AtomicI32, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::Duration;
use std::{net::SocketAddr, sync::LazyLock};
use tokio::net::{TcpListener, UdpSocket};
use tokio::select;
use tokio::sync::Mutex;
use tokio::time::sleep;
use tokio_util::sync::CancellationToken;
use tokio_util::task::TaskTracker;
use tracing::{debug, error, info, warn};
use tracing_subscriber::filter::LevelFilter;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;

pub mod block;
pub mod command;
pub mod crash;
pub mod data;
pub mod entity;
pub mod error;
pub mod item;
pub mod logging;
pub mod net;
pub mod plugin;
pub mod server;
pub mod world;

pub struct LoggingConfig {
    pub color: bool,
    pub threads: bool,
    pub timestamp: bool,
}

pub type LoggerOption = Option<(ReadlineLogWrapper, LevelFilter, LoggingConfig)>;
pub static LOGGER_IMPL: LazyLock<Arc<OnceLock<LoggerOption>>> =
    LazyLock::new(|| Arc::new(OnceLock::new()));

/// Global logging locale, resolved from `server_logging` configuration.
///
/// Initialized during [`init_logger`] before the server starts so that
/// early log messages can also respect the configured locale.
pub static SERVER_LOGGING_LOCALE: OnceLock<Locale> = OnceLock::new();

/// Returns the server logging locale, falling back to [`Locale::EnUs`].
pub fn server_locale() -> Locale {
    *SERVER_LOGGING_LOCALE.get().unwrap_or(&Locale::EnUs)
}

#[expect(clippy::print_stderr, clippy::too_many_lines)]
pub fn init_logger(advanced_config: &AdvancedConfiguration) {
    use tracing_subscriber::EnvFilter;
    use tracing_subscriber::fmt;

    // Resolve and cache the server logging locale so it's available
    // globally before the tracing subscriber is set up.
    let _ = SERVER_LOGGING_LOCALE.set(resolve_server_locale(
        &advanced_config.locale.server_logging,
    ));

    let logger = advanced_config.logging.enabled.then(|| {
        let level = std::env::var("RUST_LOG")
            .ok()
            .as_deref()
            .map(LevelFilter::from_str)
            .and_then(Result::ok)
            .unwrap_or(LevelFilter::INFO);

        let env_filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| {
            let level_str = match level {
                LevelFilter::OFF => "off",
                LevelFilter::ERROR => "error",
                LevelFilter::WARN => "warn",
                LevelFilter::INFO => "info",
                LevelFilter::DEBUG => "debug",
                LevelFilter::TRACE => "trace",
            };
            EnvFilter::new(level_str)
        });

        let file_logger: Option<GzipRollingLogger> = if advanced_config.logging.file.is_empty() {
            None
        } else {
            Some(
                GzipRollingLogger::new(level, advanced_config.logging.file.clone()).unwrap_or_else(
                    |e| {
                        panic!(
                            "{}: {e}",
                            get_translation(
                                "pumpkin:server.log.file_logger_init_failed",
                                server_locale(),
                            )
                        );
                    },
                ),
            )
        };

        let (logger, rl): (
            Box<dyn std::io::Write + Send + 'static>,
            Option<Editor<PumpkinCommandCompleter, FileHistory>>,
        ) = if advanced_config.commands.use_tty && stdin().is_terminal() {
            let rl_config = Config::builder()
                .auto_add_history(true)
                .completion_type(rustyline::CompletionType::List)
                .edit_mode(rustyline::EditMode::Emacs)
                .build();
            let helper = PumpkinCommandCompleter::new();

            match Editor::with_config(rl_config) {
                Ok(mut rl) => {
                    rl.set_helper(Some(helper));
                    (Box::new(std::io::stdout()), Some(rl))
                }
                Err(e) => {
                    let locale = server_locale();
                    let msg =
                        get_translation("pumpkin:server.log.failed_init_console_input", locale)
                            .replace("%s", &e.to_string());
                    eprintln!("{msg}");
                    (Box::new(std::io::stdout()), None)
                }
            }
        } else {
            (Box::new(std::io::stdout()), None)
        };

        let fmt_layer = fmt::layer()
            .with_writer(std::sync::Mutex::new(logger))
            .with_ansi(advanced_config.logging.color)
            .with_ansi_sanitization(false)
            .with_target(true)
            .with_thread_names(advanced_config.logging.threads)
            .with_thread_ids(advanced_config.logging.threads);

        if advanced_config.logging.timestamp {
            let local_offset =
                time::UtcOffset::current_local_offset().unwrap_or(time::UtcOffset::UTC);
            let fmt_layer = fmt_layer.with_timer(fmt::time::OffsetTime::new(
                local_offset,
                time::macros::format_description!("[year]-[month]-[day] [hour]:[minute]:[second]"),
            ));
            let registry = tracing_subscriber::registry()
                .with(env_filter)
                .with(fmt_layer);
            if let Some(file_logger) = file_logger {
                registry.with(file_logger).init();
            } else {
                registry.init();
            }
        } else {
            let fmt_layer = fmt_layer.without_time();
            let registry = tracing_subscriber::registry()
                .with(env_filter)
                .with(fmt_layer);
            if let Some(file_logger) = file_logger {
                registry.with(file_logger).init();
            } else {
                registry.init();
            }
        }

        let logging_config = LoggingConfig {
            color: advanced_config.logging.color,
            threads: advanced_config.logging.threads,
            timestamp: advanced_config.logging.timestamp,
        };

        (ReadlineLogWrapper::new(rl), level, logging_config)
    });

    assert!(
        LOGGER_IMPL.set(logger).is_ok(),
        "{}",
        get_translation(
            "pumpkin:server.log.logger_already_initialized",
            server_locale(),
        ),
    );
}

pub static SHOULD_STOP: AtomicBool = AtomicBool::new(false);
pub static STOP_INTERRUPT: LazyLock<CancellationToken> = LazyLock::new(CancellationToken::new);
pub static SERVER_IS_STOPPING: AtomicBool = AtomicBool::new(false);
pub static CRASH_REPORT: OnceLock<CrashReport> = OnceLock::new();
pub static SERVER_EXIT_CODE: AtomicI32 = AtomicI32::new(0);

pub fn stop_server() {
    SHOULD_STOP.store(true, Ordering::Relaxed);
    STOP_INTERRUPT.cancel();
}

pub fn stop_or_exit_server() {
    if SERVER_IS_STOPPING.load(Ordering::Acquire) {
        // Server is already stopping, so we forcefully exit.
        exit(SERVER_EXIT_CODE.load(Ordering::Acquire));
    } else {
        stop_server();
    }
}

fn resolve_some<T: Future, D, F: FnOnce(D) -> T>(
    opt: Option<D>,
    func: F,
) -> futures::future::Either<T, std::future::Pending<T::Output>> {
    use futures::future::Either;
    opt.map_or_else(
        || Either::Right(std::future::pending()),
        |val| Either::Left(func(val)),
    )
}

pub struct PumpkinServer {
    pub server: Arc<Server>,
    pub tcp_listener: Option<TcpListener>,
    pub udp_socket: Option<Arc<UdpSocket>>,
}

impl PumpkinServer {
    pub fn log_info(&self, message: &str) {
        tracing::info!(target: "plugin", "{}", message);
    }
    #[allow(clippy::too_many_lines)]
    pub async fn new(
        basic_config: BasicConfiguration,
        advanced_config: AdvancedConfiguration,
        vanilla_data: VanillaData,
    ) -> Self {
        let server = Server::new(basic_config, advanced_config, vanilla_data).await;

        let rcon = server.advanced_config.networking.rcon.clone();

        if rcon.enabled {
            let locale = server_locale();
            warn!(
                "{}",
                get_translation("pumpkin:server.log.rcon_insecure_warning", locale),
            );
            let rcon_server = server.clone();
            server.spawn_task(async move {
                RCONServer::run(&rcon, rcon_server).await;
            });
        }

        let tcp_listener = if server.basic_config.java_edition {
            let address = server.basic_config.java_edition_address;
            // Setup the TCP server socket.
            let listener = match TcpListener::bind(address).await {
                Ok(l) => l,
                Err(e) => {
                    let locale = server_locale();
                    match e.kind() {
                        ErrorKind::AddrInUse => {
                            error!(
                                "{}",
                                get_translation(
                                    "pumpkin:server.log.address_already_in_use",
                                    locale,
                                )
                                .replace("%s", &address.to_string()),
                            );
                            error!(
                                "{}",
                                get_translation(
                                    "pumpkin:server.log.duplicate_instance_warning",
                                    locale,
                                ),
                            );
                            std::process::exit(1);
                        }
                        ErrorKind::PermissionDenied => {
                            error!(
                                "{}",
                                get_translation(
                                    "pumpkin:server.log.bind_permission_denied",
                                    locale,
                                )
                                .replace("%s", &address.to_string()),
                            );
                            error!(
                                "{}",
                                get_translation("pumpkin:server.log.bind_privilege_hint", locale,),
                            );
                            std::process::exit(1);
                        }
                        ErrorKind::AddrNotAvailable => {
                            error!(
                                "{}",
                                get_translation(
                                    "pumpkin:server.log.address_not_available",
                                    locale,
                                )
                                .replace("%s", &address.to_string()),
                            );
                            std::process::exit(1);
                        }
                        _ => {
                            let msg =
                                get_translation("pumpkin:server.log.failed_start_tcp", locale)
                                    .replace("%s", &address.to_string())
                                    .replacen("%s", &e.to_string(), 1);
                            error!("{}", msg);
                            std::process::exit(1);
                        }
                    }
                }
            };
            // In the event the user puts 0 for their port, this will allow us to know what port it is running on
            let addr = listener.local_addr().unwrap_or_else(|e| {
                panic!(
                    "{}: {e}",
                    get_translation("pumpkin:server.log.cannot_get_address", server_locale(),)
                );
            });

            if server.advanced_config.networking.query.enabled {
                info!("Query protocol is enabled. Starting...");
                server.spawn_task(query::start_query_handler(
                    server.clone(),
                    server.advanced_config.networking.query.address,
                ));
            }

            if server.advanced_config.networking.lan_broadcast.enabled {
                info!(
                    "{}",
                    get_translation("pumpkin:server.log.lan_broadcast_enabled", server_locale(),),
                );

                let lan_broadcast = LANBroadcast::new(
                    &server.advanced_config.networking.lan_broadcast,
                    &server.basic_config,
                );
                server.spawn_task(lan_broadcast.start(addr));
            }

            Some(listener)
        } else {
            None
        };

        // Ticker
        {
            let ticker_server = server.clone();
            server.spawn_task(async move {
                Ticker::run(&ticker_server).await;
            });
        };

        let udp_socket = if server.basic_config.bedrock_edition {
            Some(Arc::new(
                UdpSocket::bind(server.basic_config.bedrock_edition_address)
                    .await
                    .unwrap_or_else(|e| {
                        panic!(
                            "{}: {e}",
                            get_translation("pumpkin:server.log.udp_bind_failed", server_locale(),)
                        );
                    }),
            ))
        } else {
            None
        };

        Self {
            server,
            tcp_listener,
            udp_socket,
        }
    }

    pub async fn init_plugins(&self) -> std::time::Duration {
        self.server
            .plugin_manager
            .set_self_ref(self.server.plugin_manager.clone())
            .await;
        self.server
            .plugin_manager
            .set_server(self.server.clone())
            .await;
        match self.server.plugin_manager.load_plugins().await {
            Ok(duration) => duration,
            Err(err) => {
                error!("{err}");
                std::time::Duration::ZERO
            }
        }
    }

    pub async fn unload_plugins(&self) {
        let locale = server_locale();
        if let Err(err) = self.server.plugin_manager.unload_all_plugins().await {
            let msg = get_translation("pumpkin:server.log.error_unloading_plugins", locale)
                .replace("%s", &err.to_string());
            error!("{}", msg);
        } else {
            info!(
                "{}",
                get_translation("pumpkin:server.log.plugins_unloaded_successfully", locale,),
            );
        }
    }

    pub async fn start(&self) {
        if self.server.advanced_config.commands.use_console
            && let Some((wrapper, _, _)) = LOGGER_IMPL.wait()
        {
            if let Some(rl) = wrapper.take_readline() {
                setup_console(rl, self.server.clone());
            } else {
                if self.server.advanced_config.commands.use_tty {
                    warn!(
                        "{}",
                        get_translation("pumpkin:server.log.input_not_tty", server_locale(),),
                    );
                }
                setup_stdin_console(self.server.clone());
            }
        }

        let tasks = Arc::new(TaskTracker::new());
        let mut master_client_id: u64 = 0;
        let bedrock_clients = Arc::new(Mutex::new(HashMap::new()));

        while !SHOULD_STOP.load(Ordering::Relaxed) {
            if !self
                .unified_listener_task(&mut master_client_id, &tasks, &bedrock_clients)
                .await
            {
                break;
            }
        }

        SERVER_IS_STOPPING.store(true, Ordering::Release);

        if let Some(crash_report) = CRASH_REPORT.get() {
            crash_report.print_to_console();
            crash_report.save_and_log();

            let locale = server_locale();
            info!(
                "{}",
                TextComponent::text(get_translation(
                    "pumpkin:server.shutdown.graceful_shutdown",
                    locale,
                ))
                .color(Color::Named(NamedColor::Green))
                .to_pretty_console()
            );

            SERVER_EXIT_CODE.store(1, Ordering::Release);
        }

        let locale = server_locale();
        info!(
            "{}",
            get_translation("pumpkin:server.log.stopped_accepting_connections", locale,),
        );

        if let Err(e) = self
            .server
            .player_data_storage
            .save_all_players(&self.server)
            .await
        {
            let msg = get_translation("pumpkin:server.log.error_saving_all_players", locale)
                .replace("%s", &e.to_string());
            error!("{}", msg);
        }

        let kick_message = TextComponent::text(get_translation(
            "pumpkin:server.shutdown.server_stopped_kick",
            locale,
        ));
        for player in self.server.get_all_players() {
            player
                .kick(DisconnectReason::Shutdown, kick_message.clone())
                .await;
        }

        info!(
            "{}",
            get_translation("pumpkin:server.log.ending_player_tasks", locale),
        );

        tasks.close();
        tasks.wait().await;

        self.unload_plugins().await;

        info!(
            "{}",
            get_translation("pumpkin:server.log.starting_save", locale),
        );

        self.server.shutdown().await;

        info!(
            "{}",
            get_translation("pumpkin:server.log.completed_save", locale),
        );

        if let Some((wrapper, _, _)) = LOGGER_IMPL.wait()
            && let Some(rl) = wrapper.take_readline()
        {
            let _ = rl;
        }
    }

    #[allow(clippy::too_many_lines)]
    pub async fn unified_listener_task(
        &self,
        master_client_id_counter: &mut u64,
        tasks: &Arc<TaskTracker>,
        bedrock_clients: &Arc<Mutex<HashMap<SocketAddr, Arc<BedrockClient>>>>,
    ) -> bool {
        let mut udp_buf = [0; 1496]; // Buffer for UDP receive

        select! {
            // Branch for TCP connections (Java Edition)
            tcp_result = resolve_some(self.tcp_listener.as_ref(), tokio::net::TcpListener::accept) => {
                match tcp_result {
                    Ok((connection, client_addr)) => {
                        if let Err(e) = connection.set_nodelay(true) {
                            let msg = get_translation(
                                "pumpkin:server.log.failed_set_tcp_nodelay",
                                server_locale(),
                            )
                            .replace("%s", &e.to_string());
                            warn!("{}", msg);
                        }

                        let client_id = *master_client_id_counter;
                        *master_client_id_counter += 1;

                        let formatted_address = if self.server.basic_config.scrub_ips {
                            scrub_address(&format!("{client_addr}"))
                        } else {
                            format!("{client_addr}")
                        };
                        let msg = get_translation(
                            "pumpkin:server.log.accepted_java_connection",
                            server_locale(),
                        )
                        .replace("%s", &formatted_address)
                        .replacen("%s", &client_id.to_string(), 1);
                        debug!("{}", msg);
                        let server_clone = self.server.clone();

                        tasks.spawn(async move {
                            let mut java_client = JavaClient::new(connection, client_addr, client_id);
                            java_client.start_outgoing_packet_task();
                            let login_result = java_client.handle_login_sequence(&server_clone).await;

                            match login_result {
                                PacketHandlerResult::Stop => {
                                     java_client.close();
                                     java_client.await_tasks().await;
                                },
                                PacketHandlerResult::ReadyToPlay(profile, config) => {
                                     pumpkin_i18n::set_player_locale(
                                         &profile.id.to_string(),
                                         &config.locale,
                                         &server_clone.advanced_config.locale.client_java_edition,
                                     );
                                     if let Some((player, world)) = server_clone
                                     .add_player(ClientPlatform::Java(java_client), profile, Some(config))
                                          .await
                                {
                                    world
                                        .spawn_java_player(&server_clone.basic_config, &player, &server_clone)
                                        .await;
                                    if let ClientPlatform::Java(client) = &player.client {
                                        *client.player.lock().await = Some(player.clone());
                                        client.progress_player_packets(&player, &server_clone).await;

                                        // Close when done
                                        client.close();
                                        client.await_tasks().await;
                                    }
                                    player.remove().await;
                                    server_clone.remove_player(&player).await;
                                    pumpkin_i18n::remove_player_locale(&player.gameprofile.id.to_string());
                                    if let Err(e) = server_clone.player_data_storage
                                        .handle_player_leave(&player)
                                        .await {
                                            let save_err_msg = get_translation(
                                                "pumpkin:server.log.failed_save_player_disconnect",
                                                server_locale(),
                                            )
                                            .replace("%s", &e.to_string());
                                            error!("{}", save_err_msg);
                                        }
                                    }
                                },
                            }
                        });
                    }
                    Err(e) => {
                        let msg = get_translation(
                            "pumpkin:server.log.failed_accept_java_connection",
                            server_locale(),
                        )
                        .replace("%s", &e.to_string());
                        error!("{}", msg);
                        sleep(Duration::from_millis(50)).await;
                    }
                }
            },

            // Branch for UDP packets (Bedrock Edition)
            udp_result = resolve_some(self.udp_socket.as_ref(), |sock: &Arc<UdpSocket>| sock.recv_from(&mut udp_buf)) => {
                match udp_result {
                    Ok((len, client_addr)) => {
                        if len > 0 {
                            let id = udp_buf[0];
                            let is_online = id & 128 != 0;

                            if is_online {
                                let be_clients = bedrock_clients.clone();
                                let mut clients_guard = bedrock_clients.lock().await;

                                if clients_guard
                                    .get(&client_addr)
                                    .is_some_and(|client| client.is_closed())
                                {
                                    clients_guard.remove(&client_addr);
                                }

                                let mut is_new = false;
                                let client = clients_guard.entry(client_addr).or_insert_with(|| {
                                    is_new = true;
                                    *master_client_id_counter += 1;

                                    let new_client = Arc::new(BedrockClient::new(
                                        self.udp_socket.as_ref().unwrap().clone(),
                                        client_addr,
                                        be_clients
                                    ));

                                    new_client.start_outgoing_packet_task();
                                    new_client
                                }).clone();

                                if is_new {
                                    let server_clone = self.server.clone();
                                    let client_clone = client.clone();
                                    tasks.spawn(async move {
                                        let login_result = client_clone.handle_login_sequence(&server_clone).await;

                                         match login_result {
                                            PacketHandlerResult::Stop => {
                                                client_clone.close().await;
                                                client_clone.await_tasks().await;
                                            }
                                            PacketHandlerResult::ReadyToPlay(profile, config) => {
                                                pumpkin_i18n::set_player_locale(
                                                    &profile.id.to_string(),
                                                    &config.locale,
                                                    &server_clone.advanced_config.locale.client_bedrock_edition,
                                                );
                                                if let Some((player, _world)) = server_clone
                                                    .add_player(ClientPlatform::Bedrock(client_clone.clone()), profile, Some(config))
                                                    .await
                                                {
                                                    *client_clone.player.lock().await = Some(player.clone());

                                                    client_clone.progress_player_packets(&player, &server_clone).await;

                                                    client_clone.close().await;
                                                    client_clone.await_tasks().await;
                                                    player.remove().await;
                                                    server_clone.remove_player(&player).await;
                                                    pumpkin_i18n::remove_player_locale(&player.gameprofile.id.to_string());
                                                    if let Err(e) = server_clone.player_data_storage
                                                        .handle_player_leave(&player)
                                                        .await {
                                                            let save_err_msg = get_translation(
                                                "pumpkin:server.log.failed_save_player_disconnect",
                                                server_locale(),
                                            )
                                            .replace("%s", &e.to_string());
                                            error!("{}", save_err_msg);
                                                        }
                                                }
                                            }
                                        }
                                    });
                                }

                                let packet_bytes = udp_buf[..len].to_vec();
                                let server = self.server.clone();

                                tasks.spawn(async move {
                                    client.process_packet(&server, packet_bytes.into()).await;
                                });
                            } else if let Some(sock) = self.udp_socket.as_ref() {
                                let _ = BedrockClient::handle_offline_packet(
                                    &self.server,
                                    id,
                                    &mut Cursor::new(&udp_buf[1..len]),
                                    client_addr,
                                    sock,
                                    bedrock_clients,
                                ).await;
                            }
                        }
                    }
                    Err(e) => {
                        let msg = get_translation(
                            "pumpkin:server.log.udp_socket_error",
                            server_locale(),
                        )
                        .replace("%s", &e.to_string());
                        error!("{}", msg);
                    }
                }
            },

            // Branch for the global stop signal
            () = STOP_INTERRUPT.cancelled() => {
                return false;
            }
        }
        true
    }
}

fn setup_stdin_console(server: Arc<Server>) {
    let (tx, mut rx) = tokio::sync::mpsc::channel(1);
    let rt = tokio::runtime::Handle::current();
    std::thread::spawn(move || {
        while !SHOULD_STOP.load(Ordering::Relaxed) {
            let mut line = String::new();
            if let Ok(size) = stdin().read_line(&mut line) {
                // if no bytes were read, we may have hit EOF
                if size == 0 {
                    break;
                }
            } else {
                break;
            }
            if line.is_empty() || line.as_bytes()[line.len() - 1] != b'\n' {
                warn!(
                    "{}",
                    get_translation("pumpkin:server.log.console_no_newline", server_locale(),),
                );
            }
            rt.block_on(tx.send(line.trim().to_string()))
                .unwrap_or_else(|e| {
                    panic!(
                        "{}: {e}",
                        get_translation("pumpkin:server.log.failed_send_command", server_locale(),)
                    );
                });
        }
    });
    tokio::spawn(async move {
        while !SHOULD_STOP.load(Ordering::Relaxed) {
            if let Some(command) = rx.recv().await {
                send_cancellable! {{
                    &server;
                    ServerCommandEvent::new(command.clone());

                    'after: {
                        server.command_dispatcher.read().await
                            .handle_command(&command::CommandSender::Console.into_source(&server).await, command.as_str())
                            .await;
                    };
                }}
            }
        }
    });
}

fn setup_console(mut rl: Editor<PumpkinCommandCompleter, FileHistory>, server: Arc<Server>) {
    let (tx, mut rx) = tokio::sync::mpsc::channel(1);
    let (tx_reply, mut rx_reply) = tokio::sync::mpsc::channel(1);

    if let Some(helper) = rl.helper_mut() {
        if let Ok(mut server_lock) = helper.server.write() {
            *server_lock = Some(server.clone());
        }
        let _ = helper.rt.set(tokio::runtime::Handle::current());
    }

    std::thread::spawn(move || {
        while !SHOULD_STOP.load(Ordering::Relaxed) {
            let readline = rl.readline("$ ");
            match readline {
                Ok(line) => {
                    let _ = rl.add_history_entry(line.clone());
                    if tx.blocking_send(line).is_err() {
                        break;
                    }

                    // Wait for the command to be fully processed before continuing
                    let _ = rx_reply.blocking_recv();
                }
                Err(ReadlineError::Interrupted) => {
                    info!("CTRL-C");
                    stop_or_exit_server();
                    break;
                }
                Err(ReadlineError::Eof) => {
                    info!("CTRL-D");
                    stop_server();
                    break;
                }
                Err(err) => {
                    let msg = get_translation(
                        "pumpkin:server.log.error_reading_console",
                        server_locale(),
                    )
                    .replace("%s", &err.to_string());
                    error!("{}", msg);
                    break;
                }
            }
        }
        if let Some((wrapper, _, _)) = LOGGER_IMPL.wait() {
            wrapper.return_readline(rl);
        }
    });

    server.clone().spawn_task(async move {
        while !SHOULD_STOP.load(Ordering::Relaxed) {
            let t1 = rx.recv();
            let t2 = STOP_INTERRUPT.cancelled();

            let result = select! {
                line = t1 => line,
                () = t2 => None,
            };

            if let Some(line) = result {
                send_cancellable! {{
                    &server;
                    ServerCommandEvent::new(line.clone());

                    'after: {
                        server.command_dispatcher.read().await
                            .handle_command(&command::CommandSender::Console.into_source(&server).await, &line)
                            .await;

                        let _ = tx_reply.send(1).await;
                    }
                }}
            } else {
                break;
            }
        }
        drop(rx);
        debug!("Stopped console commands task");
    });
}

fn scrub_address(ip: &str) -> String {
    ip.chars()
        .map(|ch| if ch == '.' || ch == ':' { ch } else { 'x' })
        .collect()
}
