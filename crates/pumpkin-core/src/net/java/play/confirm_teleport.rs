#[allow(clippy::wildcard_imports)]
use super::*;
use pumpkin_util::version::JavaMinecraftVersion;

#[derive(Debug, PartialEq)]
enum TeleportConfirmation {
    Ignore,
    Invalid,
    Accepted(Vector3<f64>),
}

fn accept_teleport(
    latest_id: i32,
    awaiting: &mut Option<(VarInt, Vector3<f64>)>,
    packet: &SConfirmTeleport,
    version: JavaMinecraftVersion,
) -> TeleportConfirmation {
    if packet.teleport_id.0 != latest_id {
        return TeleportConfirmation::Ignore;
    }
    let Some((id, position)) = awaiting.as_ref() else {
        return TeleportConfirmation::Invalid;
    };
    if *id != packet.teleport_id {
        return TeleportConfirmation::Ignore;
    }
    if version >= JavaMinecraftVersion::V_26_3
        && (packet.position.x.is_nan()
            || packet.position.y.is_nan()
            || packet.position.z.is_nan()
            || !packet.yaw.is_finite()
            || !packet.pitch.is_finite())
    {
        return TeleportConfirmation::Invalid;
    }
    let position = *position;
    *awaiting = None;
    TeleportConfirmation::Accepted(position)
}

impl JavaClient {
    pub fn handle_confirm_teleport(
        &self,
        player: &Arc<Player>,
        server: &Arc<Server>,
        packet: &SConfirmTeleport,
    ) {
        let confirmation = {
            let mut awaiting = player
                .awaiting_teleport
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            accept_teleport(
                player.teleport_id_count.load(Ordering::Relaxed),
                &mut awaiting,
                packet,
                self.version.load(),
            )
        };
        match confirmation {
            TeleportConfirmation::Ignore => {}
            TeleportConfirmation::Invalid => {
                self.try_kick(&TextComponent::translate_cross(
                    translation::java::MULTIPLAYER_DISCONNECT_INVALID_PLAYER_MOVEMENT,
                    translation::java::MULTIPLAYER_DISCONNECT_INVALID_PLAYER_MOVEMENT,
                    [],
                ));
            }
            TeleportConfirmation::Accepted(position) => {
                player.get_entity().set_pos(position);
                if self.version.load() >= JavaMinecraftVersion::V_26_3 {
                    self.handle_position_rotation_now(
                        player,
                        server,
                        &SPlayerPositionRotation {
                            position: Vector3::new(
                                Self::clamp_horizontal(packet.position.x),
                                Self::clamp_vertical(packet.position.y),
                                Self::clamp_horizontal(packet.position.z),
                            ),
                            yaw: packet.yaw,
                            pitch: packet.pitch,
                            collision: 0,
                        },
                    );
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn confirmation(id: i32) -> SConfirmTeleport {
        SConfirmTeleport {
            teleport_id: VarInt(id),
            position: Vector3::new(1.0, 2.0, 3.0),
            yaw: 30.0,
            pitch: 10.0,
        }
    }

    #[test]
    fn stale_confirmation_preserves_the_newer_teleport() {
        let position = Vector3::new(4.0, 5.0, 6.0);
        let mut awaiting = Some((VarInt(2), position));
        assert_eq!(
            accept_teleport(
                2,
                &mut awaiting,
                &confirmation(1),
                JavaMinecraftVersion::V_26_3
            ),
            TeleportConfirmation::Ignore
        );
        assert_eq!(awaiting, Some((VarInt(2), position)));
        assert_eq!(
            accept_teleport(
                2,
                &mut awaiting,
                &confirmation(2),
                JavaMinecraftVersion::V_26_3
            ),
            TeleportConfirmation::Accepted(position)
        );
        assert!(awaiting.is_none());
    }

    #[test]
    fn invalid_echo_is_checked_only_for_the_current_teleport() {
        let mut packet = confirmation(1);
        packet.position.x = f64::NAN;
        let mut awaiting = Some((VarInt(2), Vector3::new(4.0, 5.0, 6.0)));
        assert_eq!(
            accept_teleport(2, &mut awaiting, &packet, JavaMinecraftVersion::V_26_3),
            TeleportConfirmation::Ignore
        );
        packet.teleport_id = VarInt(2);
        assert_eq!(
            accept_teleport(2, &mut awaiting, &packet, JavaMinecraftVersion::V_26_3),
            TeleportConfirmation::Invalid
        );
        assert!(awaiting.is_some());
    }
}
