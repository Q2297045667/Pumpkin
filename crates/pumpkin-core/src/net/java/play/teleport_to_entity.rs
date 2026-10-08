#[allow(clippy::wildcard_imports)]
use super::*;

impl JavaClient {
    pub fn handle_teleport_to_entity(
        &self,
        player: &Arc<Player>,
        packet: &STeleportToEntity,
        server: &Server,
    ) {
        if !player.is_spectator() {
            return;
        }
        for world in server.worlds.load().iter() {
            let target = world.get_entity_by_uuid(packet.target).or_else(|| {
                world
                    .get_player_by_uuid(packet.target)
                    .map(|player| player as Arc<dyn EntityBase>)
            });
            if let Some(target) = target {
                player.teleport_spectator_to(target.as_ref());
                return;
            }
        }
    }
}
