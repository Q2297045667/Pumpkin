#[allow(clippy::wildcard_imports)]
use super::*;
use pumpkin_protocol::java::server::play::SSpectateEntity;

impl JavaClient {
    pub fn handle_spectate_entity(&self, player: &Arc<Player>, packet: &SSpectateEntity) {
        if !player.has_client_loaded() || !player.is_spectator() {
            return;
        }
        player.update_last_action_time();
        let Some(target_id) = packet.target_entity_id() else {
            return;
        };
        let world = player.world();
        let Some(target) = world.get_entity_or_part(target_id) else {
            return;
        };
        let entity = target.get_entity();
        let block_pos = entity.block_pos.load().0;
        if !world
            .worldborder
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .contains(f64::from(block_pos.x), f64::from(block_pos.z))
            || !player.is_within_entity_interaction_range(&entity.bounding_box.load(), 3.0)
            || entity.is_removed()
            || !target.is_pickable()
        {
            return;
        }
        player.set_camera_now(Some(target));
    }
}
