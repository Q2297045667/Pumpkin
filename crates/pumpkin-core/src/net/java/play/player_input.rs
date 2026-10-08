#[allow(clippy::wildcard_imports)]
use super::*;

impl JavaClient {
    pub fn handle_player_input(
        &self,
        player: &Arc<Player>,
        input: &SPlayerInput,
        server: &Arc<Server>,
    ) {
        let mut input_event =
            crate::plugin::api::events::player::player_input::PlayerInputEvent::new(
                player.clone(),
                format!("{:b}", input.input),
            );
        server
            .plugin_manager
            .fire_blocking(server, &mut input_event);
        if input_event.cancelled {
            return;
        }

        player.last_input.store(input.input, Ordering::Relaxed);

        if !player.has_client_loaded() {
            return;
        }
        player.update_last_action_time();
        let sneak = input.input & SPlayerInput::SNEAK != 0;
        if player.get_entity().is_sneaking() != sneak {
            send_cancellable_blocking! {{
                server;
                PlayerToggleSneakEvent::new(player.clone(), sneak);
                'after: {
                    player.get_entity().set_sneaking(event.is_sneaking);
                    if event.is_sneaking {
                        let vehicle = player
                            .get_entity()
                            .vehicle
                            .lock()
                            .unwrap_or_else(std::sync::PoisonError::into_inner)
                            .clone();
                        if let Some(vehicle) = vehicle {
                            vehicle.get_entity().remove_passenger(player.entity_id());
                        }
                    }
                }
            }}
        } else if sneak {
            let vehicle = player
                .get_entity()
                .vehicle
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .clone();
            if let Some(vehicle) = vehicle {
                vehicle.get_entity().remove_passenger(player.entity_id());
            }
        }
    }
}
