use super::Player;
use crate::{entity::EntityBase, world::World};
use pumpkin_protocol::java::client::play::CSetCamera;
use pumpkin_util::math::{boundingbox::BoundingBox, vector3::Vector3};
use std::sync::{Arc, Weak, atomic::Ordering};
use tokio::sync::oneshot;

#[cfg(test)]
mod tests;

pub(super) enum CameraRequest {
    Set {
        target: Option<Weak<dyn EntityBase>>,
        spectator_only: bool,
        completion: Option<oneshot::Sender<bool>>,
    },
    ChangeWorld {
        world: Weak<World>,
        position: Vector3<f64>,
        yaw: f32,
        pitch: f32,
        expected_target: Weak<dyn EntityBase>,
    },
}

pub(super) struct CameraCompletion {
    target: Option<Weak<dyn EntityBase>>,
    completion: Option<oneshot::Sender<bool>>,
}

impl Player {
    pub(crate) fn is_within_entity_interaction_range(
        &self,
        bounds: &BoundingBox,
        buffer: f64,
    ) -> bool {
        let range = self
            .living_entity
            .get_attribute_value(&pumpkin_data::attributes::Attributes::ENTITY_INTERACTION_RANGE)
            + buffer;
        bounds.squared_magnitude(self.eye_position()) < range * range
    }

    pub(crate) fn teleport_spectator_to(&self, target: &dyn EntityBase) {
        if !self.set_camera_now(None) {
            return;
        }
        let entity = target.get_entity();
        let world = entity.world.load_full();
        let position = entity.pos.load();
        let yaw = entity.yaw.load();
        let pitch = entity.pitch.load();
        if Arc::ptr_eq(&world, &self.world()) {
            let Some(position) = self.teleport_destination(position) else {
                return;
            };
            if let Some(vehicle) = self.get_entity().get_vehicle() {
                vehicle
                    .get_entity()
                    .remove_passenger_before_teleport(self.entity_id());
                if self.get_entity().has_vehicle() {
                    return;
                }
            }
            self.get_entity()
                .velocity
                .store(Vector3::new(0.0, 0.0, 0.0));
            self.send_teleport_now(position, yaw, pitch);
        } else {
            self.start_camera_world_change(world, position, yaw, pitch, None, None);
        }
    }

    pub(crate) fn request_spectate(
        &self,
        target: Option<Arc<dyn EntityBase>>,
    ) -> oneshot::Receiver<bool> {
        let (completion, result) = oneshot::channel();
        self.camera_requests.push(CameraRequest::Set {
            target: target.map(|target| Arc::downgrade(&target)),
            spectator_only: true,
            completion: Some(completion),
        });
        result
    }

    pub(crate) fn request_camera_reset(&self) {
        self.camera_requests.push(CameraRequest::Set {
            target: None,
            spectator_only: false,
            completion: None,
        });
    }

    pub(crate) fn camera_entity(&self) -> Option<Arc<dyn EntityBase>> {
        self.camera_target
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .as_ref()
            .and_then(Weak::upgrade)
    }

    fn camera_is_same(&self, target: Option<&Arc<dyn EntityBase>>) -> bool {
        let current = self.camera_entity();
        match (current.as_ref(), target) {
            (Some(current), Some(target)) => Arc::ptr_eq(current, target),
            (None, None) => self.camera_target_id.load().is_none(),
            _ => false,
        }
    }

    fn finish_camera_change(&self, target: Option<&Arc<dyn EntityBase>>) {
        let id = target
            .as_ref()
            .map_or_else(|| self.entity_id(), |target| target.get_entity().entity_id);
        *self
            .camera_target
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = target.map(Arc::downgrade);
        self.camera_target_id
            .store((id != self.entity_id()).then_some(id));
        if target.is_some()
            && let Some(player) = self.world().get_player_by_uuid(self.gameprofile.id)
        {
            crate::world::chunker::update_position(&player);
        }
        self.try_send_client_packet(&CSetCamera::new(id.into()));
    }

    pub(crate) fn set_camera_now(&self, target: Option<Arc<dyn EntityBase>>) -> bool {
        let target = target.filter(|target| !std::ptr::eq(target.get_entity(), self.get_entity()));
        if self.camera_is_same(target.as_ref()) {
            return true;
        }
        if self.get_entity().is_removed() {
            return false;
        }
        if target.as_ref().is_some_and(|target| {
            !Arc::ptr_eq(&target.get_entity().world.load_full(), &self.world())
        }) {
            return false;
        }
        let position = target
            .as_ref()
            .map_or_else(|| self.position(), |target| target.get_entity().pos.load());
        let entity = self.get_entity();
        let yaw = entity.yaw.load();
        let pitch = entity.pitch.load();
        let Some(position) = self.teleport_destination(position) else {
            return false;
        };
        if let Some(vehicle) = entity.get_vehicle() {
            vehicle
                .get_entity()
                .remove_passenger_before_teleport(self.entity_id());
            if entity.has_vehicle() {
                return false;
            }
        }
        entity.velocity.store(Vector3::new(0.0, 0.0, 0.0));
        self.send_teleport_now(position, yaw, pitch);
        self.finish_camera_change(target.as_ref());
        true
    }

    pub(crate) fn process_camera_requests(&self) {
        if self.camera_world_change.load(Ordering::Relaxed) {
            let completed = self
                .camera_completion
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .take();
            let Some(CameraCompletion { target, completion }) = completed else {
                return;
            };
            self.camera_world_change.store(false, Ordering::Relaxed);
            let accepted = target.and_then(|target| target.upgrade()).map_or_else(
                || self.set_camera_now(None),
                |target| {
                    self.get_entity()
                        .velocity
                        .store(Vector3::new(0.0, 0.0, 0.0));
                    self.finish_camera_change(Some(&target));
                    true
                },
            );
            if let Some(completion) = completion {
                let _ = completion.send(accepted);
            }
        }
        while !self.camera_world_change.load(Ordering::Relaxed) {
            let Some(request) = self.camera_requests.pop() else {
                break;
            };
            match request {
                CameraRequest::Set {
                    target,
                    spectator_only,
                    completion,
                } => {
                    if spectator_only && !self.is_spectator() {
                        if let Some(completion) = completion {
                            let _ = completion.send(false);
                        }
                        continue;
                    }
                    let target = match target {
                        Some(target) => {
                            let Some(target) = target.upgrade() else {
                                if let Some(completion) = completion {
                                    let _ = completion.send(false);
                                }
                                continue;
                            };
                            Some(target)
                        }
                        None => None,
                    };
                    if let Some(target) = target.as_ref()
                        && !Arc::ptr_eq(&target.get_entity().world.load_full(), &self.world())
                    {
                        self.start_camera_world_change(
                            target.get_entity().world.load_full(),
                            target.get_entity().pos.load(),
                            self.get_entity().yaw.load(),
                            self.get_entity().pitch.load(),
                            Some(Arc::downgrade(target)),
                            completion,
                        );
                    } else {
                        let accepted = self.set_camera_now(target);
                        if let Some(completion) = completion {
                            let _ = completion.send(accepted);
                        }
                    }
                }
                CameraRequest::ChangeWorld {
                    world,
                    position,
                    yaw,
                    pitch,
                    expected_target,
                } => {
                    if let (Some(world), Some(expected), Some(current)) = (
                        world.upgrade(),
                        expected_target.upgrade(),
                        self.camera_entity(),
                    ) && Arc::ptr_eq(&current, &expected)
                    {
                        self.start_camera_world_change(world, position, yaw, pitch, None, None);
                    }
                }
            }
        }
    }

    fn start_camera_world_change(
        &self,
        world: Arc<World>,
        position: Vector3<f64>,
        yaw: f32,
        pitch: f32,
        target: Option<Weak<dyn EntityBase>>,
        completion: Option<oneshot::Sender<bool>>,
    ) {
        let Some(player) = self.world().get_player_by_uuid(self.gameprofile.id) else {
            if let Some(completion) = completion {
                let _ = completion.send(false);
            }
            return;
        };
        self.camera_world_change.store(true, Ordering::Relaxed);
        let task_player = player.clone();
        if player
            .spawn_task(async move {
                let accepted = task_player
                    .teleport_world_inner(world, position, Some(yaw), Some(pitch))
                    .await;
                if accepted {
                    *task_player
                        .camera_completion
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner) =
                        Some(CameraCompletion { target, completion });
                } else {
                    task_player
                        .camera_world_change
                        .store(false, Ordering::Relaxed);
                    if let Some(completion) = completion {
                        let _ = completion.send(false);
                    }
                }
            })
            .is_none()
        {
            self.camera_world_change.store(false, Ordering::Relaxed);
        }
    }

    pub(crate) fn tick_camera(&self) {
        if self.camera_world_change.load(Ordering::Relaxed)
            || self.camera_target_id.load().is_none()
        {
            return;
        }
        let Some(target) = self.camera_entity() else {
            self.set_camera_now(None);
            return;
        };
        let entity = target.get_entity();
        let alive = entity.is_alive()
            && target
                .get_living_entity()
                .is_none_or(|living| living.health.load() > 0.0);
        if !alive {
            self.set_camera_now(None);
            return;
        }
        self.get_entity().set_pos(entity.pos.load());
        self.get_entity()
            .set_rotation(entity.yaw.load(), entity.pitch.load());
        if let Some(player) = self.world().get_player_by_uuid(self.gameprofile.id) {
            crate::world::chunker::update_position(&player);
        }
        if self.get_entity().is_sneaking() {
            self.set_camera_now(None);
        }
    }
}
