use pumpkin_protocol::java::client::play::{
    CInitializeWorldBorder, CSetBorderCenter, CSetBorderLerpSize, CSetBorderSize,
    CSetBorderWarningDelay, CSetBorderWarningDistance,
};

use crate::net::java::JavaClient;

use super::World;

pub struct Worldborder {
    pub center_x: f64,
    pub center_z: f64,
    pub old_diameter: f64,
    pub new_diameter: f64,
    pub speed: i64,
    pub portal_teleport_boundary: i32,
    pub warning_blocks: i32,
    pub warning_time: i32,
    pub damage_per_block: f32,
    pub buffer: f32,
    size: f64,
    previous_size: f64,
    lerp_duration: i64,
}

impl Worldborder {
    pub const MAX_SIZE: f64 = 5.999997E7f32 as f64;
    pub const MAX_CENTER_COORDINATE: f64 = 29_999_984.0;
    pub const MILLIS_PER_TICK: u64 = 50;

    #[must_use]
    pub const fn new(
        x: f64,
        z: f64,
        diameter: f64,
        speed: i64,
        warning_blocks: i32,
        warning_time: i32,
    ) -> Self {
        Self {
            center_x: x,
            center_z: z,
            old_diameter: diameter,
            new_diameter: diameter,
            speed,
            portal_teleport_boundary: Self::MAX_CENTER_COORDINATE as i32,
            warning_blocks,
            warning_time,
            damage_per_block: 0.2,
            buffer: 5.0,
            size: diameter,
            previous_size: diameter,
            lerp_duration: speed,
        }
    }

    pub fn init_client(&self, client: &JavaClient) {
        if let Ok(data) = client.serialize_packet(&CInitializeWorldBorder::new(
            self.center_x,
            self.center_z,
            self.get_size(),
            self.new_diameter,
            self.speed.into(),
            self.portal_teleport_boundary.into(),
            self.warning_blocks.into(),
            self.warning_time.into(),
        )) {
            client.try_enqueue_packet(data);
        }
    }

    pub fn set_center(&mut self, world: &World, x: f64, z: f64) {
        self.center_x = x;
        self.center_z = z;
        world.broadcast_packet_all(&CSetBorderCenter::new(x, z));
    }

    pub fn set_diameter(&mut self, world: &World, diameter: f64, speed: Option<i64>) {
        let from = self.get_size();
        let ticks = speed.unwrap_or(0);
        self.lerp_size_between(from, diameter, ticks);
        if ticks > 0 {
            world.broadcast_packet_all(&CSetBorderLerpSize::new(from, diameter, ticks.into()));
        } else {
            world.broadcast_packet_all(&CSetBorderSize::new(diameter));
        }
    }

    fn lerp_size_between(&mut self, from: f64, to: f64, ticks: i64) {
        self.old_diameter = from;
        self.new_diameter = to;
        self.speed = if from == to { 0 } else { ticks.max(0) };
        self.lerp_duration = self.speed;
        self.size = if self.speed > 0 { from } else { to };
        self.previous_size = self.size;
    }

    pub fn tick(&mut self) {
        if self.speed > 0 {
            self.previous_size = self.size;
            self.speed -= 1;
            if self.speed == 0 {
                self.size = self.new_diameter;
                self.previous_size = self.size;
            } else {
                let progress = (self.lerp_duration - self.speed) as f64 / self.lerp_duration as f64;
                self.size = self.old_diameter + progress * (self.new_diameter - self.old_diameter);
            }
        }
    }

    #[must_use]
    pub const fn get_size(&self) -> f64 {
        self.size
    }

    pub fn add_diameter(&mut self, world: &World, offset: f64, speed: Option<i64>) {
        self.set_diameter(world, self.get_size() + offset, speed);
    }

    pub fn set_warning_delay(&mut self, world: &World, delay: i32) {
        self.warning_time = delay;
        world.broadcast_packet_all(&CSetBorderWarningDelay::new(delay.into()));
    }

    pub fn set_warning_distance(&mut self, world: &World, distance: i32) {
        self.warning_blocks = distance;
        world.broadcast_packet_all(&CSetBorderWarningDistance::new(distance.into()));
    }

    pub const fn set_damage_buffer(&mut self, buffer: f32) {
        self.buffer = buffer;
    }

    pub const fn set_damage_per_block(&mut self, damage: f32) {
        self.damage_per_block = damage;
    }

    pub fn reset(&mut self, world: &World) {
        *self = Self::new(0.0, 0.0, Self::MAX_SIZE, 0, 5, 300);
        world.broadcast_packet_all(&CInitializeWorldBorder::new(
            self.center_x,
            self.center_z,
            self.size,
            self.new_diameter,
            self.speed.into(),
            self.portal_teleport_boundary.into(),
            self.warning_blocks.into(),
            self.warning_time.into(),
        ));
    }

    fn bounds(&self) -> (f64, f64, f64, f64) {
        // Server-side vanilla queries use partial tick zero, hence the previous moving extent.
        let size = if self.speed > 0 {
            self.previous_size
        } else {
            self.size
        };
        let half = size / 2.0;
        let max = f64::from(self.portal_teleport_boundary);
        (
            (self.center_x - half).clamp(-max, max),
            (self.center_x + half).clamp(-max, max),
            (self.center_z - half).clamp(-max, max),
            (self.center_z + half).clamp(-max, max),
        )
    }

    #[must_use]
    pub fn contains(&self, x: f64, z: f64) -> bool {
        let (min_x, max_x, min_z, max_z) = self.bounds();
        x >= min_x && x < max_x && z >= min_z && z < max_z
    }

    #[must_use]
    pub fn contains_block(&self, x: i32, z: i32) -> bool {
        self.contains(f64::from(x), f64::from(z))
            && self.contains(f64::from(x + 1), f64::from(z + 1))
    }

    #[must_use]
    pub fn clamp_block(&self, x: i32, z: i32) -> (i32, i32) {
        let (min_x, max_x, min_z, max_z) = self.bounds();
        // Preserve the portal fallback for borders that span no complete block boundary.
        let min_x = min_x.floor() as i32;
        let max_x = (max_x.floor() as i32 - 1).max(min_x);
        let min_z = min_z.floor() as i32;
        let max_z = (max_z.floor() as i32 - 1).max(min_z);
        (x.clamp(min_x, max_x), z.clamp(min_z, max_z))
    }
}

#[cfg(test)]
mod tests {
    use super::Worldborder;

    fn centered_border(diameter: f64) -> Worldborder {
        Worldborder::new(0.0, 0.0, diameter, 0, 5, 300)
    }

    #[test]
    fn a_zero_width_border_contains_no_block() {
        let border = centered_border(0.0);
        assert!(!border.contains_block(0, 0));
        assert!((-32..=32).all(|x| (-32..=32).all(|z| !border.contains_block(x, z))));
    }

    #[test]
    fn clamp_block_handles_a_zero_width_border() {
        let border = centered_border(0.0);
        assert_eq!(border.clamp_block(0, 0), (0, 0));
        assert_eq!(border.clamp_block(100, -100), (0, 0));
    }

    #[test]
    fn clamp_block_handles_a_border_narrower_than_one_block() {
        let border = Worldborder::new(0.5, 0.5, 0.5, 0, 5, 300);
        assert_eq!(border.clamp_block(100, -100), (0, 0));
    }

    #[test]
    fn clamp_block_still_clamps_a_normal_border() {
        let border = centered_border(10.0);
        assert_eq!(border.clamp_block(100, 100), (4, 4));
        assert_eq!(border.clamp_block(-100, -100), (-5, -5));
        assert_eq!(border.clamp_block(2, -3), (2, -3));
    }

    #[test]
    fn moving_border_uses_tick_progress_and_previous_extent() {
        let mut border = centered_border(10.0);
        border.lerp_size_between(10.0, 20.0, 4);
        assert!(!border.contains(6.0, 0.0));
        border.tick();
        assert_eq!(border.get_size(), 12.5);
        assert!(!border.contains(6.0, 0.0));
        border.tick();
        assert!(border.contains(6.0, 0.0));
        assert!(!border.contains(7.0, 0.0));
        border.tick();
        assert!(!border.contains(9.0, 0.0));
        border.tick();
        assert!(border.contains(9.0, 0.0));
        assert!(border.contains(-10.0, 0.0));
        assert!(!border.contains(10.0, 0.0));
    }

    #[test]
    fn bounds_are_clamped_at_absolute_world_limit() {
        let border = Worldborder::new(Worldborder::MAX_CENTER_COORDINATE, 0.0, 20.0, 0, 5, 300);
        assert!(border.contains(Worldborder::MAX_CENTER_COORDINATE - 1.0, 0.0));
        assert!(!border.contains(Worldborder::MAX_CENTER_COORDINATE, 0.0));
        assert!(!border.contains(Worldborder::MAX_CENTER_COORDINATE + 1.0, 0.0));
    }
}
