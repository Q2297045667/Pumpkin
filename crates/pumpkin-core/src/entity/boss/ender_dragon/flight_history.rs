pub const LENGTH: usize = 64;
const MASK: i32 = LENGTH as i32 - 1;

#[derive(Clone, Copy, Default, Debug)]
pub struct Sample {
    pub y: f64,
    pub y_rot: f32,
}

#[derive(Clone, Copy, Debug)]
pub struct DragonFlightHistory {
    samples: [Sample; LENGTH],
    head: i32,
}

impl Default for DragonFlightHistory {
    fn default() -> Self {
        Self {
            samples: [Sample::default(); LENGTH],
            head: -1,
        }
    }
}

impl DragonFlightHistory {
    pub fn record(&mut self, y: f64, y_rot: f32) {
        let sample = Sample { y, y_rot };
        if self.head < 0 {
            self.samples.fill(sample);
        }
        self.head = (self.head + 1) & MASK;
        self.samples[self.head as usize] = sample;
    }

    #[must_use]
    pub const fn get(&self, offset: i32) -> Sample {
        self.samples[(self.head.wrapping_sub(offset) & MASK) as usize]
    }
}

#[cfg(test)]
mod tests {
    use super::DragonFlightHistory;

    #[test]
    fn first_record_initializes_delayed_part_positions() {
        let mut history = DragonFlightHistory::default();
        history.record(100.0, 45.0);
        for delay in [0, 5, 10, 63] {
            let sample = history.get(delay);
            assert_eq!((sample.y, sample.y_rot), (100.0, 45.0));
        }
        history.record(102.0, 50.0);
        let newest = history.get(0);
        let previous = history.get(1);
        assert_eq!((newest.y, newest.y_rot), (102.0, 50.0));
        assert_eq!((previous.y, previous.y_rot), (100.0, 45.0));
    }

    #[test]
    fn ring_wrap_preserves_the_oldest_delayed_sample() {
        let mut history = DragonFlightHistory::default();
        history.record(100.0, 0.0);
        for tick in 1..=64 {
            history.record(200.0 + f64::from(tick), tick as f32);
        }
        assert_eq!(history.get(0).y, 264.0);
        assert_eq!(history.get(63).y, 201.0);
        assert_eq!(history.get(64).y, 264.0);
    }
}
