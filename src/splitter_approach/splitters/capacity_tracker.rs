use std::collections::VecDeque;
use std::ops::AddAssign;

pub(crate) trait BatchCapacity:
    Default + Copy + std::fmt::Debug + std::cmp::PartialEq + AddAssign
{
    fn num_rows(&self) -> usize;
}

/// Tracks capacity requirements per partition to enable efficient preallocation.
///
/// For each partition, maintains a queue of `BatchCapacity` entries representing
/// upcoming batches. This allows us to know exactly how much space to preallocate
/// before adding rows.
pub(crate) struct CapacityTracker<T> {
    max_batch_size: usize,

    /// Queue of pending batch capacities per partition.
    /// Each partition has a queue where:
    /// - The front entry represents the capacity for the current/next batch to allocate
    /// - Subsequent entries represent future batches
    pending_batches: Vec<VecDeque<T>>,
}

impl<T: BatchCapacity> CapacityTracker<T> {
    pub(crate) fn new(max_batch_size: usize, num_partitions: usize) -> Self {
        Self {
            pending_batches: vec![VecDeque::with_capacity(2); num_partitions],
            max_batch_size,
        }
    }

    /// Returns the capacity for the next batch in the given partition.
    ///
    /// If there are multiple pending batches, removes and returns the front one.
    /// If there's only one batch (the current one being built), returns a copy without removing.
    pub(crate) fn next_capacity(&mut self, partition: usize) -> Option<T> {
        let batches = &mut self.pending_batches[partition];

        if batches.len() > 1 {
            // Multiple batches queued - pop the front one (it's complete)
            batches.pop_front()
        } else {
            // Only one batch and not enough to flush so keeping it as next rows will be added to it
            batches.back().copied()
        }
    }

    /// Records that a row with the given size will be added to the partition.
    pub(crate) fn record_row(&mut self, partition: u32, size: T) {
        assert_eq!(size.num_rows(), 1, "Expected a single row");
        let batches = &mut self.pending_batches[partition as usize];
        let mut len = batches.len();

        if batches.is_empty() {
            batches.push_back(T::default());
            len = 1;
        }

        let current_batch = &mut batches[len - 1];

        current_batch.add_assign(size);

        if current_batch.num_rows() >= self.max_batch_size {
            // Current batch is full, start tracking the next one
            batches.push_back(T::default());
        }
    }
}
