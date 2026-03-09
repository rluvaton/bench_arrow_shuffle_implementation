use arrow::array::ArrayRef;
use arrow::datatypes::Field;
use arrow_schema::ArrowError;
use crate::splitter_approach::ready_partitions_sink::traits::ReadyPartitionsSink;

pub(crate) trait Splitter: Sync + Send {
    /// Create a new instance of the splitter with the given maximum array length and number of partitions.
    fn new(args: CreateSplitterArgs<'_>) -> Self
    where
        Self: Sized;

    /// Add values to the splitter for the given indices.
    ///
    /// `input_column` is the column to add values from, and `indices` are the indices of the values to add.
    fn add_values(
        &mut self,
        input_column: &ArrayRef,
        indices: &[u32],
        ready_partitions_sink: &mut dyn ReadyPartitionsSink,
    ) -> Result<(), ArrowError>;

    /// Consume the current state and return a vector of vectors of `ArrayRef`.
    ///
    /// Outer vector is for each partition, and inner vector contains the arrays for that partition.
    fn finish(mut self, ready_partitions_sink: &mut dyn ReadyPartitionsSink) -> Result<(), ArrowError>
    where
        Self: Sized,
    {
        self.finish_and_reset(ready_partitions_sink)
    }

    fn finish_and_reset(
        &mut self,
        ready_partitions_sink: &mut dyn ReadyPartitionsSink,
    ) -> Result<(), ArrowError>;
}

pub(crate) struct CreateSplitterArgs<'a> {
    pub(crate) batch_size: usize,
    pub(crate) number_of_partitions: usize,
    pub(crate) field: &'a Field,
    pub(crate) column_index: usize,
}
