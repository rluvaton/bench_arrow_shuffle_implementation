use arrow::array::ArrayRef;
use arrow_schema::ArrowError;

pub(crate) trait ReadyPartitionsSink {
    fn add_column_to_partition(
        &mut self,
        column_index: usize,
        partition_index: usize,
        array: ArrayRef,
    ) -> Result<(), ArrowError>;
}
