use arrow::array::{Array, ArrayRef, ArrowPrimitiveType, AsArray, PrimitiveArray};
use arrow::buffer::ScalarBuffer;
use arrow::datatypes::DataType;
use std::mem;
use std::sync::Arc;
use arrow_schema::ArrowError;
use crate::splitter_approach::ready_partitions_sink::traits::ReadyPartitionsSink;
use crate::splitters::splitter_nullable_helper::SplitterNullableHelper;
use crate::splitters::traits::{CreateSplitterArgs, Splitter};

/// Splitting a [`PrimitiveArray`] into multiple partitions.
pub(crate) struct PrimitiveColumnSplitter<T: ArrowPrimitiveType> {
    /// The maximum length for partition array
    max_array_length: usize,

    /// Outer vector is for each partition, inner vector is for the values in that partition.
    ///
    /// Not using [`arrow::array::PrimitiveBuilder`] as we want to split the handling
    /// of nulls and valid values allowing for adding values/nulls in separate tight loops.
    partitions: Vec<Vec<T::Native>>,

    nulls: SplitterNullableHelper,

    /// The index of the column in the schema this splitter is for.
    column_index: usize,

    /// The data type of the column, needed when [`T::DATA_TYPE`] is not enough (e.g. `DataType::Decimal`).
    /// or timestamps with timezones.
    data_type: DataType,
}

impl<T: ArrowPrimitiveType> PrimitiveColumnSplitter<T> {
    /// Finish the partition at `partition_index` and reset it for next use.
    ///
    /// Add the finished partition data to the `ready` arrays.
    fn finish_in_progress(
        &mut self,
        partition_index: usize,
        ready_partitions_sink: &mut dyn ReadyPartitionsSink,
    ) -> Result<(), ArrowError> {
        assert!(
            partition_index < self.partitions.len(),
            "Partition index out of bounds"
        );

        let capacity = self.get_capacity_for_partition();
        let partition = &mut self.partitions[partition_index];

        if partition.is_empty() {
            return Ok(()); // Nothing to finish
        }

        let current = mem::replace(partition, Vec::with_capacity(capacity));

        let scalar_buffer = ScalarBuffer::from(current);

        let nulls = self
            .nulls
            .build_nulls_for_partition(partition_index, scalar_buffer.len());

        let arr = PrimitiveArray::<T>::try_new(scalar_buffer, nulls)?
            .with_data_type(self.data_type.clone());

        ready_partitions_sink.add_column_to_partition(
            self.column_index,
            partition_index,
            Arc::new(arr),
        )
    }

    /// Get how much to preallocate for each partition.
    fn get_capacity_for_partition(&self) -> usize {
        // TODO - this should be depend on the partition, because we don't want to allocate for partitions that we don't have any data for
        //        when we have a lot of partitions, this can be a problem or when the data is sorted (which is common for sort before and after shuffle
        self.max_array_length
    }
}

impl<T: ArrowPrimitiveType> Splitter for PrimitiveColumnSplitter<T> {
    fn new(args: CreateSplitterArgs<'_>) -> Self {
        // Not asserting the data type to be the same as `T::DATA_TYPE` because it might have
        // additional information like timezone for timestamps or precision and scale for decimals.
        // causing the assertion to fail.

        // TODO - this will be a problem for memory constrained environments, and is not optimal for large number of partitions and skewed data
        //        as we preallocate all the partitions which is bad for the following scenarios:
        //        1. Some partitions might not have any data at all
        //        2. We only work we small number of partitions at a time
        //           (e.g. for range partition when the data is sorted)

        let partitions = vec![Vec::with_capacity(args.batch_size); args.number_of_partitions];
        let nulls = SplitterNullableHelper::new(
            args.field.is_nullable(),
            args.number_of_partitions,
            args.batch_size,
        );

        Self {
            max_array_length: args.batch_size,
            partitions,
            nulls,
            column_index: args.column_index,
            data_type: args.field.data_type().clone(),
        }
    }

    fn add_values(
        &mut self,
        input_column: &ArrayRef,
        indices: &[u32],
        ready_partitions_sink: &mut dyn ReadyPartitionsSink,
    ) -> Result<(), ArrowError> {
        assert_eq!(
            input_column.data_type(),
            &self.data_type,
            "Input column data type must match the splitter data type"
        );
        let input_column = input_column.as_primitive::<T>();

        // Adding the nulls before the values as we finish in progress
        // array when we reach the max_array_length
        self.nulls.add_nulls(input_column, indices);

        for (&partition_index, &column_value) in
            indices.indices().iter().zip(input_column.values().iter())
        {
            let partition = &mut self.partitions[partition_index as usize];
            partition.push(column_value);

            if partition.len() >= self.max_array_length {
                self.finish_in_progress(partition_index as usize, ready_partitions_sink)?;
            }
        }

        Ok(())
    }

    /// Finish all the in progress partitions and return the arrays for each partition.
    fn finish_and_reset(
        &mut self,
        ready_partitions_sink: &mut dyn ReadyPartitionsSink,
    ) -> Result<(), ArrowError> {
        for partition_index in 0..self.partitions.len() {
            if !self.partitions[partition_index].is_empty() {
                self.finish_in_progress(partition_index, ready_partitions_sink)?;
            }
        }

        Ok(())
    }
}
