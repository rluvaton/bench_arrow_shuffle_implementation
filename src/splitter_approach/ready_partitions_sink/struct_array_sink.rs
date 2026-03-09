use arrow::array::{Array, ArrayRef, StructArray};
use arrow::buffer::NullBuffer;
use arrow::datatypes::Fields;
use std::collections::VecDeque;
use std::sync::Arc;
use arrow_schema::ArrowError;
use crate::splitter_approach::ready_partitions_sink::traits::ReadyPartitionsSink;
use crate::splitters::splitter_nullable_helper::SplitterNullableHelper;

#[derive(Debug, Clone)]
struct PendingPartition {
    /// The fields for the schema of the struct array.
    fields: Fields,

    /// ready_batches_columns[batch_index][column_index] contains the array for that column in the batch ready to be created.
    ready_batches_columns: VecDeque<(Option<NullBuffer>, Vec<ArrayRef>)>,

    /// how much arrays for a column are in `ready_batches_columns`.
    number_of_arrays_per_column: Vec<usize>,

    /// How much null buffers are in `ready_batches_columns`
    number_of_null_buffers: usize,
}

impl PendingPartition {
    fn new(fields: Fields) -> Self {
        assert_ne!(fields.len(), 0, "Empty fields are not supported");
        let ready_batches_columns = VecDeque::with_capacity(2);

        Self {
            number_of_arrays_per_column: vec![0; fields.len()],
            number_of_null_buffers: 0,
            fields,
            ready_batches_columns,
        }
    }

    fn add_null_buffer(&mut self, null_buffer: Option<NullBuffer>) {
        assert!(
            self.number_of_null_buffers <= self.ready_batches_columns.len(),
            "Number of null buffers {} must be either total ready batches columns or next batch, current number of batches {}",
            self.number_of_null_buffers,
            self.ready_batches_columns.len()
        );

        if self.number_of_null_buffers == self.ready_batches_columns.len() {
            self.ready_batches_columns
                .push_back((None, Vec::with_capacity(self.fields.len())));
        }

        let (nulls, _) = &mut self.ready_batches_columns[self.number_of_null_buffers];

        assert_eq!(
            nulls, &mut None,
            "New null buffer should have placeholder None, but it already has a value"
        );

        *nulls = null_buffer;

        self.number_of_null_buffers += 1;
    }

    fn add_column(&mut self, column_index: usize, array: ArrayRef) {
        assert_ne!(
            column_index,
            self.fields.len() - 1,
            "Last column should not be added here, it should be added in `finish_with_last_column`"
        );

        let number_of_arrays_for_column = self.number_of_arrays_per_column[column_index];
        assert!(
            number_of_arrays_for_column <= self.ready_batches_columns.len(),
            "Number of arrays for column {number_of_arrays_for_column} must be either total ready batches columns or next batch, current number of batches {}",
            self.ready_batches_columns.len()
        );
        if number_of_arrays_for_column == self.ready_batches_columns.len() {
            self.ready_batches_columns.push_back((
                // Setting None until we add a null buffer
                None,
                Vec::with_capacity(self.fields.len()),
            ));
        }
        self.ready_batches_columns[number_of_arrays_for_column]
            .1
            .push(array);

        self.number_of_arrays_per_column[column_index] += 1;
    }

    fn finish_with_last_column(&mut self, last_column_array: ArrayRef) -> Result<StructArray, ArrowError> {
        // Assert that all other columns have at least one array
        self.number_of_arrays_per_column
            .iter_mut()
            .enumerate()
            .rev()
            // Skip the last column as we are adding it now
            .skip(1)
            .for_each(|(column_index, number_of_arrays_for_column)| {
                assert_ne!(
                    *number_of_arrays_for_column, 0,
                    "Column {column_index} is empty, but it should have at least one array"
                );

                // Decrease the count as we will remove it
                *number_of_arrays_for_column -= 1;
            });

        assert_ne!(
            self.number_of_null_buffers, 0,
            "There should be at least one null buffer"
        );
        self.number_of_null_buffers -= 1;

        let (null_buffer, mut columns_in_batch) = self
            .ready_batches_columns
            .pop_front()
            .expect("There should be at least one struct field and nulls ready to be created");
        columns_in_batch.push(last_column_array);

        Ok(StructArray::try_new(
            self.fields.clone(),
            columns_in_batch,
            null_buffer,
        )?)
    }
}

/// Add ready columns that is already split into partitions and build StructArray for each
pub(crate) struct StructArraySink {
    /// The struct fields to build with
    fields: Fields,

    /// Vector of partitions, each partition contains a vector of arrays for each column.
    partitions: Vec<PendingPartition>,
}

impl StructArraySink {
    pub(crate) fn new(number_of_partitions: usize, fields: Fields) -> Self {
        Self {
            partitions: vec![PendingPartition::new(fields.clone()); number_of_partitions],
            fields,
        }
    }

    /// Assert that all the partitions are empty, meaning fully consumed.
    pub(crate) fn assert_all_partitions_are_empty(&self) {
        assert!(
            self.partitions
                .iter()
                .all(|p| p.ready_batches_columns.is_empty()),
            "Partitions should be empty, but they are not"
        );
    }

    /// Wrap [`ReadyPartitionsSink`] with this so we won't add every ready column to the sink,
    /// but only when the entire struct is ready.
    pub(crate) fn wrap<'a>(
        &'a mut self,
        ready_partitions_sink: &'a mut dyn ReadyPartitionsSink,
        nullable_helper: &'a mut SplitterNullableHelper,
        column_index: usize,
    ) -> ReadyPartitionsSinkToStructArrayWrapper<'a> {
        ReadyPartitionsSinkToStructArrayWrapper {
            column_index,
            struct_array_sink: self,
            ready_partitions_sink,
            nullable_helper,
        }
    }
}

/// Wrapper around existing [`ReadyPartitionsSink`] so when a struct is finish it will add it there
pub(crate) struct ReadyPartitionsSinkToStructArrayWrapper<'a> {
    /// Column index for the StructArray
    column_index: usize,

    struct_array_sink: &'a mut StructArraySink,

    /// The wrapped `ReadyPartitionsSink` to add the finished StructArray to
    ready_partitions_sink: &'a mut dyn ReadyPartitionsSink,

    /// The nullable helper that hold the nulls for all the partitions
    nullable_helper: &'a mut SplitterNullableHelper,
}

impl<'a> ReadyPartitionsSink for ReadyPartitionsSinkToStructArrayWrapper<'a> {
    fn add_column_to_partition(
        &mut self,
        column_index: usize,
        partition_index: usize,
        array: ArrayRef,
    ) -> Result<(), ArrowError> {
        assert_ne!(
            array.len(),
            0,
            "Array at column {column_index} inside struct (struct is in column {}) in partition {partition_index} is empty while it should not be",
            self.column_index
        );

        let current_partition = &mut self.struct_array_sink.partitions[partition_index];

        // If this is the last column in the struct array we can finalize the struct and add it to the wrapped `AddReady`
        if column_index == self.struct_array_sink.fields.len() - 1 {
            let nulls = self
                .nullable_helper
                .build_nulls_for_partition(partition_index, array.len());
            current_partition.add_null_buffer(nulls);

            let struct_array = current_partition.finish_with_last_column(array)?;

            if !struct_array.is_empty() {
                self.ready_partitions_sink.add_column_to_partition(
                    self.column_index,
                    partition_index,
                    Arc::new(struct_array),
                )?;
            }
        } else {
            current_partition.add_column(column_index, array);
        }

        Ok(())
    }
}
