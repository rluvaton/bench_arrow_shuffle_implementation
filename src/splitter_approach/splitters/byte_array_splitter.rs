use super::capacity_tracker::{BatchCapacity, CapacityTracker};
use arrow::array::{
    Array, ArrayRef, AsArray, GenericByteArray, NullBufferBuilder, OffsetBufferBuilder,
};
use arrow::buffer::{Buffer, NullBuffer};
use arrow::datatypes::{ArrowNativeType, ByteArrayType};
use std::mem;
use std::sync::Arc;
use arrow_schema::ArrowError;
use crate::splitter_approach::ready_partitions_sink::traits::ReadyPartitionsSink;
use crate::splitter_approach::splitters::traits::{CreateSplitterArgs, Splitter};

#[derive(Debug, Clone, Copy, Default, PartialEq)]
struct BytesArrayBatchCapacity {
    total_bytes: usize,
    num_of_items: usize,
}

impl BatchCapacity for BytesArrayBatchCapacity {
    fn num_rows(&self) -> usize {
        self.num_of_items
    }
}

impl std::ops::AddAssign for BytesArrayBatchCapacity {
    fn add_assign(&mut self, rhs: Self) {
        self.total_bytes += rhs.total_bytes;
        self.num_of_items += rhs.num_of_items;
    }
}

/// Splitting a [`GenericByteArray<T>`] into multiple partitions.
pub(crate) struct ByteArrayColumnSplitter<T: ByteArrayType> {
    /// The maximum length for partition array
    max_array_length: usize,

    /// Not using [`arrow::array::GenericByteBuilder`] as we:
    /// 1. want complete control over the allocation to allow us to preallocate
    /// 2. Want to work with bytes instead of `str` in the case of String
    /// 3. Want to avoid appending to the null buffer and allocate if the field is not nullable
    partitions: Vec<(
        // The bytes
        Vec<u8>,
        // The offsets for the byte arrays
        OffsetBufferBuilder<T::Offset>,
        // The null buffer builder for the partition,
        // when the field is not nullable this will be empty and not used
        NullBufferBuilder,
    )>,

    is_nullable: bool,

    /// The index of the column in the schema this splitter is for.
    column_index: usize,
    /// Tracks capacity requirements to enable efficient preallocation
    capacity_tracker: CapacityTracker<BytesArrayBatchCapacity>,
}

impl<T: ByteArrayType> ByteArrayColumnSplitter<T> {
    fn preallocate(&mut self, indices: &[u32], input_column: &GenericByteArray<T>) {
        let mut touched_partitions =
            hashbrown::HashSet::<u32>::with_capacity(self.partitions.len());

        for (&partition_index, length) in indices.iter().zip(input_column.offsets().lengths()) {
            touched_partitions.insert(partition_index);
            // TODO - if the field is nullable and the item is null don't count the length
            self.capacity_tracker.record_row(
                partition_index,
                BytesArrayBatchCapacity {
                    // Add the length of the item to the total bytes
                    total_bytes: length,

                    // Each index corresponds to one item
                    num_of_items: 1,
                },
            );
        }

        for partition_index in touched_partitions.into_iter() {
            let partition = &mut self.partitions[partition_index as usize];
            let BytesArrayBatchCapacity {
                num_of_items,
                total_bytes,
            } = self
                .capacity_tracker
                .next_capacity(partition_index as usize)
                .unwrap();

            // Preallocate the offsets and bytes for the partition
            partition.0.reserve(total_bytes);
            partition.1.reserve(num_of_items);

            // We can't reserve for null buffer
        }
    }

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

        let partition_length = self.array_length(partition_index);

        if partition_length == 0 {
            return Ok(()); // Nothing to finish
        }

        let (bytes, indices, nulls) = &mut self.partitions[partition_index];

        let output_nulls = if self.is_nullable {
            assert_eq!(
                nulls.len(),
                partition_length,
                "Null buffer length must match the array length",
            );
            nulls.finish()
        } else {
            assert_eq!(
                nulls.len(),
                0,
                "when field is not nullable null buffer must not be updated",
            );
            None
        };

        let BytesArrayBatchCapacity {
            num_of_items,
            total_bytes,
        } = self
            .capacity_tracker
            .next_capacity(partition_index)
            .unwrap_or_default();

        let output_indices = mem::replace(indices, OffsetBufferBuilder::new(num_of_items));
        let output_indices = output_indices.finish();
        let current = mem::replace(bytes, Vec::with_capacity(total_bytes));

        // Safety: this is safe as we are derived from a valid Arrow array
        // Doing uncheck as for Strings array there is an expensive validation for valid UTF-8
        let array = unsafe {
            GenericByteArray::<T>::new_unchecked(
                output_indices,
                Buffer::from(current),
                output_nulls,
            )
        };

        ready_partitions_sink.add_column_to_partition(
            self.column_index,
            partition_index,
            Arc::new(array),
        )
    }

    fn array_length(&self, partition_index: usize) -> usize {
        // The first index is always 0, even for empty data
        self.partitions[partition_index].1.len() - 1
    }

    fn inner_add_values<const IS_FIELD_NULLABLE: bool, const HAS_NULLS: bool>(
        &mut self,
        input_column: &GenericByteArray<T>,
        indices: &[u32],
        ready_partitions_sink: &mut dyn ReadyPartitionsSink,
    ) -> Result<(), ArrowError> {
        let data_buffer = input_column.value_data();
        assert_eq!(
            IS_FIELD_NULLABLE, self.is_nullable,
            "IS_FIELD_NULLABLE must match the splitter's is_nullable"
        );

        let null_buffer = match (HAS_NULLS, IS_FIELD_NULLABLE) {
            (true, true) => {
                assert!(
                    input_column.null_count() > 0,
                    "Must have nulls if HAS_NULLS is true"
                );
                input_column
                    .nulls()
                    .expect("Must have nulls if HAS_NULLS is true")
                    .clone()
            }
            (true, false) => {
                panic!("HAS_NULLS cannot be true if IS_FIELD_NULLABLE is false");
            }
            (false, _) => {
                assert_eq!(
                    input_column.null_count(),
                    0,
                    "Must not have nulls if HAS_NULLS is false"
                );
                NullBuffer::new_valid(0)
            }
        };

        // TODO - in the preallocate we count how many items are there for each partition
        //        if we don't have a lot of partitions in the indices and the input column
        //        does not have any null we can add all the nulls at once instead of adding
        //        them one by one

        for (array_index, (&partition_index, start_and_end)) in indices
            .iter()
            .zip(input_column.offsets().windows(2))
            .enumerate()
        {
            let start = start_and_end[0].as_usize();
            let end = start_and_end[1].as_usize();

            let is_valid = !HAS_NULLS || null_buffer.is_valid(array_index);
            let length = if is_valid { end - start } else { 0 };

            let partition = &mut self.partitions[partition_index as usize];

            if is_valid {
                partition.0.extend_from_slice(&data_buffer[start..end]);
            }

            partition.1.push_length(length);

            if IS_FIELD_NULLABLE {
                partition.2.append(is_valid);
            }

            // if self.array_length(partition_index as usize) >= self.max_array_length {
            //     self.finish_in_progress(partition_index as usize, ready_partitions_sink)?;
            // }
        }

        Ok(())
    }
}

impl<T: ByteArrayType> Splitter for ByteArrayColumnSplitter<T> {
    fn new(args: CreateSplitterArgs<'_>) -> Self {
        assert_eq!(
            args.field.data_type(),
            &T::DATA_TYPE,
            "Data type must match the type of the splitter"
        );

        let is_nullable = args.field.is_nullable();

        // TODO - this will be a problem for memory constrained environments, and is not optimal for large number of partitions and skewed data
        //        as we preallocate all the partitions which is bad for when we only work on a
        //        small number of partitions at a time (e.g. for range partition when the data is sorted)

        let partitions = (0..args.number_of_partitions)
            .map(|_| {
                (
                    // Don't preallocate the bytes and the offset as we will preallocate them later
                    // when we get data for them allowing for more efficient memory usage when some partitions are not used
                    Vec::with_capacity(0),
                    OffsetBufferBuilder::<T::Offset>::new(0),
                    // Preallocate the null buffer now as we can't preallocate later
                    if is_nullable {
                        NullBufferBuilder::new(args.batch_size)
                    } else {
                        // No nulls than no need to allocate null buffer
                        NullBufferBuilder::new(0)
                    },
                )
            })
            .collect::<Vec<_>>();

        Self {
            max_array_length: args.batch_size,
            partitions,
            column_index: args.column_index,
            is_nullable,
            capacity_tracker: CapacityTracker::new(args.batch_size, args.number_of_partitions),
        }
    }

    fn add_values(
        &mut self,
        input_column: &ArrayRef,
        indices: &[u32],
        ready_partitions_sink: &mut dyn ReadyPartitionsSink,
    ) -> Result<(), ArrowError> {
        let input_column = input_column.as_bytes::<T>();

        self.preallocate(indices, input_column);

        match (self.is_nullable, input_column.null_count() > 0) {
            (true, true) => {
                self.inner_add_values::<true, true>(input_column, indices, ready_partitions_sink)
            }
            (true, false) => {
                self.inner_add_values::<true, false>(input_column, indices, ready_partitions_sink)
            }
            (false, true) => {
                panic!("Input column must not have nulls when the field is not nullable")
            }
            (false, false) => {
                self.inner_add_values::<false, false>(input_column, indices, ready_partitions_sink)
            }
        }
    }

    /// Finish all the in progress partitions and return the arrays for each partition.
    fn finish_and_reset(
        &mut self,
        ready_partitions_sink: &mut dyn ReadyPartitionsSink,
    ) -> Result<(), ArrowError> {
        // Finish all the partitions that are in progress
        for partition_index in 0..self.partitions.len() {
            if self.array_length(partition_index) > 0 {
                self.finish_in_progress(partition_index, ready_partitions_sink)?;
            }
        }

        Ok(())
    }
}
