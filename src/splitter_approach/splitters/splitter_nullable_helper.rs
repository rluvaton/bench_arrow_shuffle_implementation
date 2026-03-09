use arrow::array::{Array, NullBufferBuilder};
use arrow::buffer::NullBuffer;

pub(crate) struct SplitterNullableHelper {
    /// The null buffer builders for each partition
    ///
    /// This will be `None` if the field is not nullable.
    null_buffer_builders: Option<Vec<NullBufferBuilder>>,
}

impl SplitterNullableHelper {
    pub(super) fn new(nullable: bool, number_of_partitions: usize, array_size: usize) -> Self {
        let null_buffer_builders = if nullable {
            // TODO - this will be a problem for memory constrained environments, and is not optimal for large number of partitions and skewed data
            //        as we preallocate all the partitions which is bad for the following scenarios:
            //        1. Some partitions might not have any data at all
            //        2. We only work we small number of partitions at a time
            //           (e.g. for range partition when the data is sorted)

            Some(
                (0..number_of_partitions)
                    .map(|_| NullBufferBuilder::new(array_size))
                    .collect(),
            )
        } else {
            None
        };

        Self {
            null_buffer_builders,
        }
    }

    pub(super) fn add_nulls(&mut self, input_column: &impl Array, indices: &[u32]) {
        // Updating the null buffers before the values as we finish in progress
        // array when we reach the max_array_length
        match (
            self.null_buffer_builders.as_mut(),
            input_column.null_count(),
        ) {
            (None, 0) => {
                // Nothing to do
            }
            (None, _) => {
                panic!("Input column must not have nulls, when the field is not nullable");
            }
            (Some(null_buffer_builders), 0) => {
                for &partition_index in indices {
                    null_buffer_builders[partition_index as usize].append_non_null();
                }
            }
            (Some(null_buffer_builders), _) => {
                let input_nulls = input_column.nulls().expect("must have nulls");
                for (&partition_index, is_valid) in indices.iter().zip(input_nulls.iter())
                {
                    null_buffer_builders[partition_index as usize].append(is_valid);
                }
            }
        }
    }

    /// Take only the first `length` nulls from the `null_buffer_builder` for that partition
    ///
    /// There must be at least `length` nulls in the `null_buffer_builder`.
    pub(crate) fn build_nulls_for_partition(
        &mut self,
        partition_index: usize,
        length: usize,
    ) -> Option<NullBuffer> {
        // Not using question mark and instead doing the else for more explicit code flow
        #[allow(clippy::question_mark)]
        let Some(null_buffer_builders) = self.null_buffer_builders.as_mut() else {
            return None;
        };

        let null_buffer_builder = &mut null_buffer_builders[partition_index];
        let length_of_nulls = null_buffer_builder.len();
        let mut nulls = null_buffer_builder.finish();

        assert!(
            length_of_nulls >= length,
            "Number of nulls must be greater than or equal to number of values, make sure to add the nulls before adding the values, number of nulls {length_of_nulls}, number of values {length}"
        );

        // If we don't get all the nulls we need to only get the `length`
        if length_of_nulls != length {
            let number_of_nulls_to_add_back = length_of_nulls - length;
            nulls = match nulls {
                Some(nulls) => {
                    let nulls_in_current_value = nulls.slice(0, length);

                    let nulls_to_add_back = nulls.slice(length, number_of_nulls_to_add_back);

                    if nulls_to_add_back.null_count() > 0 {
                        null_buffer_builder.append_buffer(&nulls_to_add_back);
                    } else {
                        // If there are no nulls to add back, we can just ignore it
                        null_buffer_builder.append_n_non_nulls(number_of_nulls_to_add_back);
                    }

                    Some(nulls_in_current_value)
                }
                None => {
                    // Add back the valid values that we took
                    null_buffer_builder.append_n_non_nulls(number_of_nulls_to_add_back);
                    None
                }
            };
        }

        nulls.filter(|nulls| nulls.null_count() > 0)
    }
}
