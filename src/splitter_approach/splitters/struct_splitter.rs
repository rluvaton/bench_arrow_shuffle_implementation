use arrow::array::{ArrayRef, AsArray};
use arrow::datatypes::DataType;
use arrow_schema::ArrowError;
use crate::splitter_approach::ready_partitions_sink::struct_array_sink::StructArraySink;
use crate::splitter_approach::ready_partitions_sink::traits::ReadyPartitionsSink;
use crate::splitters::create_splitter;
use crate::splitters::splitter_nullable_helper::SplitterNullableHelper;
use crate::splitters::traits::{CreateSplitterArgs, Splitter};

/// Splitting a [`StructArray`] into multiple partitions.
pub(crate) struct StructSplitter {
    /// Holding the ready columns for each partition in the struct so we can finalize the struct
    struct_array_sink: StructArraySink,

    /// Splitter for each field in the struct.
    splitters: Vec<Box<dyn Splitter>>,

    nulls: SplitterNullableHelper,

    /// The index of the column in the schema this splitter is for
    column_index: usize,
}

impl Splitter for StructSplitter {
    fn new(args: CreateSplitterArgs<'_>) -> Self {
        let nulls = SplitterNullableHelper::new(
            args.field.is_nullable(),
            args.number_of_partitions,
            args.batch_size,
        );

        let fields = match args.field.data_type() {
            DataType::Struct(fields) => fields,
            _ => panic!("Data type must be a struct"),
        };

        Self {
            struct_array_sink: StructArraySink::new(args.number_of_partitions, fields.clone()),
            splitters: fields
                .iter()
                .enumerate()
                .map(|(index, f)| {
                    create_splitter(index, f, args.batch_size, args.number_of_partitions)
                        .expect("Must be able to create a splitter")
                })
                .collect(),
            nulls,
            column_index: args.column_index,
        }
    }

    fn add_values(
        &mut self,
        input_column: &ArrayRef,
        indices: &[u32],
        ready_partitions_sink: &mut dyn ReadyPartitionsSink,
    ) -> Result<(), ArrowError> {
        let input_column = input_column.as_struct();

        // Adding the nulls before the values as when the splitters reach the max_array_length,
        // they will finish the in-progress array
        self.nulls.add_nulls(input_column, indices);

        for (column, splitter) in input_column.columns().iter().zip(self.splitters.iter_mut()) {
            let mut ready_partitions_sink_wrapper = self.struct_array_sink.wrap(
                ready_partitions_sink,
                &mut self.nulls,
                self.column_index,
            );

            // No need to check for length greater than max_array_length as it is already handled inside the splitter
            // and the columns will be added to the ready wrapper
            splitter.add_values(column, indices, &mut ready_partitions_sink_wrapper)?;
        }

        Ok(())
    }

    /// Finish all the in progress partitions and return the arrays for each partition.
    fn finish_and_reset(
        &mut self,
        ready_partitions_sink: &mut dyn ReadyPartitionsSink,
    ) -> Result<(), ArrowError> {
        for splitter in self.splitters.iter_mut() {
            let mut ready_partitions_sink_wrapper = self.struct_array_sink.wrap(
                ready_partitions_sink,
                &mut self.nulls,
                self.column_index,
            );

            splitter.finish_and_reset(&mut ready_partitions_sink_wrapper)?;
        }

        // Asserting that we have no in-progress partitions
        self.struct_array_sink.assert_all_partitions_are_empty();

        Ok(())
    }
}
