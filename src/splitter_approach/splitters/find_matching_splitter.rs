use super::*;
use arrow::array::{downcast_primitive, ArrayRef};
use arrow::datatypes::{DataType, Field, FieldRef, Schema, Utf8Type};
use std::sync::Arc;
use arrow_array::{GenericByteArray, GenericStringArray, RecordBatch};
use arrow_array::types::{GenericBinaryType, GenericStringType};
use arrow_schema::{ArrowError, SchemaRef};
use crate::splitter_approach::ready_partitions_sink::shuffle_encoded_sink::ShuffleEncodedSink;
use crate::splitter_approach::ready_partitions_sink::traits::ReadyPartitionsSink;
use crate::splitter_approach::splitters::byte_array_splitter::ByteArrayColumnSplitter;
use crate::splitter_approach::splitters::traits::{CreateSplitterArgs, Splitter};

pub struct ShuffleArgs<'a> {
    pub number_of_partitions: usize,
    pub batch_size: usize,
    pub schema: &'a SchemaRef,
    pub columns: &'a [Vec<ArrayRef>],
    pub indices: &'a [Vec<u32>],
}

pub fn shuffle_by_splitters(args: ShuffleArgs<'_>) -> Result<Vec<Vec<RecordBatch>>, ArrowError> {
    let mut ready_partitions_sink = ShuffleEncodedSink::new(
        args.number_of_partitions,
        Arc::clone(args.schema),
    );

    let schema_fields = args.schema.fields();

    for (column_index, (column_in_all_batches, field)) in
        args.columns.into_iter().zip(schema_fields.iter()).enumerate()
    {
        let args = FillSingleSplitterArgs {
            batch_size: args.batch_size,
            column_index,
            number_of_partitions: args.number_of_partitions,
            field: field.as_ref(),
            column_in_all_batches,
            ready_partitions_sink: &mut ready_partitions_sink,
            indices: args.indices,
        };
        fill_all_data_for_single_column_in_all_partitions(args)?;
    }

    let output = ready_partitions_sink.into_inner();

    Ok(output)
}

pub(crate) struct FillSingleSplitterArgs<'a> {
    /// The max size of the array to before starting a new one
    pub(crate) batch_size: usize,

    /// The index of the column in the schema
    pub(crate) column_index: usize,

    /// The schema field for this column
    pub(crate) field: &'a Field,

    /// How many partitions are there
    pub(crate) number_of_partitions: usize,

    /// The column in each batch, column_in_all_batches[batch_index] is the array for this column in batch `batch_index`
    pub(crate) column_in_all_batches: &'a [ArrayRef],
    pub(crate) ready_partitions_sink: &'a mut dyn ReadyPartitionsSink,

    /// The indices to determinate which rows to add to each partition.
    pub(crate) indices: &'a [Vec<u32>],
}

/// Fill all data for a single column in all partitions into the [`FillSingleSplitterArgs::ready_partitions_sink`] provided in the args
pub(crate) fn fill_all_data_for_single_column_in_all_partitions(
    args: FillSingleSplitterArgs<'_>,
) -> Result<(), ArrowError> {
    match args.field.data_type() {
        DataType::Utf8 => {
            inner_fill_all_data_for_single_column_in_all_partitions::<ByteArrayColumnSplitter<GenericStringType<i32>>>(args)
        },
        DataType::LargeUtf8 => {
            inner_fill_all_data_for_single_column_in_all_partitions::<ByteArrayColumnSplitter<GenericStringType<i64>>>(args)
        },
        DataType::Binary => {
            inner_fill_all_data_for_single_column_in_all_partitions::<ByteArrayColumnSplitter<GenericBinaryType<i32>>>(args)
        },
        DataType::LargeBinary => {
            inner_fill_all_data_for_single_column_in_all_partitions::<ByteArrayColumnSplitter<GenericBinaryType<i64>>>(args)
        },
        dt => Err(ArrowError::InvalidArgumentError(format!(
            "Unsupported data type {:?} for column {}",
            dt, args.column_index
        ))),
    }
}

fn inner_fill_all_data_for_single_column_in_all_partitions<S: Splitter>(args: FillSingleSplitterArgs<'_>) -> Result<(), ArrowError> {
    let mut splitter = S::new(CreateSplitterArgs {
        batch_size: args.batch_size,
        field: args.field,
        column_index: args.column_index,
        number_of_partitions: args.number_of_partitions,
    });
    for (column, indices) in args
      .column_in_all_batches
      .iter()
      .zip(args.indices.iter())
    {
        // Add the column to the splitter
        splitter.add_values(column, indices.as_slice(), args.ready_partitions_sink)?;
    }

    splitter.finish(args.ready_partitions_sink)?;

    Ok(())
}
