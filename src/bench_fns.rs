use arrow_array::builder::UInt32Builder;
use arrow_array::{Array, ArrayRef, RecordBatch, UInt32Array};
use arrow_row::SortField;
use arrow_schema::SchemaRef;
// use arrow_row::unordered_row::UnorderedRowConverter;
use rand::prelude::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use std::hint;
use std::sync::Arc;
use crate::generate_utils::generate_batch;
use crate::splitters::ShuffleArgs;

pub struct Generate {
  number_of_partitions: usize,
  batch_size: usize,
  inputs: Vec<Input>,
  splitters_input: SplittersInput,
}

impl Generate {
  pub fn new(
    generate_args: GenerateArgs,
  ) -> Self {
    let number_of_partitions = generate_args.num_partitions;
    let batch_size = generate_args.num_rows;

    let inputs = generate_inputs(generate_args);
    Self {
      number_of_partitions,
      splitters_input: SplittersInput::from(&inputs),
      inputs,
      batch_size,
    }
  }

  pub fn derived(&self) -> GenerateInputsDerived<'_> {
    GenerateInputsDerived::from(self)
  }
}

pub struct GenerateInputsDerived<'a> {
  pub generate: &'a Generate,
  pub inputs_ref: Vec<InputRef<'a>>,
  pub input_columns: InputColumns<'a>,
  pub interleave_optimized_input: InterleaveOptimizedInput<'a>,
  pub interleave_column_wise_optimized_input: InterleaveColumnWiseOptimizedInput<'a>,
  pub splitter_args_ref: SplittersInputRef<'a>,
}

impl<'a> From<&'a Generate> for GenerateInputsDerived<'a> {
  fn from(generate: &'a Generate) -> Self {
    let input_columns = InputColumns::from(&generate.inputs);
    Self {
      inputs_ref: generate.inputs.iter().map(|x| x.as_ref()).collect(),
      interleave_optimized_input: InterleaveOptimizedInput::new(&generate.inputs, generate.number_of_partitions),
      interleave_column_wise_optimized_input: InterleaveColumnWiseOptimizedInput::new(&generate.inputs, &input_columns, generate.number_of_partitions),
      splitter_args_ref:generate.splitters_input.as_ref(),

      input_columns,
      generate,
    }
  }
}
pub fn take_to_builders_approach_bench(generated_derive: &GenerateInputsDerived) {
  let output = take_to_builders_approach(generated_derive.inputs_ref.as_slice(), generated_derive.generate.batch_size, generated_derive.generate.number_of_partitions);
  hint::black_box(output);
}

pub fn take_to_builders_column_wise_approach_bench(generated_derive: &GenerateInputsDerived) {
  let output = take_to_builders_column_wise_approach(generated_derive.inputs_ref.as_slice(), &generated_derive.input_columns, generated_derive.generate.batch_size, generated_derive.generate.number_of_partitions);
  hint::black_box(output);
}

pub fn take_approach_bench(generated_derive: &GenerateInputsDerived) {
  let output = take_approach(generated_derive.inputs_ref.as_slice(), generated_derive.generate.batch_size, generated_derive.generate.number_of_partitions);
  hint::black_box(output);
}

pub fn take_column_wise_approach_bench(generated_derive: &GenerateInputsDerived) {
  let output = take_column_wise_approach(generated_derive.inputs_ref.as_slice(), &generated_derive.input_columns, generated_derive.generate.batch_size, generated_derive.generate.number_of_partitions);
  hint::black_box(output);
}

pub fn interleave_approach_bench(generated_derive: &GenerateInputsDerived) {
  let output = interleave_approach(&generated_derive.interleave_optimized_input, generated_derive.generate.batch_size, generated_derive.generate.number_of_partitions);
  hint::black_box(output);
}

pub fn interleave_column_wise_approach_bench(generated_derive: &GenerateInputsDerived) {
  let output = interleave_column_wise_approach(generated_derive.inputs_ref.as_slice(), &generated_derive.interleave_column_wise_optimized_input, generated_derive.generate.batch_size, generated_derive.generate.number_of_partitions);
  hint::black_box(output);
}

pub fn row_format_approach_bench(generated_derive: &GenerateInputsDerived) {
  let output = row_format_approach(generated_derive.inputs_ref.as_slice(), generated_derive.generate.batch_size, generated_derive.generate.number_of_partitions);
  hint::black_box(output);
}

pub fn row_format_approach_partition_wise_bench(generated_derive: &GenerateInputsDerived) {
  let output = row_format_approach_partition_wise(generated_derive.inputs_ref.as_slice(), generated_derive.generate.batch_size, generated_derive.generate.number_of_partitions);
  hint::black_box(output);
}

pub fn optimized_row_format_approach_bench(generated_derive: &GenerateInputsDerived) {
  let output = optimized_row_format_approach(generated_derive.inputs_ref.as_slice(), generated_derive.generate.batch_size, generated_derive.generate.number_of_partitions);
  hint::black_box(output);
}

pub fn optimized_row_format_approach_partition_wise_bench(generated_derive: &GenerateInputsDerived) {
  let output = optimized_row_format_approach_partition_wise(generated_derive.inputs_ref.as_slice(), generated_derive.generate.batch_size, generated_derive.generate.number_of_partitions);
  hint::black_box(output);
}

pub fn splitters_approach_bench(generated_derive: &GenerateInputsDerived) {
  let output = splitters_approach(&generated_derive.splitter_args_ref, generated_derive.generate.batch_size, generated_derive.generate.number_of_partitions);
  hint::black_box(output);
}

fn generate_inputs(args: GenerateArgs) -> Vec<Input> {
  let mut seed = 42;
  (0..args.num_batches)
    .map(|_| {
      let partitions = {
        let mut rng = StdRng::seed_from_u64(seed);
        // Create an even number of items per partition to mimic hash partitioning on distributed data
        let mut base_partitions: Vec<usize> = (0..args.num_rows)
          .map(|row_index| row_index % args.num_partitions)
          .collect();

        // Shuffle uniformly to mimic hash partitioning
        // Using the same rng so next iteration will have different shuffle order
        base_partitions.shuffle(&mut rng);

        base_partitions
      };

      let mut partition_indices = (0..args.num_partitions).map(|_| UInt32Builder::new()).collect::<Vec<_>>();

      for (index, partition) in partitions.iter().enumerate() {
        partition_indices[*partition].append_value(index as u32);
      }



      let partitions_as_slice = partition_indices.into_iter().map(|mut x| x.finish()).collect::<Vec<_>>();
      Input {
        indices_per_partition: partitions_as_slice,
        partitions,
        batch: generate_batch(args.num_rows, &mut seed),
      }
    })
    .collect()
}

pub struct GenerateArgs {
  pub num_partitions: usize,
  pub num_rows: usize,
  pub num_batches: usize,
  pub seed: u64,
}

struct Input {
  batch: RecordBatch,
  /// `partition[row_index] = partition_index`
  partitions: Vec<usize>,

  /// Indices per partition partition[partition_index] = indices in the batch for that partition
  indices_per_partition: Vec<UInt32Array>
}

#[derive(Clone, Copy)]
struct SingleColumnInput<'a> {
  column: &'a ArrayRef,
  /// Indices per partition partition[partition_index] = indices in the batch for that partition
  indices_per_partition: &'a [UInt32Array]
}

struct InputColumns<'a> {
  /// `columns[column_index][batch_index]` = (column value for that batch, indices per partition for that batch)
  columns: Vec<Vec<SingleColumnInput<'a>>>
}

impl<'a> From<&'a Vec<Input>> for InputColumns<'a> {
  fn from(inputs: &'a Vec<Input>) -> Self {
    InputColumns {
      columns: {
        let number_of_columns = inputs[0].batch.num_columns();
        let mut columns = vec![vec![]; number_of_columns];

        for column_index in 0..number_of_columns {
          for batch in inputs {
            columns[column_index].push(SingleColumnInput {
              column: &batch.batch.column(column_index),
              indices_per_partition: &batch.indices_per_partition
            });
          }
        }

        columns
      }
    }
  }
}

impl Input {
  fn as_ref<'a>(&'a self) -> InputRef<'a> {
    InputRef { batch: &self.batch, partitions: &self.partitions, indices_per_partition: &self.indices_per_partition }
  }
}

pub struct InputRef<'a> {
  batch: &'a RecordBatch,
  /// `partition[row_index] = partition_index`
  partitions: &'a [usize],

  /// Indices per partition partition[partition_index] = indices in the batch for that partition
  indices_per_partition: &'a [UInt32Array]
}

/// The output is `output[partition_index][batches]`
type Output = Vec<Vec<RecordBatch>>;

pub fn take_to_builders_approach<'a>(input: &'a [InputRef<'a>], batch_size: usize, number_of_partitions: usize) -> Output {
  let fields = input[0].batch.schema_ref().fields();
  let mut partitions_sink = (0..number_of_partitions).map(|_| crate::take::create_sinks(fields, batch_size)).collect::<Vec<_>>();

  for input in input.iter() {
    for (indices, sinks) in input.indices_per_partition.iter().zip(partitions_sink.iter_mut()) {
      crate::take::take_to_sinks(input.batch.columns(), sinks, indices).expect("should be able to take");
    }
  }

  let output = partitions_sink
    .iter_mut()
    .map(|sinks| {
      let batch = crate::take::finish_sinks(
        fields,
        sinks,
        batch_size
      );

      assert!(batch.num_rows() <= batch_size);
      vec![
        batch
      ]
    }).collect::<Vec<_>>();

  output
}

pub fn take_to_builders_column_wise_approach<'a>(input: &'a [InputRef<'a>], columns_based: &'a InputColumns<'a>, batch_size: usize, number_of_partitions: usize) -> Output {
  let fields = input[0].batch.schema_ref().fields();

  // columns_sink[column_index][partition_index] = sink for that column in that partition
  let mut columns_sink = fields.iter().map(|f| (0..number_of_partitions).map(|_| crate::take::create_sink(f.as_ref(), batch_size)).collect::<Vec<_>>()).collect::<Vec<_>>();

  for (column_and_indices, partition_sinks) in columns_based.columns.iter().zip(columns_sink.iter_mut()) {
    for SingleColumnInput { column, indices_per_partition } in column_and_indices {
      for (indices, sink) in indices_per_partition.iter().zip(partition_sinks.iter_mut()) {
        crate::take::take_to_sink(column.as_ref(), sink, indices).expect("should be able to take");
      }
    }
  }

  // columns[partition_index][column_index] = column values for that partition
  let mut output_partitions = vec![Vec::with_capacity(fields.len()); number_of_partitions];

  for (partitions, field) in columns_sink.iter_mut().zip(fields.iter()) {
    for (partition_index, sink) in partitions.iter_mut().enumerate() {
      let column = crate::take::finish_sink(
        field.as_ref(),
        sink,
        batch_size
      );

      output_partitions[partition_index].push(column);
    }
  }

  let schema = input[0].batch.schema_ref();

  let mut output = vec![Vec::with_capacity(1); number_of_partitions];

  for (partition_index, partition) in output_partitions.into_iter().enumerate() {
    let batch = RecordBatch::try_new(Arc::clone(schema), partition).expect("should be able to create batch");
    assert!(batch.num_rows() <= batch_size);
    output[partition_index].push(batch);
  }

  output
}



pub fn take_approach<'a>(input: &'a [InputRef<'a>], batch_size: usize, number_of_partitions: usize) -> Output {
  let fields = input[0].batch.schema_ref().fields();

  // array[partition_index][column_index][input_batch_index] = array of
  let mut partitions = vec![vec![Vec::with_capacity(input.len()); fields.len()]; number_of_partitions];

  for input in input.iter() {
    for (indices, output_partition) in input.indices_per_partition.iter().zip(partitions.iter_mut()) {
      let columns = arrow_select::take::take_arrays(input.batch.columns(), indices, None).expect("should be able to take");

      for (column_index, column) in columns.into_iter().enumerate() {
        output_partition[column_index].push(column);
      }
    }
  }

  // array[partition_index][column_index] = column values for that partition
  let mut output_columns = vec![Vec::with_capacity(fields.len()); number_of_partitions];

  for (partition, output_partition) in partitions.into_iter().zip(output_columns.iter_mut()) {
    for column_in_batches in partition {
      let concat_input = column_in_batches.iter().map(|a| a.as_ref()).collect::<Vec<_>>();
      let column = arrow_select::concat::concat(concat_input.as_slice()).expect("should be able to concat");
      output_partition.push(column);
    }
  }

  let schema = input[0].batch.schema_ref();

  let output = output_columns.into_iter()
    .map(|partition| {
      let batch = RecordBatch::try_new(Arc::clone(schema), partition).expect("should be able to create batch");
      assert!(batch.num_rows() <= batch_size);

      vec![batch]
    })
    .collect::<Vec<_>>();

  output
}


pub fn take_column_wise_approach<'a>(input: &'a [InputRef<'a>], columns_based: &'a InputColumns<'a>, batch_size: usize, number_of_partitions: usize) -> Output {
  let fields = input[0].batch.schema_ref().fields();

  // array[column_index][partition_index][input_batch_index] = array of
  let mut columns_partitions_input_take = vec![vec![Vec::with_capacity(input.len()); number_of_partitions]; fields.len()];

  for (column_and_indices, output_partitions) in columns_based.columns.iter().zip(columns_partitions_input_take.iter_mut()) {
    for SingleColumnInput { column, indices_per_partition } in column_and_indices {
      for (indices, partition_columns) in indices_per_partition.iter().zip(output_partitions.iter_mut()) {
        let output_column = arrow_select::take::take(column.as_ref(), indices, None).expect("should be able to take");
        partition_columns.push(output_column);
      }
    }
  }

  // array[partition_index][column_index] = column values for that partition
  let mut output_partitions = vec![Vec::with_capacity(fields.len()); number_of_partitions];

  for (column_partitions, field) in columns_partitions_input_take.iter_mut().zip(fields.iter()) {
    for (partition_index, partition) in column_partitions.iter().enumerate() {
      let concat_input = partition.iter().map(|a| a.as_ref()).collect::<Vec<_>>();
      let column = arrow_select::concat::concat(concat_input.as_slice()).expect("should be able to concat");


      output_partitions[partition_index].push(column);
    }
  }

  let schema = input[0].batch.schema_ref();

  let output = output_partitions.into_iter()
    .map(|partition| {
      let batch = RecordBatch::try_new(Arc::clone(schema), partition).expect("should be able to create batch");
      assert!(batch.num_rows() <= batch_size);

      vec![batch]
    })
    .collect::<Vec<_>>();

  output
}

pub struct InterleaveOptimizedInput<'a> {
  batches: Vec<&'a RecordBatch>,
  /// `indices[partition_index][output_row_index] = (batch_index, row_index)`
  partitions: Vec<Vec<(usize, usize)>>,
}

impl<'a> InterleaveOptimizedInput<'a> {
  fn new(inputs: &'a Vec<Input>, number_of_partitions: usize) -> Self {
    let batches = inputs.iter().map(|x| &x.batch).collect::<Vec<_>>();
    let partition_indices = (0..number_of_partitions)
      .map(|partition_index| {
        let indices = inputs.iter().enumerate().flat_map(|(batch_index, input)| {
          input.indices_per_partition[partition_index].values().iter().map(move |index| (batch_index, *index as usize))
        })
          .collect::<Vec<(usize, usize)>>();

        indices
      })
      .collect::<Vec<_>>();
    Self {
      batches,
      partitions: partition_indices
    }
  }
}

pub fn interleave_approach<'a>(input: &'a InterleaveOptimizedInput<'a>, batch_size: usize, number_of_partitions: usize) -> Output {
  let output = input.partitions
    .iter()
    .map(|partition| {
      let batch = arrow_select::interleave::interleave_record_batch(input.batches.as_slice(), partition.as_slice()).expect("should be able to interleave");
      assert!(batch.num_rows() <= batch_size);

      vec![batch]
    })
    .collect::<Vec<_>>();

  output
}

pub struct InterleaveColumnWiseOptimizedInput<'a> {
  /// `columns[column_index][batch_index] = column array for that batch`
  columns: Vec<Vec<&'a dyn Array>>,
  /// `indices[partition_index][output_row_index] = (batch_index, row_index)`
  partitions: Vec<Vec<(usize, usize)>>,
}

impl<'a> InterleaveColumnWiseOptimizedInput<'a> {
  fn new<'b>(inputs: &'a Vec<Input>, input_columns: &'b InputColumns<'a>, number_of_partitions: usize) -> Self {
    let columns = input_columns.columns.iter().map(|partitions| partitions.iter().map(|x| x.column.as_ref()).collect::<Vec<_>>()).collect::<Vec<_>>();
    let partition_indices = (0..number_of_partitions)
      .map(|partition_index| {
        let indices = inputs.iter().enumerate().flat_map(|(batch_index, input)| {
          input.indices_per_partition[partition_index].values().iter().map(move |index| (batch_index, *index as usize))
        })
          .collect::<Vec<(usize, usize)>>();

        indices
      })
      .collect::<Vec<_>>();
    Self {
      columns,
      partitions: partition_indices
    }
  }
}

pub fn interleave_column_wise_approach<'a>(input: &'a [InputRef<'a>], interleave_input: &'a InterleaveColumnWiseOptimizedInput<'a>, batch_size: usize, number_of_partitions: usize) -> Output {
  let schema = input[0].batch.schema_ref();
  let number_of_columns = interleave_input.columns.len();

  interleave_input.partitions.iter()
    .map(|partition| {
      let mut output_partition_columns = Vec::with_capacity(number_of_columns);
      for column_arrays in interleave_input.columns.iter() {
        let output_partition = arrow_select::interleave::interleave(column_arrays.as_slice(), partition.as_slice()).expect("should be able to interleave");
        output_partition_columns.push(output_partition);
      }

      let batch = RecordBatch::try_new(Arc::clone(schema), output_partition_columns).expect("should be able to create batch");
      assert!(batch.num_rows() <= batch_size);

      vec![batch]
    })
    .collect::<Vec<_>>()
}

/// NOTE: in real life we encode data as soon as we get it and save the rows and we don't have
pub fn row_format_approach<'a>(input: &'a [InputRef<'a>], batch_size: usize, number_of_partitions: usize) -> Output {
  let schema = input[0].batch.schema_ref();
  let fields = schema.fields();
  let row_converter = arrow_row::RowConverter::new(
    fields.iter().map(|f| SortField::new(f.data_type().clone())).collect(),
  ).expect("should be able to create row converter");

  let mut partitions_sink = (0..number_of_partitions).map(|_|
    // trying to reserve
    row_converter.empty_rows(batch_size, batch_size * fields.len() * 50)
  ).collect::<Vec<_>>();

  for input in input.iter() {
    let rows = row_converter.convert_columns(input.batch.columns()).expect("should be able to convert");

    // TODO - reserve for each partition
    for (index, partition) in input.partitions.iter().enumerate() {
      let partition_output = unsafe { partitions_sink.get_unchecked_mut(*partition)};

      let row = unsafe { rows.row_unchecked(index) };
      partition_output.push(row);
    }
  }

  let output = partitions_sink
    .iter_mut()
    .map(|partition_rows| {
      let columns = row_converter.convert_rows(partition_rows.iter()).expect("should be able to convert");
      let batch = RecordBatch::try_new(Arc::clone(schema), columns).expect("should be able to create a batch");

      assert!(batch.num_rows() <= batch_size);
      vec![
        batch
      ]
    }).collect::<Vec<_>>();

  output
}


/// NOTE: in real life we encode data as soon as we get it and save the rows and we don't have
pub fn row_format_approach_partition_wise<'a>(input: &'a [InputRef<'a>], batch_size: usize, number_of_partitions: usize) -> Output {
  let schema = input[0].batch.schema_ref();
  let fields = schema.fields();
  let row_converter = arrow_row::RowConverter::new(
    fields.iter().map(|f| SortField::new(f.data_type().clone())).collect(),
  ).expect("should be able to create row converter");

  let mut partitions_sink = (0..number_of_partitions).map(|_|
    // trying to reserve
    row_converter.empty_rows(batch_size, batch_size * fields.len() * 50)
  ).collect::<Vec<_>>();

  for input in input.iter() {
    let rows = row_converter.convert_columns(input.batch.columns()).expect("should be able to convert");

    // TODO - reserve for each partition
    for (indices, partition) in input.indices_per_partition.iter().zip(partitions_sink.iter_mut()) {
      // TODO - add extend for Rows

      for index in indices.values() {

        let row = unsafe { rows.row_unchecked(*index as usize) };
        partition.push(row);
      }
    }
  }

  let output = partitions_sink
    .iter_mut()
    .map(|partition_rows| {
      let columns = row_converter.convert_rows(partition_rows.iter()).expect("should be able to convert");
      let batch = RecordBatch::try_new(Arc::clone(schema), columns).expect("should be able to create a batch");

      assert!(batch.num_rows() <= batch_size);
      vec![
        batch
      ]
    }).collect::<Vec<_>>();

  output
}

fn test_combination_that_fail<'a>(input: &'a [InputRef<'a>], batch_size: usize, number_of_partitions: usize) {
  let schema = input[0].batch.schema_ref();
  let fields = schema.fields();
  let row_converter = crate::unordered_row::UnorderedRowConverter::new(
    fields.clone(),
  ).expect("should be able to create row converter");

  let mut partitions_sink = (0..number_of_partitions).map(|_|
    // trying to reserve
    row_converter.empty_rows(batch_size, batch_size * fields.len() * 50)
  ).collect::<Vec<_>>();

  for input in input.iter() {
    let rows = row_converter.convert_columns(input.batch.columns()).expect("should be able to convert");

    // TODO - reserve for each partition
    for (index, partition) in input.partitions.iter().enumerate() {
      let partition_output = unsafe { partitions_sink.get_unchecked_mut(*partition)};

      let row = unsafe { rows.row_unchecked(index) };
      partition_output.push(row);
    }
  }

  let output = partitions_sink
    .iter_mut()
    .map(|partition_rows| {
      let columns = row_converter.convert_rows(partition_rows.iter()).expect("should be able to convert");
      let batch = RecordBatch::try_new(Arc::clone(schema), columns).expect("should be able to create a batch");

      assert!(batch.num_rows() <= batch_size);
      vec![
        batch
      ]
    }).collect::<Vec<_>>();

  // output
}

/// NOTE: in real life we encode data as soon as we get it and save the rows and we don't have
pub fn optimized_row_format_approach<'a>(input: &'a [InputRef<'a>], batch_size: usize, number_of_partitions: usize) -> Output {
  let schema = input[0].batch.schema_ref();
  let fields = schema.fields();
  let row_converter = crate::unordered_row::UnorderedRowConverter::new(
    fields.clone(),
  ).expect("should be able to create row converter");

  let mut partitions_sink = (0..number_of_partitions).map(|_|
    // trying to reserve
    row_converter.empty_rows(batch_size, batch_size * fields.len() * 50)
  ).collect::<Vec<_>>();

  for input in input.iter() {
    let rows = row_converter.convert_columns(input.batch.columns()).expect("should be able to convert");

    // TODO - reserve for each partition
    for (index, partition) in input.partitions.iter().enumerate() {
      let partition_output = unsafe { partitions_sink.get_unchecked_mut(*partition)};

      let row = unsafe { rows.row_unchecked(index) };
      partition_output.push(row);
    }
  }

  let output = partitions_sink
    .iter_mut()
    .map(|partition_rows| {
      let columns = row_converter.convert_rows(partition_rows.iter()).expect("should be able to convert");
      let batch = RecordBatch::try_new(Arc::clone(schema), columns).expect("should be able to create a batch");

      assert!(batch.num_rows() <= batch_size);
      vec![
        batch
      ]
    }).collect::<Vec<_>>();

  output
}



/// NOTE: in real life we encode data as soon as we get it and save the rows and we don't have
pub fn optimized_row_format_approach_partition_wise<'a>(input: &'a [InputRef<'a>], batch_size: usize, number_of_partitions: usize) -> Output {
  let schema = input[0].batch.schema_ref();
  let fields = schema.fields();
  let row_converter = crate::unordered_row::UnorderedRowConverter::new(
    fields.clone(),
  ).expect("should be able to create row converter");

  let mut partitions_sink = (0..number_of_partitions).map(|_|
    // trying to reserve
    row_converter.empty_rows(batch_size, batch_size * fields.len() * 50)
  ).collect::<Vec<_>>();

  for input in input.iter() {
    let rows = row_converter.convert_columns(input.batch.columns()).expect("should be able to convert");

    // TODO - reserve for each partition
    for (indices, partition) in input.indices_per_partition.iter().zip(partitions_sink.iter_mut()) {
      // TODO - add extend for Rows

      for index in indices.values() {

        let row = unsafe { rows.row_unchecked(*index as usize) };
        partition.push(row);
      }
    }
  }

  let output = partitions_sink
    .iter_mut()
    .map(|partition_rows| {
      let columns = row_converter.convert_rows(partition_rows.iter()).expect("should be able to convert");
      let batch = RecordBatch::try_new(Arc::clone(schema), columns).expect("should be able to create a batch");

      assert!(batch.num_rows() <= batch_size);
      vec![
        batch
      ]
    }).collect::<Vec<_>>();

  output
}

pub struct SplittersInput {
  schema: SchemaRef,
  columns: Vec<Vec<ArrayRef>>,
  indices: Vec<Vec<u32>>,
}

impl SplittersInput {
  pub fn as_ref(&self) -> SplittersInputRef<'_> {
    SplittersInputRef::from(self)
  }
}


impl<'a> From<&'a Vec<Input>> for SplittersInput {
  fn from(inputs: &'a Vec<Input>) -> Self {
    let number_of_columns = inputs[0].batch.num_columns();
    let mut columns = vec![vec![]; number_of_columns];
    let mut indices = vec![];

    for input in inputs {
      indices.push(
        input.partitions.iter().map(|partition_index| *partition_index as u32).collect::<Vec<u32>>()
      )
    }

    for column_index in 0..number_of_columns {
      for input in inputs {
        columns[column_index].push(Arc::clone(input.batch.column(column_index)))
      }
    }

    Self {
      schema: inputs[0].batch.schema(),
      indices,
      columns,
    }
  }

}

pub struct SplittersInputRef<'a> {
  schema: &'a SchemaRef,
  columns: &'a [Vec<ArrayRef>],
  indices: &'a [Vec<u32>],
}

impl<'a> From<&'a SplittersInput> for SplittersInputRef<'a> {
  fn from(splitter_args: &'a SplittersInput) -> Self {
    Self {
      schema: &splitter_args.schema,
      columns: splitter_args.columns.as_slice(),
      indices: splitter_args.indices.as_slice()
    }
  }
}

pub fn splitters_approach<'a>(input: &'a SplittersInputRef<'a>, batch_size: usize, number_of_partitions: usize) -> Output {
  let schema = input.schema;

  crate::splitters::shuffle_by_splitters(ShuffleArgs {
    batch_size,
    number_of_partitions,
    schema,
    indices: input.indices,
    columns: input.columns,
  }).unwrap()
}
