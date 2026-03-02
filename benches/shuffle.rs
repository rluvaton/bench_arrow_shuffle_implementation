extern crate core;
#[macro_use]
extern crate criterion;

use arrow_array::{Array, ArrayRef, Int32Array, RecordBatch, UInt32Array};
// use arrow_row::unordered_row::UnorderedRowConverter;
use bench_shuffle::generate_utils::generate_batch;
use criterion::Criterion;
use rand::prelude::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use std::hint;
use std::sync::Arc;
use arrow_array::builder::{Int32Builder, UInt32Builder};
use arrow_row::SortField;

fn run_benchmark(c: &mut Criterion) {
  let number_of_partitions = 1500;
  let batch_size = 8192;
  let number_of_batches = 128;

  // this will create output batches of size ~700

  let inputs = generate_inputs(GenerateArgs {
    num_partitions: number_of_partitions,
    num_rows: batch_size,
    num_batches: number_of_batches,
    seed: 42,
  });

  let inputs_refs_vec: Vec<InputRef<'_>> = inputs.iter().map(|x| InputRef { batch: &x.batch, partitions: &x.partitions, indices_per_partition: &x.indices_per_partition }).collect();
  let input_columns: InputColumns = InputColumns {
    columns: {
      let number_of_columns = inputs[0].batch.num_columns();
      let mut columns = vec![vec![]; number_of_columns];

      for column_index in 0..number_of_columns {
        for batch in &inputs {
          columns[column_index].push(SingleColumnInput {
            column: &batch.batch.column(column_index),
            indices_per_partition: &batch.indices_per_partition
          });
        }
      }

      columns
    }
  };
  let interleave_optimized_input: InterleaveOptimizedInput<'_> = {
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
    InterleaveOptimizedInput {
      batches,
      partitions: partition_indices
    }
  };
  let interleave_column_wise_optimized_input: InterleaveColumnWiseOptimizedInput<'_> = {
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
    InterleaveColumnWiseOptimizedInput {
      columns,
      partitions: partition_indices
    }
  };
  let inputs_refs_slice = inputs_refs_vec.as_slice();

  let mut group = c.benchmark_group("shuffle");

  //
  // {
  //   group.bench_function("take_to_builders_approach", |b| {
  //     b.iter(|| {
  //       let output = take_to_builders_approach(inputs_refs_slice, batch_size, number_of_partitions);
  //       hint::black_box(output);
  //     });
  //   });
  // }
  //
  // {
  //   group.bench_function("take_to_builders_column_wise_approach", |b| {
  //     b.iter(|| {
  //       let output = take_to_builders_column_wise_approach(inputs_refs_slice, &input_columns, batch_size, number_of_partitions);
  //       hint::black_box(output);
  //     });
  //   });
  // }
  //
  // {
  //   group.bench_function("take_approach", |b| {
  //     b.iter(|| {
  //       let output = take_approach(inputs_refs_slice, batch_size, number_of_partitions);
  //       hint::black_box(output);
  //     });
  //   });
  // }
  //
  // {
  //   group.bench_function("take_column_wise_approach", |b| {
  //     b.iter(|| {
  //       let output = take_column_wise_approach(inputs_refs_slice, &input_columns, batch_size, number_of_partitions);
  //       hint::black_box(output);
  //     });
  //   });
  // }
  //
  // {
  //   group.bench_function("interleave_approach", |b| {
  //     b.iter(|| {
  //       let output = interleave_approach(&interleave_optimized_input, batch_size, number_of_partitions);
  //       hint::black_box(output);
  //     });
  //   });
  // }
  //
  // {
  //   group.bench_function("interleave_column_wise_approach", |b| {
  //     b.iter(|| {
  //       let output = interleave_column_wise_approach(inputs_refs_slice, &interleave_column_wise_optimized_input, batch_size, number_of_partitions);
  //       hint::black_box(output);
  //     });
  //   });
  // }
  //
  // {
  //   group.bench_function("row_format_approach", |b| {
  //     b.iter(|| {
  //       let output = row_format_approach(inputs_refs_slice, batch_size, number_of_partitions);
  //       hint::black_box(output);
  //     });
  //   });
  // }
  //
  // {
  //   group.bench_function("row_format_approach going partition wise", |b| {
  //     b.iter(|| {
  //       let output = row_format_approach_partition_wise(inputs_refs_slice, batch_size, number_of_partitions);
  //       hint::black_box(output);
  //     });
  //   });
  // }

  // for start in 0..inputs_refs_slice[0].batch.num_columns() {
  //   for end in (start + 1)..=inputs_refs_slice[0].batch.num_columns() {
  //     let project_indices = (start..end).collect::<Vec<_>>();
  //     let projected_batch = inputs_refs_slice[0].batch.project(&project_indices).unwrap();
  //     println!("start: {}, end: {}", start, end);
  //     println!("projected batch columns types: {:?}", projected_batch.schema_ref());
  //     let input = Input {
  //       batch: projected_batch,
  //       partitions: inputs_refs_slice[0].partitions.to_vec(),
  //       indices_per_partition: inputs_refs_slice[0].indices_per_partition.to_vec(),
  //     };
  //     let input_ref = input.as_ref();
  //
  //     test_combination_that_fail(&[input_ref], batch_size, number_of_partitions);
  //   }
  // }
  {
    group.bench_function("encode", |b| {
      let schema = inputs[0].batch.schema_ref();
      let fields = schema.fields();
      let row_converter = bench_shuffle::unordered_row::UnorderedRowConverter::new(
        fields.clone(),
      ).expect("should be able to create row converter");
      b.iter(|| {
        for batch in inputs_refs_slice {
          let output = row_converter.convert_columns::<false>(batch.batch.columns()).expect("should be able to convert columns");
          hint::black_box(output);

        }
      });
    });
  }
  {
    group.bench_function("encode multiple at a time", |b| {
      let schema = inputs[0].batch.schema_ref();
      let fields = schema.fields();
      let row_converter = bench_shuffle::unordered_row::UnorderedRowConverter::new(
        fields.clone(),
      ).expect("should be able to create row converter");
      b.iter(|| {
        for batch in inputs_refs_slice {
          let output = row_converter.convert_columns::<true>(batch.batch.columns()).expect("should be able to convert columns");
          hint::black_box(output);

        }
      });
    });
  }

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

struct GenerateArgs {
  num_partitions: usize,
  num_rows: usize,
  num_batches: usize,
  seed: u64,
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

impl Input {
  fn as_ref<'a>(&'a self) -> InputRef<'a> {
    InputRef { batch: &self.batch, partitions: &self.partitions, indices_per_partition: &self.indices_per_partition }
  }
}

struct InputRef<'a> {
  batch: &'a RecordBatch,
  /// `partition[row_index] = partition_index`
  partitions: &'a [usize],

  /// Indices per partition partition[partition_index] = indices in the batch for that partition
  indices_per_partition: &'a [UInt32Array]
}

/// The output is `output[partition_index][batches]`
type Output = Vec<Vec<RecordBatch>>;

fn take_to_builders_approach<'a>(input: &'a [InputRef<'a>], batch_size: usize, number_of_partitions: usize) -> Output {
  let fields = input[0].batch.schema_ref().fields();
  let mut partitions_sink = (0..number_of_partitions).map(|_| bench_shuffle::take::create_sinks(fields, batch_size)).collect::<Vec<_>>();

  for input in input.iter() {
    for (indices, sinks) in input.indices_per_partition.iter().zip(partitions_sink.iter_mut()) {
      bench_shuffle::take::take_to_sinks(input.batch.columns(), sinks, indices).expect("should be able to take");
    }
  }

  let output = partitions_sink
    .iter_mut()
    .map(|sinks| {
      let batch = bench_shuffle::take::finish_sinks(
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

fn take_to_builders_column_wise_approach<'a>(input: &'a [InputRef<'a>], columns_based: &'a InputColumns<'a>, batch_size: usize, number_of_partitions: usize) -> Output {
  let fields = input[0].batch.schema_ref().fields();

  // columns_sink[column_index][partition_index] = sink for that column in that partition
  let mut columns_sink = fields.iter().map(|f| (0..number_of_partitions).map(|_| bench_shuffle::take::create_sink(f.as_ref(), batch_size)).collect::<Vec<_>>()).collect::<Vec<_>>();

  for (column_and_indices, partition_sinks) in columns_based.columns.iter().zip(columns_sink.iter_mut()) {
    for SingleColumnInput { column, indices_per_partition } in column_and_indices {
      for (indices, sink) in indices_per_partition.iter().zip(partition_sinks.iter_mut()) {
        bench_shuffle::take::take_to_sink(column.as_ref(), sink, indices).expect("should be able to take");
      }
    }
  }

  // columns[partition_index][column_index] = column values for that partition
  let mut output_partitions = vec![Vec::with_capacity(fields.len()); number_of_partitions];

  for (partitions, field) in columns_sink.iter_mut().zip(fields.iter()) {
    for (partition_index, sink) in partitions.iter_mut().enumerate() {
      let column = bench_shuffle::take::finish_sink(
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



fn take_approach<'a>(input: &'a [InputRef<'a>], batch_size: usize, number_of_partitions: usize) -> Output {
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


fn take_column_wise_approach<'a>(input: &'a [InputRef<'a>], columns_based: &'a InputColumns<'a>, batch_size: usize, number_of_partitions: usize) -> Output {
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

struct InterleaveOptimizedInput<'a> {
  batches: Vec<&'a RecordBatch>,
  /// `indices[partition_index][output_row_index] = (batch_index, row_index)`
  partitions: Vec<Vec<(usize, usize)>>,
}

fn interleave_approach<'a>(input: &'a InterleaveOptimizedInput<'a>, batch_size: usize, number_of_partitions: usize) -> Output {
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

struct InterleaveColumnWiseOptimizedInput<'a> {
  /// `columns[column_index][batch_index] = column array for that batch`
  columns: Vec<Vec<&'a dyn Array>>,
  /// `indices[partition_index][output_row_index] = (batch_index, row_index)`
  partitions: Vec<Vec<(usize, usize)>>,
}

fn interleave_column_wise_approach<'a>(input: &'a [InputRef<'a>], interleave_input: &'a InterleaveColumnWiseOptimizedInput<'a>, batch_size: usize, number_of_partitions: usize) -> Output {
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
fn row_format_approach<'a>(input: &'a [InputRef<'a>], batch_size: usize, number_of_partitions: usize) -> Output {
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
fn row_format_approach_partition_wise<'a>(input: &'a [InputRef<'a>], batch_size: usize, number_of_partitions: usize) -> Output {
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
  let row_converter = bench_shuffle::unordered_row::UnorderedRowConverter::new(
    fields.clone(),
  ).expect("should be able to create row converter");

  let mut partitions_sink = (0..number_of_partitions).map(|_|
    // trying to reserve
    row_converter.empty_rows(batch_size, batch_size * fields.len() * 50)
  ).collect::<Vec<_>>();

  for input in input.iter() {
    let rows = row_converter.convert_columns::<false>(input.batch.columns()).expect("should be able to convert");

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
fn optimized_row_format_approach<'a, const ENCODE_MULTI_COLUMNS_AT_ONCE: bool>(input: &'a [InputRef<'a>], batch_size: usize, number_of_partitions: usize) -> Output {
  let schema = input[0].batch.schema_ref();
  let fields = schema.fields();
  let row_converter = bench_shuffle::unordered_row::UnorderedRowConverter::new(
    fields.clone(),
  ).expect("should be able to create row converter");

  let mut partitions_sink = (0..number_of_partitions).map(|_|
    // trying to reserve
    row_converter.empty_rows(batch_size, batch_size * fields.len() * 50)
  ).collect::<Vec<_>>();

  for input in input.iter() {
    let rows = row_converter.convert_columns::<ENCODE_MULTI_COLUMNS_AT_ONCE>(input.batch.columns()).expect("should be able to convert");

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
fn optimized_row_format_approach_partition_wise<'a, const ENCODE_MULTI_COLUMNS_AT_ONCE: bool>(input: &'a [InputRef<'a>], batch_size: usize, number_of_partitions: usize) -> Output {
  let schema = input[0].batch.schema_ref();
  let fields = schema.fields();
  let row_converter = bench_shuffle::unordered_row::UnorderedRowConverter::new(
    fields.clone(),
  ).expect("should be able to create row converter");

  let mut partitions_sink = (0..number_of_partitions).map(|_|
    // trying to reserve
    row_converter.empty_rows(batch_size, batch_size * fields.len() * 50)
  ).collect::<Vec<_>>();

  for input in input.iter() {
    let rows = row_converter.convert_columns::<ENCODE_MULTI_COLUMNS_AT_ONCE>(input.batch.columns()).expect("should be able to convert");

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

criterion_group!{
    name = benches;
    config = Criterion::default().sample_size(10);
    targets = run_benchmark
}
// criterion_group!(benches, run_benchmark);
criterion_main!(benches);
