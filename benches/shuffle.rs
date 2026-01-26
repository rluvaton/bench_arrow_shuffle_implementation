extern crate core;
#[macro_use]
extern crate criterion;

use arrow_array::{Int32Array, RecordBatch, UInt32Array};
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
  let inputs_refs_slice = inputs_refs_vec.as_slice();

  {
    let mut group = c.benchmark_group("shuffle_take_approach");
    group.bench_function("take_approach", |b| {
      b.iter(|| {
        let output = take_to_builders_approach(inputs_refs_slice, batch_size, number_of_partitions);
        hint::black_box(output);
      });
    });
    group.finish();
  }

  {
    let mut group = c.benchmark_group("shuffle_row_format_approach");
    group.bench_function("row_format_approach", |b| {
      b.iter(|| {
        let output = row_format_approach(inputs_refs_slice, batch_size, number_of_partitions);
        hint::black_box(output);
      });
    });
    group.finish();
  }

  {
    let mut group = c.benchmark_group("shuffle_row_format_approach");
    group.bench_function("row_format_approach going partition wise", |b| {
      b.iter(|| {
        let output = row_format_approach_partition_wise(inputs_refs_slice, batch_size, number_of_partitions);
        hint::black_box(output);
      });
    });
    group.finish();
  }

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
    let mut group = c.benchmark_group("shuffle_optimized_row_format_approach");
    group.bench_function("optimized_row_format_approach", |b| {
      b.iter(|| {
        let output = optimized_row_format_approach(inputs_refs_slice, batch_size, number_of_partitions);
        hint::black_box(output);
      });
    });
    group.finish();
  }

  {
    let mut group = c.benchmark_group("shuffle_optimized_row_format_approach");
    group.bench_function("optimized_row_format_approach going partition wise", |b| {
      b.iter(|| {
        let output = optimized_row_format_approach_partition_wise(inputs_refs_slice, batch_size, number_of_partitions);
        hint::black_box(output);
      });
    });
    group.finish();
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

/// Take and concat
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
fn optimized_row_format_approach<'a>(input: &'a [InputRef<'a>], batch_size: usize, number_of_partitions: usize) -> Output {
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
fn optimized_row_format_approach_partition_wise<'a>(input: &'a [InputRef<'a>], batch_size: usize, number_of_partitions: usize) -> Output {
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

criterion_group!(benches, run_benchmark);
criterion_main!(benches);
