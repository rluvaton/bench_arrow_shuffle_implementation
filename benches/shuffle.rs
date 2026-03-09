extern crate core;
#[macro_use]
extern crate criterion;

use bench_shuffle::bench_fns::{interleave_approach, interleave_column_wise_approach, optimized_row_format_approach, optimized_row_format_approach_partition_wise, row_format_approach, row_format_approach_partition_wise, splitters_approach, take_approach, take_column_wise_approach, take_to_builders_approach, take_to_builders_column_wise_approach, Generate, GenerateArgs};
use criterion::Criterion;
use std::hint;

fn run_benchmark(c: &mut Criterion) {
  let number_of_partitions = 1000;
  let batch_size = 8192;
  let number_of_batches = 24;

  // this will create output batches of size ~700

  let generate = Generate::new(GenerateArgs {
    num_partitions: number_of_partitions,
    num_rows: batch_size,
    num_batches: number_of_batches,
    seed: 42,
  });
  let generated_derive = generate.derived();

  let mut group = c.benchmark_group("shuffle");


  {
    group.bench_function("take_to_builders_approach", |b| {
      b.iter(|| {
        let output = take_to_builders_approach(generated_derive.inputs_ref.as_slice(), batch_size, number_of_partitions);
        hint::black_box(output);
      });
    });
  }

  {
    group.bench_function("take_to_builders_column_wise_approach", |b| {
      b.iter(|| {
        let output = take_to_builders_column_wise_approach(generated_derive.inputs_ref.as_slice(), &generated_derive.input_columns, batch_size, number_of_partitions);
        hint::black_box(output);
      });
    });
  }

  {
    group.bench_function("take_approach", |b| {
      b.iter(|| {
        let output = take_approach(generated_derive.inputs_ref.as_slice(), batch_size, number_of_partitions);
        hint::black_box(output);
      });
    });
  }

  {
    group.bench_function("take_column_wise_approach", |b| {
      b.iter(|| {
        let output = take_column_wise_approach(generated_derive.inputs_ref.as_slice(), &generated_derive.input_columns, batch_size, number_of_partitions);
        hint::black_box(output);
      });
    });
  }
  
  {
    group.bench_function("interleave_approach", |b| {
      b.iter(|| {
        let output = interleave_approach(&generated_derive.interleave_optimized_input, batch_size, number_of_partitions);
        hint::black_box(output);
      });
    });
  }
  
  {
    group.bench_function("interleave_column_wise_approach", |b| {
      b.iter(|| {
        let output = interleave_column_wise_approach(generated_derive.inputs_ref.as_slice(), &generated_derive.interleave_column_wise_optimized_input, batch_size, number_of_partitions);
        hint::black_box(output);
      });
    });
  }
  
  {
    group.bench_function("row_format_approach", |b| {
      b.iter(|| {
        let output = row_format_approach(generated_derive.inputs_ref.as_slice(), batch_size, number_of_partitions);
        hint::black_box(output);
      });
    });
  }

  {
    group.bench_function("row_format_approach going partition wise", |b| {
      b.iter(|| {
        let output = row_format_approach_partition_wise(generated_derive.inputs_ref.as_slice(), batch_size, number_of_partitions);
        hint::black_box(output);
      });
    });
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
    group.bench_function("optimized_row_format_approach", |b| {
      b.iter(|| {
        let output = optimized_row_format_approach(generated_derive.inputs_ref.as_slice(), batch_size, number_of_partitions);
        hint::black_box(output);
      });
    });
  }

  {
    group.bench_function("optimized_row_format_approach going partition wise", |b| {
      b.iter(|| {
        let output = optimized_row_format_approach_partition_wise(generated_derive.inputs_ref.as_slice(), batch_size, number_of_partitions);
        hint::black_box(output);
      });
    });
  }

  {
    group.bench_function("splitters", |b| {
      b.iter(|| {
        let output = splitters_approach(&generated_derive.splitter_args_ref, batch_size, number_of_partitions);
        hint::black_box(output);
      });
    });
  }

}

criterion_group!{
    name = benches;
    config = Criterion::default().sample_size(10);
    targets = run_benchmark
}
// criterion_group!(benches, run_benchmark);
criterion_main!(benches);
