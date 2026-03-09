extern crate core;
#[macro_use]
extern crate criterion;

use bench_shuffle::bench_fns::*;
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
      b.iter(|| take_to_builders_approach_bench(&generated_derive));
    });
  }

  {
    group.bench_function("take_to_builders_column_wise_approach", |b| {
      b.iter(|| take_to_builders_column_wise_approach_bench(&generated_derive));
    });
  }

  {
    group.bench_function("take_approach", |b| {
      b.iter(|| take_approach_bench(&generated_derive));
    });
  }

  {
    group.bench_function("take_column_wise_approach", |b| {
      b.iter(|| take_column_wise_approach_bench(&generated_derive));
    });
  }

  {
    group.bench_function("interleave_approach", |b| {
      b.iter(|| interleave_approach_bench(&generated_derive));
    });
  }

  {
    group.bench_function("interleave_column_wise_approach", |b| {
      b.iter(|| interleave_column_wise_approach_bench(&generated_derive));
    });
  }

  {
    group.bench_function("row_format_approach", |b| {
      b.iter(|| row_format_approach_bench(&generated_derive));
    });
  }

  {
    group.bench_function("row_format_approach going partition wise", |b| {
      b.iter(|| row_format_approach_partition_wise_bench(&generated_derive));
    });
  }

  {
    group.bench_function("optimized_row_format_approach", |b| {
      b.iter(|| optimized_row_format_approach_bench(&generated_derive));
    });
  }

  {
    group.bench_function("optimized_row_format_approach going partition wise", |b| {
      b.iter(|| optimized_row_format_approach_partition_wise_bench(&generated_derive));
    });
  }

  {
    group.bench_function("splitters", |b| {
      b.iter(|| splitters_approach_bench(&generated_derive));
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
