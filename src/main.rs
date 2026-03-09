use bench_shuffle::bench_fns::*;

fn main() {
  let start_time = std::time::Instant::now();
  let args: Vec<String> = std::env::args().collect();

  if args.len() < 3 {
    eprintln!("Usage: {} <benchmark_name> <iterations>", args[0]);
    eprintln!("Available benchmarks:");
    eprintln!("  take_to_builders_approach");
    eprintln!("  take_to_builders_column_wise_approach");
    eprintln!("  take_approach");
    eprintln!("  take_column_wise_approach");
    eprintln!("  interleave_approach");
    eprintln!("  interleave_column_wise_approach");
    eprintln!("  row_format_approach");
    eprintln!("  row_format_approach_partition_wise");
    eprintln!("  optimized_row_format_approach");
    eprintln!("  optimized_row_format_approach_partition_wise");
    eprintln!("  splitters");
    std::process::exit(1);
  }

  let benchmark_name = &args[1];
  let iterations: usize = args[2].parse().expect("iterations must be a valid number");
  let number_of_partitions = 1000;
  let batch_size = 8192;
  let number_of_batches = 24;

  let generate = Generate::new(GenerateArgs {
    num_partitions: number_of_partitions,
    num_rows: batch_size,
    num_batches: number_of_batches,
    seed: 42,
  });
  let generated_derive = generate.derived();

  let bench_fn: fn(&GenerateInputsDerived) = match benchmark_name.as_str() {
    "take_to_builders_approach" => take_to_builders_approach_bench,
    "take_to_builders_column_wise_approach" => take_to_builders_column_wise_approach_bench,
    "take_approach" => take_approach_bench,
    "take_column_wise_approach" => take_column_wise_approach_bench,
    "interleave_approach" => interleave_approach_bench,
    "interleave_column_wise_approach" => interleave_column_wise_approach_bench,
    "row_format_approach" => row_format_approach_bench,
    "row_format_approach_partition_wise" => row_format_approach_partition_wise_bench,
    "optimized_row_format_approach" => optimized_row_format_approach_bench,
    "optimized_row_format_approach_partition_wise" => optimized_row_format_approach_partition_wise_bench,
    "splitters" => splitters_approach_bench,
    _ => {
      eprintln!("Unknown benchmark: {}", benchmark_name);
      std::process::exit(1);
    }
  };

  println!("Running '{}' for {} iterations...", benchmark_name, iterations);

  println!("sleeping until reached 3s since start to make sure benchmark setup is not included in the benchmark time");

  let elapsed = start_time.elapsed();
  if elapsed.as_secs() > 3 {
    panic!("Benchmark setup took too long: {}s", elapsed.as_secs());
  }

  std::thread::sleep(std::time::Duration::from_millis(3000 - elapsed.as_millis() as u64));

  for _ in 0..iterations {
    bench_fn(&generated_derive);
  }

  println!("Done.");
}
