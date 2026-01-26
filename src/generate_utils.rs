
use arrow::array::ArrayRef;
use arrow::datatypes::{Int64Type, UInt64Type};
use arrow::row::{RowConverter, SortField};
use arrow::util::bench_util::{
  create_boolean_array, create_dict_from_values, create_primitive_array,
  create_primitive_array_with_seed, create_string_array_with_len,
  create_string_array_with_len_range_and_prefix_and_seed, create_string_dict_array,
  create_string_view_array_with_len, create_string_view_array_with_max_len,
};
use arrow::util::data_gen::create_random_array;
use arrow_array::types::{Int8Type, Int32Type, UInt8Type, UInt32Type};
use arrow_array::{Array, BooleanArray, Float64Array, RecordBatch};
use arrow_schema::{DataType, Field, FieldRef, Fields, Schema};
use rand::distr::{Distribution, StandardUniform};
use rand::prelude::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use std::{hint, sync::Arc};
// use arrow_select::take::{take, take_arrays};

//
// /// Returns `batches[batch_index][row_index] = partition_index`
// fn generate_partitions(args: GenerateArgs) -> Vec<Vec<usize>> {
//     let mut rng = StdRng::seed_from_u64(args.seed);
//
//     // Create an even number of items per partition to mimic hash partitioning on distributed data
//     let base_partitions: Vec<usize> = (0..args.num_rows)
//         .map(|row_index| row_index % args.num_partitions)
//         .collect();
//
//     (0..args.num_batches)
//         .map(|_| {
//             let mut partitions = base_partitions.clone();
//
//             // Shuffle uniformly to mimic hash partitioning
//             // Using the same rng so next iteration will have different shuffle order
//             partitions.shuffle(&mut rng);
//
//             partitions
//         })
//         .collect()
// }



/// A single benchmark with a medium number of columns (around 50) without nested columns for real-world use cases
/// This also makes sure there is a large gap between each value in the column and how it is laid out in the row format.
/// and it is on the edge of not fitting in L3 on some machines
pub fn generate_batch(
  batch_size: usize,
  seed: &mut u64,
) -> RecordBatch {
  let mut cols: Vec<ArrayRef> = vec![];
  let mut fields: Vec<FieldRef> = vec![];
  // columnar vs row

  // columnar:
  // going column, column and for each value writing in a partition
  // so if we have a column in L1, we partition write in different memory locations having cache misses?

  // If we write in row format, we write all values for a row in one go but we still write in different location and the partitioning have cache misses
  //
  // the current columnar based implementation don't use the column right away but only in the end
  // which means that we need to fetch it again from memory.
  // and when we tested in rows I think we converted to rows right away and stored the rows.
  // and then the partitioning of the rows is much small copies and more larger ones.
  //
  // But converting to row-based still copies around small pieces of memory, except it is sequentially.
  //
  // but if we look at number of iterations.
  // columnar based: for each column

  for nulls in [0.0, 0.1, 0.2, 0.5] {
    *seed += 1;
    let array = Arc::new(create_primitive_array_with_seed::<Int8Type>(
      batch_size, nulls, *seed,
    )) as ArrayRef;

    fields.push(Arc::new(Field::new(
      format!("col_{}", cols.len()),
      array.data_type().clone(),
      nulls != 0.0,

    )));

    cols.push(array);
  }

  for nulls in [0.0, 0.1, 0.2, 0.5] {
    *seed += 1;
    let array = Arc::new(create_primitive_array_with_seed::<Int32Type>(
      batch_size, nulls, *seed,
    )) as ArrayRef;

    fields.push(Arc::new(Field::new(
      format!("col_{}", cols.len()),
      array.data_type().clone(),
      nulls != 0.0,

    )));

    cols.push(array);
  }

  for nulls in [0.0, 0.1, 0.2, 0.5] {
    *seed += 1;
    let array = Arc::new(create_primitive_array_with_seed::<Int64Type>(
      batch_size, nulls, *seed,
    )) as ArrayRef;

    fields.push(Arc::new(Field::new(
      format!("col_{}", cols.len()),
      array.data_type().clone(),
      nulls != 0.0,

    )));

    cols.push(array);
  }

  for _ in 0..10 {
    let nulls = 0.3;

    *seed += 1;
    let array = Arc::new(create_primitive_array_with_seed::<Int64Type>(
      batch_size, nulls, *seed,
    )) as ArrayRef;

    fields.push(Arc::new(Field::new(
      format!("col_{}", cols.len()),
      array.data_type().clone(),
      nulls != 0.0,

    )));

    cols.push(array);
  }

  for nulls in [0.0, 0.1, 0.2, 0.5] {
    *seed += 1;
    let array = Arc::new(create_string_array_with_len_range_and_prefix_and_seed::<i32>(
      batch_size, nulls, 0, 50, "", *seed,
    )) as ArrayRef;

    fields.push(Arc::new(Field::new(
      format!("col_{}", cols.len()),
      array.data_type().clone(),
      nulls != 0.0,
    )));

    cols.push(array);
  }
  //
  for _ in 0..3 {
    let nulls = 0.0;

    *seed += 1;
    let array = Arc::new(create_string_array_with_len_range_and_prefix_and_seed::<i32>(
      batch_size, nulls, 0, 10, "", *seed,
    )) as ArrayRef;

    fields.push(Arc::new(Field::new(
      format!("col_{}", cols.len()),
      array.data_type().clone(),
      nulls != 0.0,
    )));

    cols.push(array);
  }

  for _ in 0..3 {
    let nulls = 0.0;

    *seed += 1;
    let array = Arc::new(create_string_array_with_len_range_and_prefix_and_seed::<i32>(
      batch_size, nulls, 10, 20, "", *seed,
    )) as ArrayRef;

    fields.push(Arc::new(Field::new(
      format!("col_{}", cols.len()),
      array.data_type().clone(),
      nulls != 0.0,
    )));

    cols.push(array);
  }

  for _ in 0..3 {
    let nulls = 0.0;

    *seed += 1;
    let array = Arc::new(create_string_array_with_len_range_and_prefix_and_seed::<i32>(
      batch_size, nulls, 20, 30, "", *seed,
    )) as ArrayRef;

    fields.push(Arc::new(Field::new(
      format!("col_{}", cols.len()),
      array.data_type().clone(),
      nulls != 0.0,
    )));

    cols.push(array);
  }

  for nulls in [0.0, 0.1, 0.2, 0.5] {
    *seed += 1;
    let array = Arc::new(create_boolean_array_with_seed(
      batch_size, nulls, 0.5, *seed,
    )) as ArrayRef;

    fields.push(Arc::new(Field::new(
      format!("col_{}", cols.len()),
      array.data_type().clone(),
      nulls != 0.0,
    )));

    cols.push(array);
  }

  for _ in 0..10 {
    *seed += 1;
    let nulls = 0.0;
    let array = Arc::new(create_primitive_array_with_seed::<Int64Type>(
      batch_size, nulls, *seed,
    )) as ArrayRef;

    fields.push(Arc::new(Field::new(
      format!("col_{}", cols.len()),
      array.data_type().clone(),
      nulls != 0.0,
    )));

    cols.push(array);
  }

  for nulls in [0.0, 0.1, 0.2, 0.5] {
    *seed += 1;
    let array = Arc::new(create_f64_array_with_seed(batch_size, nulls, *seed)) as ArrayRef;

    fields.push(Arc::new(Field::new(
      format!("col_{}", cols.len()),
      array.data_type().clone(),
      nulls != 0.0,
    )));

    cols.push(array);
  }

  let schema = Arc::new(Schema::new(fields));

  RecordBatch::try_new(schema, cols).unwrap()
}


/// Creates a random array of a given size and null density based on the provided seed
pub fn create_boolean_array_with_seed(
  size: usize,
  null_density: f32,
  true_density: f32,
  seed: u64,
) -> BooleanArray
where
  StandardUniform: Distribution<bool>,
{
  let mut rng = StdRng::seed_from_u64(seed);
  (0..size)
    .map(|_| {
      if rng.random::<f32>() < null_density {
        None
      } else {
        let value = rng.random::<f32>() < true_density;
        Some(value)
      }
    })
    .collect()
}

/// Creates a random f64 array of a given size and nan-value density based on a given seed
pub fn create_f64_array_with_seed(size: usize, nan_density: f32, seed: u64) -> Float64Array {
  let mut rng = StdRng::seed_from_u64(seed);

  (0..size)
    .map(|_| {
      if rng.random::<f32>() < nan_density {
        Some(f64::NAN)
      } else {
        Some(rng.random())
      }
    })
    .collect()
}
