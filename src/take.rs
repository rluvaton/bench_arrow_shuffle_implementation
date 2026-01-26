// take implementation that
use arrow_array::builder::UInt32Builder;
use arrow_array::cast::AsArray;
use arrow_array::types::*;
use arrow_array::*;
use arrow_buffer::{
    ArrowNativeType, BooleanBuffer, BooleanBufferBuilder, Buffer, BufferBuilder, MutableBuffer,
    NullBuffer, NullBufferBuilder, OffsetBuffer, ScalarBuffer, bit_util,
};
use arrow_data::ArrayDataBuilder;
use arrow_schema::*;
use num_traits::{One, Zero};
use std::fmt::Display;
use std::sync::Arc;

/// Take sink
pub trait Sink {
    fn as_any(&self) -> &dyn std::any::Any;

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any;

    fn finish(&mut self, field: &Field) -> ArrayRef;
}

struct PrimitiveSink<T: ArrowPrimitiveType> {
    data_type: DataType,
    values: Vec<T::Native>,
    nulls: Option<NullSink>,
}

impl<T: ArrowPrimitiveType> Sink for PrimitiveSink<T> {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn finish(&mut self, f: &Field) -> ArrayRef {
        let values = std::mem::take(&mut self.values);

        let array = PrimitiveArray::<T>::new(
            ScalarBuffer::from(values),
            NullSink::finish_option(&mut self.nulls),
        )
        .with_data_type(f.data_type().clone());

        Arc::new(array) as ArrayRef
    }
}

struct BooleanSink {
    inner: BooleanBufferSink,
    nulls: Option<NullSink>,
}

impl BooleanSink {
    fn new(field: &Field, batch_size: usize) -> Self {
        assert_eq!(field.data_type(), &DataType::Boolean);
        Self {
            nulls: if field.is_nullable() {
                Some(NullSink::new(batch_size))
            } else {
                None
            },
            inner: BooleanBufferSink {
                buffer: BooleanBufferBuilder::new(batch_size),
            },
        }
    }
}

impl Sink for BooleanSink {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn finish(&mut self, field: &Field) -> ArrayRef {
        assert_eq!(field.data_type(), &DataType::Boolean);

        let array = BooleanArray::new(
            self.inner.buffer.finish(),
            NullSink::finish_option(&mut self.nulls),
        );

        Arc::new(array) as ArrayRef
    }
}

// TODO - init offset with 0
// TODO - add new with capacity
struct GenericByteSink<T: ByteArrayType> {
    nulls: Option<NullSink>,
    offsets: Vec<T::Offset>,
    bytes: Vec<u8>,
}

impl<T: ByteArrayType> GenericByteSink<T> {
    fn new(f: &Field, batch_size: usize) -> Self {
        Self {
            nulls: if f.is_nullable() {
                Some(NullSink::new(batch_size))
            } else {
                None
            },
            offsets: {
                let mut offsets = Vec::<T::Offset>::with_capacity(batch_size + 1);

                offsets.push(T::Offset::usize_as(0));

                offsets
            },

            // Reserve 50 bytes per item
            bytes: Vec::with_capacity(batch_size * 50),
        }
    }
}

impl<T: ByteArrayType> Sink for GenericByteSink<T> {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn finish(&mut self, field: &Field) -> ArrayRef {
        assert_eq!(field.data_type(), &T::DATA_TYPE);
        // TODO - replace with correct size?
        let mut current = std::mem::replace(self, Self::new(field, 0));

        let array = unsafe {
            GenericByteArray::<T>::new_unchecked(
                OffsetBuffer::new(ScalarBuffer::from(current.offsets)),
                Buffer::from(current.bytes),
                NullSink::finish_option(&mut current.nulls),
            )
        };

        Arc::new(array) as ArrayRef
    }
}

struct NullSink {
    inner: NullBufferBuilder,
}

impl NullSink {
    fn new(capacity: usize) -> Self {
        Self {
            inner: NullBufferBuilder::new(capacity),
        }
    }

    fn finish(mut self) -> Option<NullBuffer> {
        self.inner.finish().filter(|a| a.null_count() > 0)
    }

    fn finish_boxed(mut self: Box<Self>) -> Option<NullBuffer> {
        self.inner.finish().filter(|a| a.null_count() > 0)
    }

    fn finish_option(value: &mut Option<NullSink>) -> Option<NullBuffer> {
        value
            .take()
            .and_then(|mut n| n.finish())
            .filter(|a| a.null_count() > 0)
    }
}

struct BooleanBufferSink {
    buffer: BooleanBufferBuilder,
}

/// Take elements by index from [Array], creating a new [Array] from those indexes.
///
/// ```text
/// ┌─────────────────┐      ┌─────────┐                              ┌─────────────────┐
/// │        A        │      │    0    │                              │        A        │
/// ├─────────────────┤      ├─────────┤                              ├─────────────────┤
/// │        D        │      │    2    │                              │        B        │
/// ├─────────────────┤      ├─────────┤   take(values, indices)      ├─────────────────┤
/// │        B        │      │    3    │ ─────────────────────────▶   │        C        │
/// ├─────────────────┤      ├─────────┤                              ├─────────────────┤
/// │        C        │      │    1    │                              │        D        │
/// ├─────────────────┤      └─────────┘                              └─────────────────┘
/// │        E        │
/// └─────────────────┘
///    values array          indices array                              result
/// ```
///
/// For selecting values by index from multiple arrays see [`crate::interleave`]
///
/// Note that this kernel, similar to other kernels in this crate,
/// will avoid allocating where not necessary. Consequently
/// the returned array may share buffers with the inputs
///
/// # Errors
/// This function errors whenever:
/// * An index cannot be casted to `usize` (typically 32 bit architectures)
/// * An index is out of bounds and `options` is set to check bounds.
///
/// # Safety
///
/// When `options` is not set to check bounds, taking indexes after `len` will panic.
///
/// # See also
/// * [`BatchCoalescer`]: to filter multiple [`RecordBatch`] and coalesce
///   the results into a single array.
///
/// [`BatchCoalescer`]: crate::coalesce::BatchCoalescer
///
/// # Examples
/// ```
/// # use arrow_array::{StringArray, UInt32Array, cast::AsArray};
/// # use arrow_select::take::take;
/// let values = StringArray::from(vec!["zero", "one", "two"]);
///
/// // Take items at index 2, and 1:
/// let indices = UInt32Array::from(vec![2, 1]);
/// let taken = take(&values, &indices, None).unwrap();
/// let taken = taken.as_string::<i32>();
///
/// assert_eq!(*taken, StringArray::from(vec!["two", "one"]));
/// ```
// pub fn take(
//     values: &dyn Array,
//     indices: &dyn Array,
//     options: Option<arrow_select::take::TakeOptions>,
// ) -> Result<ArrayRef, ArrowError> {
//     let options = options.unwrap_or_default();
//     downcast_integer_array!(
//         indices => {
//             if options.check_bounds {
//                 check_bounds(values.len(), indices)?;
//             }
//             let indices = indices.to_indices();
//             take_impl(values, &indices)
//         },
//         d => Err(ArrowError::InvalidArgumentError(format!("Take only supported for integers, got {d:?}")))
//     )
// }
pub fn take_to_sinks(
    arrays: &[ArrayRef],
    sinks: &mut [Box<dyn Sink>],
    indices: &dyn Array,
) -> Result<(), ArrowError> {
    downcast_integer_array!(
        indices => {
            let indices = indices.to_indices();
            sinks.iter_mut().zip(arrays).for_each(|(sink, values)| {
                take_impl(values.as_ref(), &indices, sink.as_mut())
            });

            Ok(())
        },
        d => Err(ArrowError::InvalidArgumentError(format!("Take only supported for integers, got {d:?}")))
    )
}

pub fn take_to_sink(
    array: &dyn Array,
    sink: &mut Box<dyn Sink>,
    indices: &dyn Array,
) -> Result<(), ArrowError> {
    downcast_integer_array!(
        indices => {
            let indices = indices.to_indices();
            take_impl(array, &indices, sink.as_mut());

            Ok(())
        },
        d => Err(ArrowError::InvalidArgumentError(format!("Take only supported for integers, got {d:?}")))
    )
}

pub fn create_sinks(fields: &Fields, batch_size: usize) -> Vec<Box<dyn Sink>> {
    fields
      .iter()
      .map(|f| create_sink(f, batch_size))
      .collect()
}

pub fn create_sink(field: &Field, batch_size: usize) -> Box<dyn Sink> {
    macro_rules! primitive_size_helper {
        ($t:ty) => {
            create_primitive_sink::<$t>(field, batch_size)
        };
    }
    let dt = field.data_type();
    downcast_primitive!(
        dt => (primitive_size_helper),
        DataType::Boolean => {
            Box::new(BooleanSink::new(field.as_ref(), batch_size)) as Box<dyn Sink>
        },
        DataType::Utf8 => {
            Box::new(GenericByteSink::<Utf8Type>::new(field.as_ref(), batch_size))
        },
        dt => unimplemented!("not implemented create sink for {dt:?}")
    )
}

fn create_primitive_sink<T: ArrowPrimitiveType>(field: &Field, batch_size: usize) -> Box<dyn Sink> {
    Box::new(PrimitiveSink::<T> {
        data_type: field.data_type().clone(),
        nulls: if field.is_nullable() {
            Some(NullSink::new(batch_size))
        } else {
            None
        },
        values: Vec::with_capacity(batch_size),
    })
}

pub fn finish_sinks(fields: &Fields, sinks: &mut [Box<dyn Sink>], batch_size: usize) -> RecordBatch {
    let columns = fields
      .iter()
      .zip(sinks)
      .map(|(f, sink)| {
          finish_sink(f.as_ref(), sink, batch_size)
      })
      .collect::<Vec<ArrayRef>>();

    RecordBatch::try_new(Arc::new(Schema::new(fields.clone())), columns)
      .expect("should be able to create record batch")
}

pub fn finish_sink(field: &Field, sink: &mut Box<dyn Sink>, batch_size: usize) -> ArrayRef {
    macro_rules! primitive_size_helper {
        ($t:ty) => {
            finish_primitive_sink::<$t>(field, sink)
        };
    }

  let dt = field.data_type();
  downcast_primitive!(
        dt => (primitive_size_helper),
        DataType::Boolean => finish_boolean_sink(field, sink),
        DataType::Utf8 => finish_byte_sink::<Utf8Type>(field, sink),
        dt => unimplemented!("not implemented create sink for {dt:?}")
    )
}

fn finish_primitive_sink<T: ArrowPrimitiveType>(
    field: &Field,
    sink: &mut Box<dyn Sink>,
) -> ArrayRef {
    sink.as_any_mut()
        .downcast_mut::<PrimitiveSink<T>>()
        .map(|primitive_sink| primitive_sink.finish(field))
        .expect("should be a primitive sink")
}

fn finish_boolean_sink(field: &Field, sink: &mut Box<dyn Sink>) -> ArrayRef {
    sink.as_any_mut()
      .downcast_mut::<BooleanSink>()
      .map(|sink| sink.finish(field))
      .expect("should be a primitive sink")
}

fn finish_byte_sink<T: ByteArrayType>(field: &Field, sink: &mut Box<dyn Sink>) -> ArrayRef {
    sink.as_any_mut()
      .downcast_mut::<GenericByteSink<T>>()
      .map(|sink| sink.finish(field))
      .expect("should be a primitive sink")
}


/// Verifies that the non-null values of `indices` are all `< len`
fn check_bounds<T: ArrowPrimitiveType>(
    len: usize,
    indices: &PrimitiveArray<T>,
) -> Result<(), ArrowError>
where
    T::Native: Display,
{
    let len = match T::Native::from_usize(len) {
        Some(len) => len,
        None => {
            if T::DATA_TYPE.is_integer() {
                // the biggest representable value for T::Native is lower than len, e.g: u8::MAX < 512, no need to check bounds
                return Ok(());
            } else {
                return Err(ArrowError::ComputeError("Cast to usize failed".to_string()));
            }
        }
    };

    if indices.null_count() > 0 {
        indices.iter().flatten().try_for_each(|index| {
            if index >= len {
                return Err(ArrowError::ComputeError(format!(
                    "Array index out of bounds, cannot get item at index {index} from {len} entries"
                )));
            }
            Ok(())
        })
    } else {
        let in_bounds = indices.values().iter().fold(true, |in_bounds, &i| {
            in_bounds & (i >= T::Native::ZERO) & (i < len)
        });

        if !in_bounds {
            for &index in indices.values() {
                if index < T::Native::ZERO || index >= len {
                    return Err(ArrowError::ComputeError(format!(
                        "Array index out of bounds, cannot get item at index {index} from {len} entries"
                    )));
                }
            }
        }

        Ok(())
    }
}

#[inline(never)]
fn take_impl<IndexType: ArrowPrimitiveType>(
    values: &dyn Array,
    indices: &PrimitiveArray<IndexType>,
    sink: &mut dyn Sink,
) {
    downcast_primitive_array! {
        values => take_primitive(values, indices, sink.as_any_mut().downcast_mut().unwrap()),
        DataType::Boolean => {
            let values = values.as_any().downcast_ref::<BooleanArray>().unwrap();
            take_boolean(values, indices, sink.as_any_mut().downcast_mut().unwrap())
        }
        DataType::Utf8 => {
            take_bytes(values.as_string::<i32>(), indices, sink.as_any_mut().downcast_mut().unwrap())
        }
        DataType::LargeUtf8 => {
            take_bytes(values.as_string::<i64>(), indices, sink.as_any_mut().downcast_mut().unwrap())
        }
        // DataType::Utf8View => {
        //     Ok(Arc::new(take_byte_view(values.as_string_view(), indices)?))
        // }
        // DataType::List(_) => {
        //     Ok(Arc::new(take_list::<_, Int32Type>(values.as_list(), indices)?))
        // }
        // DataType::LargeList(_) => {
        //     Ok(Arc::new(take_list::<_, Int64Type>(values.as_list(), indices)?))
        // }
        // DataType::ListView(_) => {
        //     Ok(Arc::new(take_list_view::<_, Int32Type>(values.as_list_view(), indices)?))
        // }
        // DataType::LargeListView(_) => {
        //     Ok(Arc::new(take_list_view::<_, Int64Type>(values.as_list_view(), indices)?))
        // }
        // DataType::FixedSizeList(_, length) => {
        //     let values = values
        //         .as_any()
        //         .downcast_ref::<FixedSizeListArray>()
        //         .unwrap();
        //     Ok(Arc::new(take_fixed_size_list(
        //         values,
        //         indices,
        //         *length as u32,
        //     )?))
        // }
        // DataType::Map(_, _) => {
        //     let list_arr = ListArray::from(values.as_map().clone());
        //     let list_data = take_list::<_, Int32Type>(&list_arr, indices)?;
        //     let builder = list_data.into_data().into_builder().data_type(values.data_type().clone());
        //     Ok(Arc::new(MapArray::from(unsafe { builder.build_unchecked() })))
        // }
        // DataType::Struct(fields) => {
        //     let array: &StructArray = values.as_struct();
        //     let arrays  = array
        //         .columns()
        //         .iter()
        //         .map(|a| take_impl(a.as_ref(), indices))
        //         .collect::<Result<Vec<ArrayRef>, _>>()?;
        //     let fields: Vec<(FieldRef, ArrayRef)> =
        //         fields.iter().cloned().zip(arrays).collect();
        //
        //     // Create the null bit buffer.
        //     let is_valid: Buffer = indices
        //         .iter()
        //         .map(|index| {
        //             if let Some(index) = index {
        //                 array.is_valid(index.to_usize().unwrap())
        //             } else {
        //                 false
        //             }
        //         })
        //         .collect();
        //
        //     if fields.is_empty() {
        //         let nulls = NullBuffer::new(BooleanBuffer::new(is_valid, 0, indices.len()));
        //         Ok(Arc::new(StructArray::new_empty_fields(indices.len(), Some(nulls))))
        //     } else {
        //         Ok(Arc::new(StructArray::from((fields, is_valid))) as ArrayRef)
        //     }
        // }
        // DataType::Dictionary(_, _) => downcast_dictionary_array! {
        //     values => Ok(Arc::new(take_dict(values, indices)?)),
        //     t => unimplemented!("Take not supported for dictionary type {:?}", t)
        // }
        // DataType::RunEndEncoded(_, _) => downcast_run_array! {
        //     values => Ok(Arc::new(take_run(values, indices)?)),
        //     t => unimplemented!("Take not supported for run type {:?}", t)
        // }
        DataType::Binary => {
            take_bytes(values.as_binary::<i32>(), indices, sink.as_any_mut().downcast_mut().unwrap())
        }
        DataType::LargeBinary => {
            take_bytes(values.as_binary::<i64>(), indices, sink.as_any_mut().downcast_mut().unwrap())
        }
        // DataType::BinaryView => {
        //     Ok(Arc::new(take_byte_view(values.as_binary_view(), indices)?))
        // }
        // DataType::FixedSizeBinary(size) => {
        //     let values = values
        //         .as_any()
        //         .downcast_ref::<FixedSizeBinaryArray>()
        //         .unwrap();
        //     Ok(Arc::new(take_fixed_size_binary(values, indices, *size)?))
        // }
        // DataType::Null => {
        //     // Take applied to a null array produces a null array.
        //     if values.len() >= indices.len() {
        //         // If the existing null array is as big as the indices, we can use a slice of it
        //         // to avoid allocating a new null array.
        //         Ok(values.slice(0, indices.len()))
        //     } else {
        //         // If the existing null array isn't big enough, create a new one.
        //         Ok(new_null_array(&DataType::Null, indices.len()))
        //     }
        // }
        // DataType::Union(fields, UnionMode::Sparse) => {
        //     let mut children = Vec::with_capacity(fields.len());
        //     let values = values.as_any().downcast_ref::<UnionArray>().unwrap();
        //     let type_ids = take_native(values.type_ids(), indices);
        //     for (type_id, _field) in fields.iter() {
        //         let values = values.child(type_id);
        //         let values = take_impl(values, indices)?;
        //         children.push(values);
        //     }
        //     let array = UnionArray::try_new(fields.clone(), type_ids, None, children)?;
        //     Ok(Arc::new(array))
        // }
        // DataType::Union(fields, UnionMode::Dense) => {
        //     let values = values.as_any().downcast_ref::<UnionArray>().unwrap();
        //
        //     let type_ids = <PrimitiveArray<Int8Type>>::try_new(take_native(values.type_ids(), indices), None)?;
        //     let offsets = <PrimitiveArray<Int32Type>>::try_new(take_native(values.offsets().unwrap(), indices), None)?;
        //
        //     let children = fields.iter()
        //         .map(|(field_type_id, _)| {
        //             let mask = BooleanArray::from_unary(&type_ids, |value_type_id| value_type_id == field_type_id);
        //
        //             let indices = arrow_select::filter::filter(&offsets, &mask)?;
        //
        //             let values = values.child(field_type_id);
        //
        //             take_impl(values, indices.as_primitive::<Int32Type>())
        //         })
        //         .collect::<Result<_, _>>()?;
        //
        //     let mut child_offsets = [0; 128];
        //
        //     let offsets = type_ids.values()
        //         .iter()
        //         .map(|&i| {
        //             let offset = child_offsets[i as usize];
        //
        //             child_offsets[i as usize] += 1;
        //
        //             offset
        //         })
        //         .collect();
        //
        //     let (_, type_ids, _) = type_ids.into_parts();
        //
        //     let array = UnionArray::try_new(fields.clone(), type_ids, Some(offsets), children)?;
        //
        //     Ok(Arc::new(array))
        // }
        t => unimplemented!("Take not supported for data type {:?}", t)
    }
}

/// Options that define how `take` should behave
#[derive(Clone, Debug, Default)]
pub struct TakeOptions {
    /// Perform bounds check before taking indices from values.
    /// If enabled, an `ArrowError` is returned if the indices are out of bounds.
    /// If not enabled, and indices exceed bounds, the kernel will panic.
    pub check_bounds: bool,
}

/// `take` implementation for all primitive arrays
///
/// This checks if an `indices` slot is populated, and gets the value from `values`
///  as the populated index.
/// If the `indices` slot is null, a null value is returned.
/// For example, given:
///     values:  [1, 2, 3, null, 5]
///     indices: [0, null, 4, 3]
/// The result is: [1 (slot 0), null (null slot), 5 (slot 4), null (slot 3)]
fn take_primitive<T, I>(
    values: &PrimitiveArray<T>,
    indices: &PrimitiveArray<I>,
    output: &mut PrimitiveSink<T>,
) where
    T: ArrowPrimitiveType,
    I: ArrowPrimitiveType,
{
    let values_buf = take_native(values.values(), indices, output);
    let nulls = take_nulls(values.nulls(), indices, output.nulls.as_mut());
    // Ok(PrimitiveArray::try_new(values_buf, nulls)?.with_data_type(values.data_type().clone()))
}

#[inline(never)]
fn take_nulls<I: ArrowPrimitiveType>(
    values: Option<&NullBuffer>,
    indices: &PrimitiveArray<I>,
    output: Option<&mut NullSink>,
) {
    let Some(output) = output else {
        assert_eq!(indices.null_count(), 0);
        assert!(values.is_none_or(|n| n.null_count() == 0));

        return;
    };

    match values.filter(|n| n.null_count() > 0) {
        Some(n) => {
            unsafe {
                output
                    .inner
                    .extend_trusted_len(get_bits(n.inner(), indices))
            }
            // take_bits(n.inner(), indices, &mut output.inner);
            // Some(NullBuffer::new(buffer)).filter(|n| n.null_count() > 0)
        }
        None => {
            assert_eq!(indices.null_count(), 0);
            output.inner.append_n_non_nulls(indices.len());
            // indices.nulls().cloned()
        }
    }
}

#[inline(never)]
fn take_native<T: ArrowPrimitiveType, I: ArrowPrimitiveType>(
    values: &[T::Native],
    indices: &PrimitiveArray<I>,
    output: &mut PrimitiveSink<T>,
) {
    match indices.nulls().filter(|n| n.null_count() > 0) {
        Some(n) => {
            output.values.extend(
                indices.values().iter().enumerate().map(|(idx, index)| {
                    match values.get(index.as_usize()) {
                        Some(v) => *v,
                        // SAFETY: idx<indices.len()
                        None => match unsafe { n.inner().value_unchecked(idx) } {
                            false => T::Native::default(),
                            true => panic!("Out-of-bounds index {index:?}"),
                        },
                    }
                }), // .collect()
            )
        }
        None => {
            output.values.extend(
                indices
                    .values()
                    .iter()
                    .map(|index| values[index.as_usize()]), // .collect()
            )
        }
    }
}

#[inline(never)]
fn take_bits<I: ArrowPrimitiveType>(
    values: &BooleanBuffer,
    indices: &PrimitiveArray<I>,
    output: &mut BooleanBufferSink,
) {
    unsafe { output.buffer.extend_trusted_len(get_bits(values, indices)) }
}

fn get_bits<I: ArrowPrimitiveType>(
    values: &BooleanBuffer,
    indices: &PrimitiveArray<I>,
) -> impl ExactSizeIterator<Item = bool> {
    let len = indices.len();

    match indices.nulls().filter(|n| n.null_count() > 0) {
        Some(nulls) => {
            // let mut output_buffer = MutableBuffer::new_null(len);
            // let output_slice = output_buffer.as_slice_mut();
            // nulls.valid_indices().for_each(|idx| {
            //     // SAFETY: idx is a valid index in indices.nulls() --> idx<indices.len()
            //     if values.value(unsafe { indices.value_unchecked(idx).as_usize() }) {
            //         // SAFETY: MutableBuffer was created with space for indices.len() bit, and idx < indices.len()
            //         unsafe { bit_util::set_bit_raw(output_slice.as_mut_ptr(), idx) };
            //     }
            // });
            // BooleanBuffer::new(output_buffer.into(), 0, len)
            unimplemented!("get_bits with nulls")
        }
        None => {
            (0..len).map(|idx| {
                // SAFETY: idx<indices.len()
                values.value(unsafe { indices.value_unchecked(idx).as_usize() })
            })
        }
    }
}

/// `take` implementation for boolean arrays
fn take_boolean<IndexType: ArrowPrimitiveType>(
    values: &BooleanArray,
    indices: &PrimitiveArray<IndexType>,
    output: &mut BooleanSink,
) {
    take_bits(values.values(), indices, &mut output.inner);
    take_nulls(values.nulls(), indices, output.nulls.as_mut());
    // BooleanArray::new(val_buf, null_buf)
}

/// `take` implementation for string arrays
fn take_bytes<T: ByteArrayType, IndexType: ArrowPrimitiveType>(
    array: &GenericByteArray<T>,
    indices: &PrimitiveArray<IndexType>,
    output: &mut GenericByteSink<T>,
) {
    // let mut offsets = Vec::with_capacity(indices.len() + 1);
    // offsets.push(T::Offset::default());

    let input_offsets = array.value_offsets();
    let initial_capacity = *output.offsets.last().unwrap();
    let mut capacity = initial_capacity;
    take_nulls(array.nulls(), indices, output.nulls.as_mut());
    assert_eq!(indices.null_count(), 0);

    if array.null_count() == 0 {
        output.offsets.reserve(indices.len());
        output.offsets.extend(
            indices.values()
              .iter()
              .map(|index| {
                  let index = index.as_usize();
                  capacity += input_offsets[index + 1] - input_offsets[index];

                  capacity
              })
        );

        // let mut values = Vec::with_capacity(capacity);
        output.bytes.reserve(capacity.as_usize() - initial_capacity.as_usize());

        for index in indices.values() {
            output
                .bytes
                .extend_from_slice(array.value(index.as_usize()).as_ref());
        }
        // (offsets, values)
    } else {
        // offsets.reserve(indices.len());
        output.offsets.reserve(indices.len());
        output.offsets.extend(
            indices.values()
              .iter()
              .map(|index| {
                  let index = index.as_usize();
                  if array.is_valid(index) {
                      capacity += input_offsets[index + 1] - input_offsets[index];
                  }
                  capacity
              })
        );

        output.bytes.reserve(capacity.as_usize() - initial_capacity.as_usize());
        for index in indices.values() {
            let index = index.as_usize();
            if array.is_valid(index) {
                output.bytes.extend_from_slice(array.value(index).as_ref());
            }
        }
        // (offsets, values)
    };

    // T::Offset::from_usize(values.len())
    //     .ok_or_else(|| ArrowError::OffsetOverflowError(values.len()))?;
    //
    // let array = unsafe {
    //     let offsets = OffsetBuffer::new_unchecked(offsets.into());
    //     GenericByteArray::<T>::new_unchecked(offsets, values.into(), nulls)
    // };
    //
    // Ok(array)
}

// TODO - uncomment when used

// /// `take` implementation for byte view arrays
// fn take_byte_view<T: ByteViewType, IndexType: ArrowPrimitiveType>(
//     array: &GenericByteViewArray<T>,
//     indices: &PrimitiveArray<IndexType>,
// ) -> Result<GenericByteViewArray<T>, ArrowError> {
//     let new_views = take_native(array.views(), indices);
//     let new_nulls = take_nulls(array.nulls(), indices);
//     // Safety:  array.views was valid, and take_native copies only valid values, and verifies bounds
//     Ok(unsafe {
//         GenericByteViewArray::new_unchecked(new_views, array.data_buffers().to_vec(), new_nulls)
//     })
// }
//
// /// `take` implementation for list arrays
// ///
// /// Calculates the index and indexed offset for the inner array,
// /// applying `take` on the inner array, then reconstructing a list array
// /// with the indexed offsets
// fn take_list<IndexType, OffsetType>(
//     values: &GenericListArray<OffsetType::Native>,
//     indices: &PrimitiveArray<IndexType>,
// ) -> Result<GenericListArray<OffsetType::Native>, ArrowError>
// where
//     IndexType: ArrowPrimitiveType,
//     OffsetType: ArrowPrimitiveType,
//     OffsetType::Native: OffsetSizeTrait,
//     PrimitiveArray<OffsetType>: From<Vec<OffsetType::Native>>,
// {
//     // TODO: Some optimizations can be done here such as if it is
//     // taking the whole list or a contiguous sublist
//     let (list_indices, offsets, null_buf) =
//         take_value_indices_from_list::<IndexType, OffsetType>(values, indices)?;
//
//     let taken = take_impl::<OffsetType>(values.values().as_ref(), &list_indices)?;
//     let value_offsets = Buffer::from_vec(offsets);
//     // create a new list with taken data and computed null information
//     let list_data = ArrayDataBuilder::new(values.data_type().clone())
//         .len(indices.len())
//         .null_bit_buffer(Some(null_buf.into()))
//         .offset(0)
//         .add_child_data(taken.into_data())
//         .add_buffer(value_offsets);
//
//     let list_data = unsafe { list_data.build_unchecked() };
//
//     Ok(GenericListArray::<OffsetType::Native>::from(list_data))
// }
//
// fn take_list_view<IndexType, OffsetType>(
//     values: &GenericListViewArray<OffsetType::Native>,
//     indices: &PrimitiveArray<IndexType>,
// ) -> Result<GenericListViewArray<OffsetType::Native>, ArrowError>
// where
//     IndexType: ArrowPrimitiveType,
//     OffsetType: ArrowPrimitiveType,
//     OffsetType::Native: OffsetSizeTrait,
// {
//     let taken_offsets = take_native(values.offsets(), indices);
//     let taken_sizes = take_native(values.sizes(), indices);
//     let nulls = take_nulls(values.nulls(), indices);
//
//     let list_view_data = ArrayDataBuilder::new(values.data_type().clone())
//         .len(indices.len())
//         .nulls(nulls)
//         .buffers(vec![taken_offsets.into(), taken_sizes.into()])
//         .child_data(vec![values.values().to_data()]);
//
//     // SAFETY: all buffers and child nodes for ListView added in constructor
//     let list_view_data = unsafe { list_view_data.build_unchecked() };
//
//     Ok(GenericListViewArray::<OffsetType::Native>::from(
//         list_view_data,
//     ))
// }
//
// /// `take` implementation for `FixedSizeListArray`
// ///
// /// Calculates the index and indexed offset for the inner array,
// /// applying `take` on the inner array, then reconstructing a list array
// /// with the indexed offsets
// fn take_fixed_size_list<IndexType: ArrowPrimitiveType>(
//     values: &FixedSizeListArray,
//     indices: &PrimitiveArray<IndexType>,
//     length: <UInt32Type as ArrowPrimitiveType>::Native,
// ) -> Result<FixedSizeListArray, ArrowError> {
//     let list_indices = take_value_indices_from_fixed_size_list(values, indices, length)?;
//     let taken = take_impl::<UInt32Type>(values.values().as_ref(), &list_indices)?;
//
//     // determine null count and null buffer, which are a function of `values` and `indices`
//     let num_bytes = bit_util::ceil(indices.len(), 8);
//     let mut null_buf = MutableBuffer::new(num_bytes).with_bitset(num_bytes, true);
//     let null_slice = null_buf.as_slice_mut();
//
//     for i in 0..indices.len() {
//         let index = indices
//             .value(i)
//             .to_usize()
//             .ok_or_else(|| ArrowError::ComputeError("Cast to usize failed".to_string()))?;
//         if !indices.is_valid(i) || values.is_null(index) {
//             bit_util::unset_bit(null_slice, i);
//         }
//     }
//
//     let list_data = ArrayDataBuilder::new(values.data_type().clone())
//         .len(indices.len())
//         .null_bit_buffer(Some(null_buf.into()))
//         .offset(0)
//         .add_child_data(taken.into_data());
//
//     let list_data = unsafe { list_data.build_unchecked() };
//
//     Ok(FixedSizeListArray::from(list_data))
// }
//
// /// The take kernel implementation for `FixedSizeBinaryArray`.
// ///
// /// The computation is done in two steps:
// /// - Compute the values buffer
// /// - Compute the null buffer
// fn take_fixed_size_binary<IndexType: ArrowPrimitiveType>(
//     values: &FixedSizeBinaryArray,
//     indices: &PrimitiveArray<IndexType>,
//     size: i32,
// ) -> Result<FixedSizeBinaryArray, ArrowError> {
//     let size_usize = usize::try_from(size).map_err(|_| {
//         ArrowError::InvalidArgumentError(format!("Cannot convert size '{}' to usize", size))
//     })?;
//
//     let values_buffer = values.values().as_slice();
//     let mut values_buffer_builder = BufferBuilder::new(indices.len() * size_usize);
//
//     if indices.null_count() == 0 {
//         let array_iter = indices.values().iter().map(|idx| {
//             let offset = idx.as_usize() * size_usize;
//             &values_buffer[offset..offset + size_usize]
//         });
//         for slice in array_iter {
//             values_buffer_builder.append_slice(slice);
//         }
//     } else {
//         // The indices nullability cannot be ignored here because the values buffer may contain
//         // nulls which should not cause a panic.
//         let array_iter = indices.iter().map(|idx| {
//             idx.map(|idx| {
//                 let offset = idx.as_usize() * size_usize;
//                 &values_buffer[offset..offset + size_usize]
//             })
//         });
//         for slice in array_iter {
//             match slice {
//                 None => values_buffer_builder.append_n(size_usize, 0),
//                 Some(slice) => values_buffer_builder.append_slice(slice),
//             }
//         }
//     }
//
//     let values_buffer = values_buffer_builder.finish();
//     let value_nulls = take_nulls(values.nulls(), indices);
//     let final_nulls = NullBuffer::union(value_nulls.as_ref(), indices.nulls());
//
//     let array_data = ArrayDataBuilder::new(DataType::FixedSizeBinary(size))
//         .len(indices.len())
//         .nulls(final_nulls)
//         .offset(0)
//         .add_buffer(values_buffer)
//         .build()?;
//
//     Ok(FixedSizeBinaryArray::from(array_data))
// }
//
// /// `take` implementation for dictionary arrays
// ///
// /// applies `take` to the keys of the dictionary array and returns a new dictionary array
// /// with the same dictionary values and reordered keys
// fn take_dict<T: ArrowDictionaryKeyType, I: ArrowPrimitiveType>(
//     values: &DictionaryArray<T>,
//     indices: &PrimitiveArray<I>,
// ) -> Result<DictionaryArray<T>, ArrowError> {
//     let new_keys = take_primitive(values.keys(), indices)?;
//     Ok(unsafe { DictionaryArray::new_unchecked(new_keys, values.values().clone()) })
// }
//
// /// `take` implementation for run arrays
// ///
// /// Finds physical indices for the given logical indices and builds output run array
// /// by taking values in the input run_array.values at the physical indices.
// /// The output run array will be run encoded on the physical indices and not on output values.
// /// For e.g. an input `RunArray{ run_ends = [2,4,6,8], values=[1,2,1,2] }` and `logical_indices=[2,3,6,7]`
// /// would be converted to `physical_indices=[1,1,3,3]` which will be used to build
// /// output `RunArray{ run_ends=[2,4], values=[2,2] }`.
// fn take_run<T: RunEndIndexType, I: ArrowPrimitiveType>(
//     run_array: &RunArray<T>,
//     logical_indices: &PrimitiveArray<I>,
// ) -> Result<RunArray<T>, ArrowError> {
//     // get physical indices for the input logical indices
//     let physical_indices = run_array.get_physical_indices(logical_indices.values())?;
//
//     // Run encode the physical indices into new_run_ends_builder
//     // Keep track of the physical indices to take in take_value_indices
//     // `unwrap` is used in this function because the unwrapped values are bounded by the corresponding `::Native`.
//     let mut new_run_ends_builder = BufferBuilder::<T::Native>::new(1);
//     let mut take_value_indices = BufferBuilder::<I::Native>::new(1);
//     let mut new_physical_len = 1;
//     for ix in 1..physical_indices.len() {
//         if physical_indices[ix] != physical_indices[ix - 1] {
//             take_value_indices.append(I::Native::from_usize(physical_indices[ix - 1]).unwrap());
//             new_run_ends_builder.append(T::Native::from_usize(ix).unwrap());
//             new_physical_len += 1;
//         }
//     }
//     take_value_indices
//         .append(I::Native::from_usize(physical_indices[physical_indices.len() - 1]).unwrap());
//     new_run_ends_builder.append(T::Native::from_usize(physical_indices.len()).unwrap());
//     let new_run_ends = unsafe {
//         // Safety:
//         // The function builds a valid run_ends array and hence need not be validated.
//         ArrayDataBuilder::new(T::DATA_TYPE)
//             .len(new_physical_len)
//             .null_count(0)
//             .add_buffer(new_run_ends_builder.finish())
//             .build_unchecked()
//     };
//
//     let take_value_indices: PrimitiveArray<I> = unsafe {
//         // Safety:
//         // The function builds a valid take_value_indices array and hence need not be validated.
//         ArrayDataBuilder::new(I::DATA_TYPE)
//             .len(new_physical_len)
//             .null_count(0)
//             .add_buffer(take_value_indices.finish())
//             .build_unchecked()
//             .into()
//     };
//
//     let new_values = arrow_select::take::take(run_array.values(), &take_value_indices, None)?;
//
//     let builder = ArrayDataBuilder::new(run_array.data_type().clone())
//         .len(physical_indices.len())
//         .add_child_data(new_run_ends)
//         .add_child_data(new_values.into_data());
//     let array_data = unsafe {
//         // Safety:
//         //  This function builds a valid run array and hence can skip validation.
//         builder.build_unchecked()
//     };
//     Ok(array_data.into())
// }
//
// /// Takes/filters a list array's inner data using the offsets of the list array.
// ///
// /// Where a list array has indices `[0,2,5,10]`, taking indices of `[2,0]` returns
// /// an array of the indices `[5..10, 0..2]` and offsets `[0,5,7]` (5 elements and 2
// /// elements)
// #[allow(clippy::type_complexity)]
// fn take_value_indices_from_list<IndexType, OffsetType>(
//     list: &GenericListArray<OffsetType::Native>,
//     indices: &PrimitiveArray<IndexType>,
// ) -> Result<
//     (
//         PrimitiveArray<OffsetType>,
//         Vec<OffsetType::Native>,
//         MutableBuffer,
//     ),
//     ArrowError,
// >
// where
//     IndexType: ArrowPrimitiveType,
//     OffsetType: ArrowPrimitiveType,
//     OffsetType::Native: OffsetSizeTrait + std::ops::Add + Zero + One,
//     PrimitiveArray<OffsetType>: From<Vec<OffsetType::Native>>,
// {
//     // TODO: benchmark this function, there might be a faster unsafe alternative
//     let offsets: &[OffsetType::Native] = list.value_offsets();
//
//     let mut new_offsets = Vec::with_capacity(indices.len());
//     let mut values = Vec::new();
//     let mut current_offset = OffsetType::Native::zero();
//     // add first offset
//     new_offsets.push(OffsetType::Native::zero());
//
//     // Initialize null buffer
//     let num_bytes = bit_util::ceil(indices.len(), 8);
//     let mut null_buf = MutableBuffer::new(num_bytes).with_bitset(num_bytes, true);
//     let null_slice = null_buf.as_slice_mut();
//
//     // compute the value indices, and set offsets accordingly
//     for i in 0..indices.len() {
//         if indices.is_valid(i) {
//             let ix = indices
//                 .value(i)
//                 .to_usize()
//                 .ok_or_else(|| ArrowError::ComputeError("Cast to usize failed".to_string()))?;
//             let start = offsets[ix];
//             let end = offsets[ix + 1];
//             current_offset += end - start;
//             new_offsets.push(current_offset);
//
//             let mut curr = start;
//
//             // if start == end, this slot is empty
//             while curr < end {
//                 values.push(curr);
//                 curr += One::one();
//             }
//             if !list.is_valid(ix) {
//                 bit_util::unset_bit(null_slice, i);
//             }
//         } else {
//             bit_util::unset_bit(null_slice, i);
//             new_offsets.push(current_offset);
//         }
//     }
//
//     Ok((
//         PrimitiveArray::<OffsetType>::from(values),
//         new_offsets,
//         null_buf,
//     ))
// }
//
// /// Takes/filters a fixed size list array's inner data using the offsets of the list array.
// fn take_value_indices_from_fixed_size_list<IndexType>(
//     list: &FixedSizeListArray,
//     indices: &PrimitiveArray<IndexType>,
//     length: <UInt32Type as ArrowPrimitiveType>::Native,
// ) -> Result<PrimitiveArray<UInt32Type>, ArrowError>
// where
//     IndexType: ArrowPrimitiveType,
// {
//     let mut values = UInt32Builder::with_capacity(length as usize * indices.len());
//
//     for i in 0..indices.len() {
//         if indices.is_valid(i) {
//             let index = indices
//                 .value(i)
//                 .to_usize()
//                 .ok_or_else(|| ArrowError::ComputeError("Cast to usize failed".to_string()))?;
//             let start = list.value_offset(index) as <UInt32Type as ArrowPrimitiveType>::Native;
//
//             // Safety: Range always has known length.
//             unsafe {
//                 values.append_trusted_len_iter(start..start + length);
//             }
//         } else {
//             values.append_nulls(length as usize);
//         }
//     }
//
//     Ok(values.finish())
// }

/// To avoid generating take implementations for every index type, instead we
/// only generate for UInt32 and UInt64 and coerce inputs to these types
trait ToIndices {
    type T: ArrowPrimitiveType;

    fn to_indices(&self) -> PrimitiveArray<Self::T>;
}

macro_rules! to_indices_reinterpret {
    ($t:ty, $o:ty) => {
        impl ToIndices for PrimitiveArray<$t> {
            type T = $o;

            fn to_indices(&self) -> PrimitiveArray<$o> {
                let cast = ScalarBuffer::new(self.values().inner().clone(), 0, self.len());
                PrimitiveArray::new(cast, self.nulls().cloned())
            }
        }
    };
}

macro_rules! to_indices_identity {
    ($t:ty) => {
        impl ToIndices for PrimitiveArray<$t> {
            type T = $t;

            fn to_indices(&self) -> PrimitiveArray<$t> {
                self.clone()
            }
        }
    };
}

macro_rules! to_indices_widening {
    ($t:ty, $o:ty) => {
        impl ToIndices for PrimitiveArray<$t> {
            type T = UInt32Type;

            fn to_indices(&self) -> PrimitiveArray<$o> {
                let cast = self.values().iter().copied().map(|x| x as _).collect();
                PrimitiveArray::new(cast, self.nulls().cloned())
            }
        }
    };
}

to_indices_widening!(UInt8Type, UInt32Type);
to_indices_widening!(Int8Type, UInt32Type);

to_indices_widening!(UInt16Type, UInt32Type);
to_indices_widening!(Int16Type, UInt32Type);

to_indices_identity!(UInt32Type);
to_indices_reinterpret!(Int32Type, UInt32Type);

to_indices_identity!(UInt64Type);
to_indices_reinterpret!(Int64Type, UInt64Type);

/// Take rows by index from [`RecordBatch`] and returns a new [`RecordBatch`] from those indexes.
///
/// This function will call [`arrow_select::take::take`] on each array of the [`RecordBatch`] and assemble a new [`RecordBatch`].
///
/// # Example
/// ```
/// # use std::sync::Arc;
/// # use arrow_array::{StringArray, Int32Array, UInt32Array, RecordBatch};
/// # use arrow_schema::{DataType, Field, Schema};
/// # use arrow_select::take::take_record_batch;
/// let schema = Arc::new(Schema::new(vec![
///     Field::new("a", DataType::Int32, true),
///     Field::new("b", DataType::Utf8, true),
/// ]));
/// let batch = RecordBatch::try_new(
///     schema.clone(),
///     vec![
///         Arc::new(Int32Array::from_iter_values(0..20)),
///         Arc::new(StringArray::from_iter_values(
///             (0..20).map(|i| format!("str-{}", i)),
///         )),
///     ],
/// )
/// .unwrap();
///
/// let indices = UInt32Array::from(vec![1, 5, 10]);
/// let taken = take_record_batch(&batch, &indices).unwrap();
///
/// let expected = RecordBatch::try_new(
///     schema,
///     vec![
///         Arc::new(Int32Array::from(vec![1, 5, 10])),
///         Arc::new(StringArray::from(vec!["str-1", "str-5", "str-10"])),
///     ],
/// )
/// .unwrap();
/// assert_eq!(taken, expected);
/// ```
pub fn take_record_batch(
    record_batch: &RecordBatch,
    indices: &dyn Array,
) -> Result<RecordBatch, ArrowError> {
    let columns = record_batch
        .columns()
        .iter()
        .map(|c| arrow_select::take::take(c, indices, None))
        .collect::<Result<Vec<_>, _>>()?;
    RecordBatch::try_new(record_batch.schema(), columns)
}

