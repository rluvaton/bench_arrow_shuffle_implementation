use arrow::array::{Array, ArrayRef};
use arrow::datatypes::SchemaRef;
use std::collections::VecDeque;
use arrow_array::RecordBatch;
use arrow_schema::ArrowError;
use crate::splitter_approach::ready_partitions_sink::arrow_batches_sink::ArrowBatchesSink;
use crate::splitter_approach::ready_partitions_sink::traits::ReadyPartitionsSink;

#[derive(Debug, Clone)]
struct PendingPartition {
    /// columns[column_index][batch_index] contains all the arrays for that column ready to be created.
    ///
    /// Note: the last column is not added here, it is added directly to the batch in `finish_with_last_column`.
    columns: Vec<VecDeque<ArrayRef>>,

    /// The next column index that should be written to the shuffle.
    ///
    /// The flow this is calling `add_column` multiple times one after another for each column in the schema,
    /// ```text
    /// For schema: <a: i32, b: i64, c: string>
    ///
    /// add
    /// For each partition
    ///
    /// ```
    ///
    /// schema: <a: i32, b: i64, c: string>
    ///
    /// call flow:
    /// ```plain
    /// add_column(0, array_a) // Should write to shuffle and increase this value to 1
    /// add_column(0, array_a) // should not write to shuffle as now it's column 1 turn
    /// add_column(0, array_a) // should not write to shuffle as now it's column 1 turn
    ///
    /// add_column(1, array_b) // Should write to shuffle and increase this value to 2
    /// add_column(1, array_b) // should not write to shuffle as now it's column 2 turn
    /// add_column(1, array_b) // should not write to shuffle as now it's column 2 turn
    ///
    /// add_column(2, array_c) // should write to shuffle and reset the value to 0, and then write column a 2nd array and
    /// add_column(2, array_c) // should write second column 0 and 1 and then write this column
    /// add_column(2, array_c) // should write third column 0 and 1 and then write this column
    /// ```
    next_column_index_to_write_to_shuffle: usize,

    /// How many columns are in the schema
    number_of_columns: usize,
}

impl PendingPartition {
    fn new(number_of_columns: usize) -> Self {
        assert_ne!(
            number_of_columns, 0,
            "Number of columns must be greater than 0"
        );
        let columns = vec![VecDeque::new(); number_of_columns - 1];

        Self {
            number_of_columns,
            columns,
            next_column_index_to_write_to_shuffle: 0,
        }
    }

    fn should_write_column(&self, column_index: usize) -> bool {
        self.next_column_index_to_write_to_shuffle == column_index
    }

    fn advance_for_next_column(&mut self) {
        self.next_column_index_to_write_to_shuffle =
            (self.next_column_index_to_write_to_shuffle + 1) % self.number_of_columns;
    }

    fn add_column(
        &mut self,
        column_index: usize,
        array: ArrayRef,
        shuffle_writer: &mut ArrowBatchesSink
    ) -> Result<(), ArrowError>
    {
        assert_ne!(
            column_index,
            self.number_of_columns - 1,
            "Last column should not be added here, it should be added in `finish_with_last_column`"
        );

        // Columns must be added in the order of the schema in our logic.
        // so either we are writing the column that is next in line to be written, or the one that was just written and we moved to the next one but still have some arrays for it.
        assert!(
            column_index == self.next_column_index_to_write_to_shuffle ||
                column_index + 1 == self.next_column_index_to_write_to_shuffle,
            "Column index {column_index} should be either the current next index {} or the prev one, but it is not.",
            self.next_column_index_to_write_to_shuffle,
        );

        // If first column than start a new batch
        if column_index == 0 && self.next_column_index_to_write_to_shuffle == 0 {
            shuffle_writer.start_batch(array.len())?;
        }

        // If we should not write this column, save it for later to be written
        if !self.should_write_column(column_index) {
            self.columns[column_index].push_back(array);
            return Ok(());
        }

        // If we just finished the second column or any column after that, make sure that the column that we just finished have the same number of buffered arrays as the previous one.
        // This is to assert the assumption that we are writing all the arrays of a column at a time before going to the next one
        if column_index >= 2 {
            let just_finished_column_index = column_index - 1;
            let previous_column_index = column_index - 2;
            let just_finished_column_length = self.columns[just_finished_column_index].len();
            let expected_length = self.columns[previous_column_index].len();
            assert_eq!(
                just_finished_column_length,
                expected_length,
                "Column {just_finished_column_index} have {just_finished_column_length} buffered arrays, \
                while the one before that ({previous_column_index}) had {expected_length} buffered arrays. \
                Not all the arrays for a column were added before going to the next one",
            );
        }

        shuffle_writer.write_column(column_index, &array)?;
        self.advance_for_next_column();

        Ok(())
    }

    /// Finish the current partition with the last column for the schema
    fn finish_with_last_column(
        &mut self,
        last_column_array: ArrayRef,
        shuffle_writer: &mut ArrowBatchesSink,
    ) -> Result<(), ArrowError> {
        if self.number_of_columns == 1 {
            shuffle_writer.start_batch(last_column_array.len())?;
        }

        let last_column_index = self.number_of_columns - 1;

        assert!(
            self.should_write_column(last_column_index),
            "Last column should be written immediately, but it is not. Current next index: {}, number of columns: {}",
            self.next_column_index_to_write_to_shuffle,
            self.number_of_columns
        );

        shuffle_writer.write_column(last_column_index, &last_column_array)?;

        self.advance_for_next_column();

        // Should reset to the start of the columns for next arrays
        assert_eq!(
            self.next_column_index_to_write_to_shuffle, 0,
            "Should reset the next column"
        );

        if self.number_of_columns == 1 {
            assert_eq!(
                self.next_column_index_to_write_to_shuffle,
                last_column_index
            );
            return Ok(());
        }

        // If there are no array buffered
        if self.columns[0].is_empty() {
            return Ok(());
        }

        // Start a new batch
        shuffle_writer.start_batch(self.columns[0][0].len())?;

        // Write the buffered columns until the last column
        while self.next_column_index_to_write_to_shuffle < last_column_index {
            let col = self.columns[self.next_column_index_to_write_to_shuffle]
                .pop_front()
                // If we reached here then we must have buffered array for all columns except the last one
                // as the first column have buffered array
                .expect("Must have column buffered, columns not added in the order of the schema");

            shuffle_writer.write_column(self.next_column_index_to_write_to_shuffle, &col)?;

            self.advance_for_next_column();
        }

        Ok(())
    }
}

/// Add ready columns that is already split into partitions and build batches
pub(crate) struct ShuffleEncodedSink
{
    pending_columns_per_partitions: Vec<PendingPartition>,

    shuffle_writer_per_partition: Vec<ArrowBatchesSink>,

    schema: SchemaRef,
}

impl ShuffleEncodedSink {
    pub(crate) fn new(
        number_of_partitions: usize,
        schema: SchemaRef,
    ) -> Self {
        Self {
            pending_columns_per_partitions: vec![
                PendingPartition::new(schema.fields.len());
                number_of_partitions
            ],
            shuffle_writer_per_partition: (0..number_of_partitions)
                .map(|_| ArrowBatchesSink::new(&schema))
                .collect(),
            schema,
        }
    }

    pub(crate) fn into_inner(self) -> Vec<Vec<RecordBatch>> {
        self.shuffle_writer_per_partition
            .into_iter()
            .map(|writer| writer.finish())
            .collect()
    }
}

impl ReadyPartitionsSink for ShuffleEncodedSink {
    fn add_column_to_partition(
        &mut self,
        column_index: usize,
        partition_index: usize,
        array: ArrayRef,
    ) -> Result<(), ArrowError> {
        assert_ne!(array.len(), 0, "Array at column {column_index} in partition {partition_index} is empty while it should not be");
        let current_partition = &mut self.pending_columns_per_partitions[partition_index];
        let shuffle_writer = &mut self.shuffle_writer_per_partition[partition_index];

        // If this is the last column we can create a batch
        if column_index == self.schema.fields.len() - 1 {
            current_partition.finish_with_last_column(array, shuffle_writer)?;
        } else {
            current_partition.add_column(column_index, array, shuffle_writer)?;
        }

        Ok(())
    }
}
