use arrow::array::{ArrayRef, RecordBatch, RecordBatchOptions};
use arrow::datatypes::SchemaRef;
use arrow_schema::ArrowError;
use std::sync::Arc;

pub struct ArrowBatchesSink {
    ready_batches: Vec<RecordBatch>,
    schema: SchemaRef,

    columns: Vec<ArrayRef>,
    number_of_rows: Option<usize>,
}

impl ArrowBatchesSink {
    pub(crate) fn new(schema: &SchemaRef) -> Self {
        Self {
            ready_batches: Vec::new(),
            schema: Arc::clone(schema),
            columns: Vec::with_capacity(schema.fields().len()),
            number_of_rows: None,
        }
    }

    pub(crate) fn start_batch(&mut self, number_of_rows: usize) -> Result<(), ArrowError> {
        self.write_number_of_rows(number_of_rows)
    }

    pub(crate) fn write_number_of_rows(&mut self, number_of_rows: usize) -> Result<(), ArrowError> {
        if let Some(number_of_rows) = self.number_of_rows {
            return Err(ArrowError::ComputeError(format!(
                "Number of rows already set to {number_of_rows}, cannot set it again"
            )));
        }

        self.number_of_rows = Some(number_of_rows);
        Ok(())
    }

    pub(crate) fn write_column(&mut self, column_index: usize, column: &ArrayRef) -> Result<(), ArrowError> {
        let Some(number_of_rows) = self.number_of_rows else {
            return Err(ArrowError::ComputeError("Number of rows must be written before writing columns".to_string()));
        };

        if self.columns.len() != column_index {
            return Err(ArrowError::ComputeError(format!(
                "Must write columns in order, expected column index {}, got {}",
                self.columns.len(),
                column_index
            )));
        }

        self.columns.push(Arc::clone(column));

        // Doing > than here as the RecordBatch will do the validation for the schema and column number
        if self.columns.len() >= self.schema.fields().len() {
            self.number_of_rows = None;
            let columns = std::mem::replace(
                &mut self.columns,
                Vec::with_capacity(self.schema.fields().len()),
            );
            // All columns are written, we can write the batch
            let batch = RecordBatch::try_new_with_options(
                Arc::clone(&self.schema),
                columns,
                &RecordBatchOptions::default().with_row_count(Some(number_of_rows)),
            )?;
            self.ready_batches.push(batch);
        }

        Ok(())
    }

    pub(crate) fn finish(self) -> Vec<RecordBatch> {
        self.ready_batches
    }
}

