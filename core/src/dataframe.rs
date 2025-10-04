use std::path::Path;

use polars::prelude::*;

use crate::errors::{MatheMixxError, Result};

#[derive(Debug, Clone)]
pub struct DataSet {
    frame: DataFrame,
}

impl DataSet {
    pub fn from_csv(path: impl AsRef<Path>) -> Result<Self> {
        let frame = CsvReadOptions::default()
            .with_infer_schema_length(Some(1024))
            .try_into_reader_with_file_path(Some(path.as_ref().into()))?
            .finish()?;
        Ok(Self { frame })
    }

    pub fn from_frame(frame: DataFrame) -> Self {
        Self { frame }
    }

    pub fn dataframe(&self) -> &DataFrame {
        &self.frame
    }

    pub fn clone_dataframe(&self) -> DataFrame {
        self.frame.clone()
    }

    pub fn into_inner(self) -> DataFrame {
        self.frame
    }

    pub fn height(&self) -> usize {
        self.frame.height()
    }

    pub fn width(&self) -> usize {
        self.frame.width()
    }

    pub fn column_names(&self) -> Vec<String> {
        self.frame
            .get_column_names_owned()
            .into_iter()
            .map(|s| s.to_string())
            .collect()
    }

    pub fn numeric_column_names(&self) -> Vec<String> {
        self.frame
            .get_columns()
            .iter()
            .filter_map(|s| match s.dtype() {
                DataType::Int8
                | DataType::Int16
                | DataType::Int32
                | DataType::Int64
                | DataType::UInt8
                | DataType::UInt16
                | DataType::UInt32
                | DataType::UInt64
                | DataType::Float32
                | DataType::Float64 => Some(s.name().to_string()),
                _ => None,
            })
            .collect()
    }

    pub fn select_columns(&self, columns: &[&str]) -> Result<DataFrame> {
        let col_refs: Vec<_> = columns.iter().map(|s| s.to_string()).collect();
        let df = self
            .frame
            .select(col_refs)
            .map_err(|_| MatheMixxError::ColumnNotFound(columns.join(",")))?;
        Ok(df)
    }

    pub fn drop_nulls(&self, columns: &[&str]) -> Result<DataFrame> {
        let df = self.select_columns(columns)?;
        let subset: Vec<String> = columns.iter().map(|s| s.to_string()).collect();
        Ok(df.drop_nulls(Some(&subset))?)
    }

    pub fn column(&self, name: &str) -> Result<Series> {
        let col = self
            .frame
            .column(name)
            .map_err(|_| MatheMixxError::ColumnNotFound(name.to_string()))?;
        Ok(col.as_materialized_series().clone())
    }

    pub fn prepare_numeric_frame(&self, columns: &[&str]) -> Result<DataFrame> {
        let df = self.drop_nulls(columns)?;
        let mut converted = Vec::with_capacity(columns.len());
        for name in columns {
            let series = df
                .column(name)
                .map_err(|_| MatheMixxError::ColumnNotFound((*name).to_string()))?;
            let casted = match series.dtype() {
                DataType::Float64 => series.clone(),
                DataType::Float32
                | DataType::Int8
                | DataType::Int16
                | DataType::Int32
                | DataType::Int64
                | DataType::UInt8
                | DataType::UInt16
                | DataType::UInt32
                | DataType::UInt64 => series.cast(&DataType::Float64)?,
                _ => {
                    return Err(MatheMixxError::Unsupported(
                        "Non-numeric column encountered while preparing design matrix",
                    ))
                }
            };
            converted.push(casted);
        }
        Ok(DataFrame::new(converted)?)
    }
}
