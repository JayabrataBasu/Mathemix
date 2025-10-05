// Phase 4: Data Manipulation - Column info and basic operations
use polars::prelude::*;
use serde::{Deserialize, Serialize};

use crate::errors::{MatheMixxError, Result};
use crate::DataSet;

/// Represents the data type of a column
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ColumnType {
    Numeric,
    Integer,
    String,
    Boolean,
    Temporal,
    Mixed,
}

/// Metadata about a column
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColumnInfo {
    pub name: String,
    pub dtype: ColumnType,
    pub null_count: usize,
    pub total_count: usize,
    pub null_percentage: f64,
}

/// Filter conditions for rows
#[derive(Debug, Clone, Copy)]
pub enum FilterCondition {
    GreaterThan(f64),
    LessThan(f64),
    Equal(f64),
    GreaterOrEqual(f64),
    LessOrEqual(f64),
    NotEqual(f64),
}

/// Column transformations
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum Transform {
    Log,
    Log10,
    Square,
    SquareRoot,
    Standardize,
    Center,
    Inverse,
}

impl DataSet {
    /// Get information about all columns
    pub fn column_info(&self) -> Result<Vec<ColumnInfo>> {
        let frame = self.dataframe();
        let mut infos = Vec::new();

        for col in frame.get_columns() {
            let name = col.name().to_string();
            let total_count = col.len();
            let null_count = col.null_count();
            let null_percentage = (null_count as f64 / total_count as f64) * 100.0;

            let dtype = match col.dtype() {
                DataType::Int8 | DataType::Int16 | DataType::Int32 | DataType::Int64 => {
                    ColumnType::Integer
                }
                DataType::UInt8 | DataType::UInt16 | DataType::UInt32 | DataType::UInt64 => {
                    ColumnType::Integer
                }
                DataType::Float32 | DataType::Float64 => ColumnType::Numeric,
                DataType::Boolean => ColumnType::Boolean,
                DataType::String => ColumnType::String,
                DataType::Date | DataType::Datetime(_, _) | DataType::Time | DataType::Duration(_) => {
                    ColumnType::Temporal
                }
                _ => ColumnType::Mixed,
            };

            infos.push(ColumnInfo {
                name,
                dtype,
                null_count,
                total_count,
                null_percentage,
            });
        }

        Ok(infos)
    }

    /// Get only numeric column names
    pub fn numeric_columns(&self) -> Result<Vec<String>> {
        let infos = self.column_info()?;
        Ok(infos
            .into_iter()
            .filter(|info| matches!(info.dtype, ColumnType::Numeric | ColumnType::Integer))
            .map(|info| info.name)
            .collect())
    }

    /// Check if a column is numeric
    pub fn is_numeric_column(&self, col_name: &str) -> Result<bool> {
        let col = self
            .dataframe()
            .column(col_name)
            .map_err(|e| MatheMixxError::ColumnNotFound(e.to_string()))?;

        Ok(matches!(
            col.dtype(),
            DataType::Int8
                | DataType::Int16
                | DataType::Int32
                | DataType::Int64
                | DataType::UInt8
                | DataType::UInt16
                | DataType::UInt32
                | DataType::UInt64
                | DataType::Float32
                | DataType::Float64
        ))
    }

    /// Select only numeric columns
    pub fn select_numeric(&self) -> Result<Self> {
        let numeric_cols = self.numeric_columns()?;
        let col_strs: Vec<_> = numeric_cols.iter().map(|s| s.as_str()).collect();
        let df = self.dataframe().select(col_strs)?;
        Ok(Self::from_frame(df))
    }

    /// Rename a column
    pub fn rename_column(&self, old_name: &str, new_name: &str) -> Result<Self> {
        let mut frame = self.clone_dataframe();
        frame.rename(old_name, new_name.into())?;
        Ok(Self::from_frame(frame))
    }

    /// Drop columns
    pub fn drop_columns(&self, columns: &[String]) -> Result<Self> {
        let col_strs: Vec<_> = columns.iter().map(|s| s.as_str()).collect();
        let frame = self.dataframe().drop_many(col_strs);
        Ok(Self::from_frame(frame))
    }

    /// Create a new column with a transformation
    pub fn add_column_transform(&self, source: &str, target: &str, transform: Transform) -> Result<Self> {
        let series = self
            .dataframe()
            .column(source)
            .map_err(|e| MatheMixxError::ColumnNotFound(e.to_string()))?
            .as_series()
            .ok_or_else(|| MatheMixxError::UnsupportedOperation("Column is not a series".into()))?;

        let new_series = match transform {
            Transform::Log => {
                let casted = series.cast(&DataType::Float64)?;
                let ca = casted.f64()?;
                ca.apply(|opt_val| opt_val.map(|v| v.ln())).into_series()
            }
            Transform::Log10 => {
                let casted = series.cast(&DataType::Float64)?;
                let ca = casted.f64()?;
                ca.apply(|opt_val| opt_val.map(|v| v.log10())).into_series()
            }
            Transform::Square => {
                let casted = series.cast(&DataType::Float64)?;
                let ca = casted.f64()?;
                ca.apply(|opt_val| opt_val.map(|v| v * v)).into_series()
            }
            Transform::SquareRoot => {
                let casted = series.cast(&DataType::Float64)?;
                let ca = casted.f64()?;
                ca.apply(|opt_val| opt_val.map(|v| v.sqrt())).into_series()
            }
            Transform::Standardize => {
                let mean = series.mean().unwrap_or(0.0);
                let std = series.std(1).unwrap_or(1.0);
                let casted = series.cast(&DataType::Float64)?;
                let ca = casted.f64()?;
                ca.apply(|opt_val| opt_val.map(|v| (v - mean) / std)).into_series()
            }
            Transform::Center => {
                let mean = series.mean().unwrap_or(0.0);
                let casted = series.cast(&DataType::Float64)?;
                let ca = casted.f64()?;
                ca.apply(|opt_val| opt_val.map(|v| v - mean)).into_series()
            }
            Transform::Inverse => {
                let casted = series.cast(&DataType::Float64)?;
                let ca = casted.f64()?;
                ca.apply(|opt_val| opt_val.map(|v| 1.0 / v)).into_series()
            }
        };

        let mut frame = self.clone_dataframe();
        let _ = frame.with_column(new_series.with_name(target.into()))?;
        Ok(Self::from_frame(frame))
    }

    /// Filter rows where column value meets condition
    pub fn filter_rows(&self, column: &str, condition: FilterCondition) -> Result<Self> {
        let series = self
            .dataframe()
            .column(column)
            .map_err(|e| MatheMixxError::ColumnNotFound(e.to_string()))?
            .as_series()
            .ok_or_else(|| MatheMixxError::UnsupportedOperation("Column is not a series".into()))?;

        let casted = series.cast(&DataType::Float64)?;
        let ca = casted.f64()?;

        let mask = match condition {
            FilterCondition::GreaterThan(val) => ca.gt(val),
            FilterCondition::LessThan(val) => ca.lt(val),
            FilterCondition::Equal(val) => ca.equal(val),
            FilterCondition::GreaterOrEqual(val) => ca.gt_eq(val),
            FilterCondition::LessOrEqual(val) => ca.lt_eq(val),
            FilterCondition::NotEqual(val) => ca.not_equal(val),
        };

        let filtered = self.dataframe().filter(&mask)?;
        Ok(Self::from_frame(filtered))
    }
}
