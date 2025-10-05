// Phase 5: Enhanced Statistical Methods
use polars::prelude::*;
use serde::{Deserialize, Serialize};

use crate::errors::{MatheMixxError, Result};
use crate::DataSet;

/// Correlation method
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CorrelationMethod {
    Pearson,
    Spearman,
}

/// Result of correlation analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorrelationMatrix {
    pub variables: Vec<String>,
    pub matrix: Vec<Vec<f64>>,
    pub method: String,
}

impl CorrelationMatrix {
    /// Get correlation between two variables
    pub fn get(&self, var1: &str, var2: &str) -> Option<f64> {
        let idx1 = self.variables.iter().position(|v| v == var1)?;
        let idx2 = self.variables.iter().position(|v| v == var2)?;
        Some(self.matrix[idx1][idx2])
    }
}

/// Enhanced summary statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnhancedSummary {
    pub variable: String,
    pub count: usize,
    pub null_count: usize,
    pub mean: f64,
    pub median: f64,
    pub std: f64,
    pub variance: f64,
    pub min: f64,
    pub max: f64,
    pub q25: f64,
    pub q50: f64,
    pub q75: f64,
    pub range: f64,
    pub iqr: f64,
    pub skewness: Option<f64>,
    pub kurtosis: Option<f64>,
}

/// Frequency table entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrequencyRow {
    pub value: String,
    pub count: usize,
    pub percentage: f64,
    pub cumulative_count: usize,
    pub cumulative_percentage: f64,
}

/// Frequency table result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrequencyTable {
    pub variable: String,
    pub total_count: usize,
    pub unique_count: usize,
    pub rows: Vec<FrequencyRow>,
}

impl DataSet {
    /// Calculate correlation matrix
    pub fn correlation(
        &self,
        columns: Option<Vec<String>>,
        method: CorrelationMethod,
    ) -> Result<CorrelationMatrix> {
        let cols = if let Some(c) = columns {
            c
        } else {
            self.numeric_columns()?
        };

        if cols.is_empty() {
            return Err(MatheMixxError::InvalidInput(
                "No numeric columns found for correlation".into(),
            ));
        }

        let n = cols.len();
        let mut matrix = vec![vec![0.0; n]; n];

        for (i, col1) in cols.iter().enumerate() {
            for (j, col2) in cols.iter().enumerate() {
                if i == j {
                    matrix[i][j] = 1.0;
                } else if i < j {
                    let corr = self.correlation_pair(col1, col2, method)?;
                    matrix[i][j] = corr;
                    matrix[j][i] = corr; // Symmetric
                }
            }
        }

        Ok(CorrelationMatrix {
            variables: cols,
            matrix,
            method: match method {
                CorrelationMethod::Pearson => "pearson".to_string(),
                CorrelationMethod::Spearman => "spearman".to_string(),
            },
        })
    }

    /// Calculate correlation between two columns
    fn correlation_pair(&self, col1: &str, col2: &str, method: CorrelationMethod) -> Result<f64> {
        let s1 = self
            .dataframe()
            .column(col1)?
            .as_series()
            .ok_or_else(|| MatheMixxError::UnsupportedOperation("Column is not a series".into()))?
            .cast(&DataType::Float64)?;
        let s2 = self
            .dataframe()
            .column(col2)?
            .as_series()
            .ok_or_else(|| MatheMixxError::UnsupportedOperation("Column is not a series".into()))?
            .cast(&DataType::Float64)?;

        let corr = match method {
            CorrelationMethod::Pearson => self.pearson_correlation(&s1, &s2)?,
            CorrelationMethod::Spearman => self.spearman_correlation(&s1, &s2)?,
        };

        Ok(corr)
    }

    /// Pearson correlation coefficient
    fn pearson_correlation(&self, s1: &Series, s2: &Series) -> Result<f64> {
        let a1 = s1.f64()?;
        let a2 = s2.f64()?;

        let mut sum_x = 0.0;
        let mut sum_y = 0.0;
        let mut sum_xy = 0.0;
        let mut sum_x2 = 0.0;
        let mut sum_y2 = 0.0;
        let mut n = 0;

        for (opt_x, opt_y) in a1.into_iter().zip(a2.into_iter()) {
            if let (Some(x), Some(y)) = (opt_x, opt_y) {
                sum_x += x;
                sum_y += y;
                sum_xy += x * y;
                sum_x2 += x * x;
                sum_y2 += y * y;
                n += 1;
            }
        }

        if n < 2 {
            return Err(MatheMixxError::InvalidInput(
                "Not enough valid observations for correlation".into(),
            ));
        }

        let n_f64 = n as f64;
        let numerator = n_f64 * sum_xy - sum_x * sum_y;
        let denominator =
            ((n_f64 * sum_x2 - sum_x * sum_x) * (n_f64 * sum_y2 - sum_y * sum_y)).sqrt();

        if denominator == 0.0 {
            Ok(0.0)
        } else {
            Ok(numerator / denominator)
        }
    }

    /// Spearman rank correlation
    fn spearman_correlation(&self, s1: &Series, s2: &Series) -> Result<f64> {
        // For Spearman, we rank the values and then calculate Pearson on ranks
        // Simplified implementation - just use Pearson for now
        // TODO: Implement proper ranking
        self.pearson_correlation(s1, s2)
    }

    /// Enhanced summary statistics with quartiles
    pub fn enhanced_summary(&self, columns: Option<Vec<String>>) -> Result<Vec<EnhancedSummary>> {
        let cols = if let Some(c) = columns {
            c
        } else {
            self.numeric_columns()?
        };

        let mut results = Vec::new();

        for col_name in cols {
            if !self.is_numeric_column(&col_name)? {
                continue;
            }

            let series = self
                .dataframe()
                .column(&col_name)?
                .as_series()
                .ok_or_else(|| {
                    MatheMixxError::UnsupportedOperation("Column is not a series".into())
                })?;

            let total = series.len();
            let null_count = series.null_count();

            let mean = series.mean().unwrap_or(0.0);
            let median = series.median().unwrap_or(0.0);
            let std = series.std(1).unwrap_or(0.0);
            let variance = std * std;

            // Min/max
            let min = series.min::<f64>().ok().flatten().unwrap_or(0.0);
            let max = series.max::<f64>().ok().flatten().unwrap_or(0.0);

            // Quantiles - simplified (will improve later)
            let q25 = min + (max - min) * 0.25; // Rough approximation
            let q50 = median;
            let q75 = min + (max - min) * 0.75; // Rough approximation

            let range = max - min;
            let iqr = q75 - q25;

            // Calculate skewness and kurtosis
            let (skewness, kurtosis) = self.calculate_moments(series, mean, std)?;

            results.push(EnhancedSummary {
                variable: col_name,
                count: total - null_count,
                null_count,
                mean,
                median,
                std,
                variance,
                min,
                max,
                q25,
                q50,
                q75,
                range,
                iqr,
                skewness,
                kurtosis,
            });
        }

        Ok(results)
    }

    /// Calculate skewness and kurtosis
    fn calculate_moments(
        &self,
        series: &Series,
        mean: f64,
        std: f64,
    ) -> Result<(Option<f64>, Option<f64>)> {
        if std == 0.0 {
            return Ok((None, None));
        }

        let ca = series.f64()?;
        let mut m3 = 0.0;
        let mut m4 = 0.0;
        let mut n = 0;

        for opt_val in ca.into_iter() {
            if let Some(val) = opt_val {
                let z = (val - mean) / std;
                m3 += z.powi(3);
                m4 += z.powi(4);
                n += 1;
            }
        }

        if n < 3 {
            return Ok((None, None));
        }

        let n_f64 = n as f64;
        let skewness = m3 / n_f64;
        let kurtosis = (m4 / n_f64) - 3.0; // Excess kurtosis

        Ok((Some(skewness), Some(kurtosis)))
    }

    /// Create frequency table for a column
    pub fn frequency_table(&self, column: &str, limit: Option<usize>) -> Result<FrequencyTable> {
        let series = self
            .dataframe()
            .column(column)?
            .as_series()
            .ok_or_else(|| MatheMixxError::UnsupportedOperation("Column is not a series".into()))?;

        let total_count = series.len();

        // Get value counts
        let value_counts = series.value_counts(true, true, "count".into(), false)?;
        let values = value_counts.column(column)?;
        let counts = value_counts.column("count")?.cast(&DataType::UInt64)?;

        let unique_count = values.len();
        let mut rows = Vec::new();
        let mut cumulative_count = 0;

        let iter_limit = limit.unwrap_or(unique_count).min(unique_count);

        for i in 0..iter_limit {
            let value = values.get(i).unwrap().to_string();
            let count = counts.u64()?.get(i).unwrap() as usize;
            cumulative_count += count;

            let percentage = (count as f64 / total_count as f64) * 100.0;
            let cumulative_percentage = (cumulative_count as f64 / total_count as f64) * 100.0;

            rows.push(FrequencyRow {
                value,
                count,
                percentage,
                cumulative_count,
                cumulative_percentage,
            });
        }

        Ok(FrequencyTable {
            variable: column.to_string(),
            total_count,
            unique_count,
            rows,
        })
    }
}
