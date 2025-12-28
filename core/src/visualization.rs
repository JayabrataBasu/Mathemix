/// Phase 6: Data Visualization Module
///
/// Provides data structures and calculations for various plot types.
/// The actual plotting is done in Python (matplotlib/seaborn) but this module
/// provides the necessary data transformations and statistical calculations.
use crate::dataframe::DataSet;
use crate::errors::{MatheMixxError, Result};
use crate::ols::OlsResult;
use polars::prelude::*;
use serde::{Deserialize, Serialize};
use statrs::distribution::{ContinuousCDF, Normal};
use std::collections::HashMap;

// ============================================================================
// Regression Diagnostic Data
// ============================================================================

/// Data for residual vs fitted plot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResidualFittedData {
    pub fitted_values: Vec<f64>,
    pub residuals: Vec<f64>,
}

/// Data for Q-Q plot (normal probability plot)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QQPlotData {
    pub theoretical_quantiles: Vec<f64>,
    pub sample_quantiles: Vec<f64>,
}

/// Data for scale-location plot (sqrt standardized residuals vs fitted)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScaleLocationData {
    pub fitted_values: Vec<f64>,
    pub sqrt_abs_residuals: Vec<f64>,
}

/// Data for residuals vs leverage plot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResidualsLeverageData {
    pub leverage: Vec<f64>,
    pub standardized_residuals: Vec<f64>,
    pub cooks_distance: Vec<f64>,
}

/// Data for residual histogram
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResidualHistogramData {
    pub residuals: Vec<f64>,
    pub bins: usize,
}

// ============================================================================
// General Plot Data Structures
// ============================================================================

/// Data for box plot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoxPlotData {
    pub variable: String,
    pub min: f64,
    pub q1: f64,
    pub median: f64,
    pub q3: f64,
    pub max: f64,
    pub outliers: Vec<f64>,
    pub mean: f64,
}

/// Data for histogram with KDE
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistogramData {
    pub variable: String,
    pub values: Vec<f64>,
    pub bins: usize,
}

/// Data for correlation heatmap
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeatmapData {
    pub variables: Vec<String>,
    pub correlation_matrix: Vec<Vec<f64>>,
    pub method: String,
}

/// Data for pair plot (scatter matrix)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PairPlotData {
    pub variables: Vec<String>,
    pub data: HashMap<String, Vec<f64>>,
}

/// Data for violin plot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ViolinPlotData {
    pub variable: String,
    pub values: Vec<f64>,
}

// ============================================================================
// Regression Diagnostic Functions
// ============================================================================

impl OlsResult {
    /// Generate data for residual vs fitted plot
    pub fn residual_fitted_data(&self) -> Result<ResidualFittedData> {
        Ok(ResidualFittedData {
            fitted_values: self.fitted_values.clone(),
            residuals: self.residuals.clone(),
        })
    }

    /// Generate data for Q-Q plot
    pub fn qq_plot_data(&self) -> Result<QQPlotData> {
        let mut residuals = self.residuals.clone();
        residuals.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let n = residuals.len();
        let theoretical_quantiles: Vec<f64> = (0..n)
            .map(|i| {
                let p = (i as f64 + 0.5) / n as f64;
                normal_quantile(p)
            })
            .collect();

        Ok(QQPlotData {
            theoretical_quantiles,
            sample_quantiles: residuals,
        })
    }

    /// Generate data for scale-location plot
    pub fn scale_location_data(&self) -> Result<ScaleLocationData> {
        let standardized_residuals = self.standardized_residuals()?;
        let sqrt_abs_residuals: Vec<f64> = standardized_residuals
            .iter()
            .map(|r| r.abs().sqrt())
            .collect();

        Ok(ScaleLocationData {
            fitted_values: self.fitted_values.clone(),
            sqrt_abs_residuals,
        })
    }

    /// Generate data for residuals vs leverage plot
    pub fn residuals_leverage_data(&self) -> Result<ResidualsLeverageData> {
        let leverage = self.calculate_leverage()?;
        let standardized_residuals = self.standardized_residuals()?;
        let cooks_distance = self.calculate_cooks_distance(&leverage, &standardized_residuals)?;

        Ok(ResidualsLeverageData {
            leverage,
            standardized_residuals,
            cooks_distance,
        })
    }

    /// Generate data for residual histogram
    pub fn residual_histogram_data(&self, bins: Option<usize>) -> Result<ResidualHistogramData> {
        let n_bins = bins.unwrap_or_else(|| {
            // Sturges' formula
            let n = self.residuals.len();
            ((1.0_f64 + 3.322 * (n as f64).log10()).ceil()) as usize
        });

        Ok(ResidualHistogramData {
            residuals: self.residuals.clone(),
            bins: n_bins,
        })
    }

    // Helper methods
    fn standardized_residuals(&self) -> Result<Vec<f64>> {
        let residual_std = calculate_std(&self.residuals)?;
        Ok(self.residuals.iter().map(|r| r / residual_std).collect())
    }

    fn calculate_leverage(&self) -> Result<Vec<f64>> {
        // Leverage (hat values) = diagonal of H = X(X'X)^-1X'
        // For now, return simple approximation based on residuals
        // TODO: Implement proper hat matrix calculation
        let n = self.residuals.len();
        let k = self.coefficients.len();
        let avg_leverage = k as f64 / n as f64;

        // Simple approximation: leverage is higher for observations with larger residuals
        let residual_std = calculate_std(&self.residuals)?;
        Ok(self
            .residuals
            .iter()
            .map(|r| {
                let standardized = (r / residual_std).abs();
                (avg_leverage * (1.0 + standardized / 3.0)).min(3.0)
            })
            .collect())
    }

    fn calculate_cooks_distance(
        &self,
        leverage: &[f64],
        std_residuals: &[f64],
    ) -> Result<Vec<f64>> {
        let k = self.coefficients.len() as f64;
        Ok(leverage
            .iter()
            .zip(std_residuals.iter())
            .map(|(h, r)| {
                let numerator = r.powi(2) * h;
                let denominator = k * (1.0 - h);
                numerator / denominator
            })
            .collect())
    }
}

// ============================================================================
// General Visualization Functions
// ============================================================================

impl DataSet {
    /// Generate box plot data for a column
    pub fn box_plot_data(&self, column: &str) -> Result<BoxPlotData> {
        let series = self.get_column(column)?;
        let values = series
            .f64()
            .map_err(|e| MatheMixxError::Polars(e))?
            .into_no_null_iter()
            .collect::<Vec<_>>();

        if values.is_empty() {
            return Err(MatheMixxError::InvalidInput(format!(
                "Column '{}' has no valid numeric values",
                column
            )));
        }

        let mut sorted = values.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let n = sorted.len();
        let min = sorted[0];
        let max = sorted[n - 1];
        let median = calculate_median(&sorted);
        let q1 = calculate_percentile(&sorted, 25.0);
        let q3 = calculate_percentile(&sorted, 75.0);
        let mean = values.iter().sum::<f64>() / n as f64;

        // Calculate outliers (values beyond 1.5*IQR from quartiles)
        let iqr = q3 - q1;
        let lower_fence = q1 - 1.5 * iqr;
        let upper_fence = q3 + 1.5 * iqr;
        let outliers: Vec<f64> = values
            .iter()
            .filter(|&&v| v < lower_fence || v > upper_fence)
            .copied()
            .collect();

        Ok(BoxPlotData {
            variable: column.to_string(),
            min,
            q1,
            median,
            q3,
            max,
            outliers,
            mean,
        })
    }

    /// Generate histogram data for a column
    pub fn histogram_data(&self, column: &str, bins: Option<usize>) -> Result<HistogramData> {
        let series = self.get_column(column)?;
        let values = series
            .f64()
            .map_err(|e| MatheMixxError::Polars(e))?
            .into_no_null_iter()
            .collect::<Vec<_>>();

        if values.is_empty() {
            return Err(MatheMixxError::InvalidInput(format!(
                "Column '{}' has no valid numeric values",
                column
            )));
        }

        let n_bins = bins
            .unwrap_or_else(|| ((1.0_f64 + 3.322 * (values.len() as f64).log10()).ceil()) as usize);

        Ok(HistogramData {
            variable: column.to_string(),
            values,
            bins: n_bins,
        })
    }

    /// Generate correlation heatmap data
    pub fn heatmap_data(
        &self,
        columns: Option<Vec<String>>,
        method: Option<String>,
    ) -> Result<HeatmapData> {
        use crate::statistics::{CorrelationMatrix, CorrelationMethod};

        let corr_method = match method.as_deref().unwrap_or("pearson") {
            "pearson" => CorrelationMethod::Pearson,
            "spearman" => CorrelationMethod::Spearman,
            _ => CorrelationMethod::Pearson,
        };

        let corr_matrix: CorrelationMatrix = self.correlation(columns.clone(), corr_method)?;

        Ok(HeatmapData {
            variables: corr_matrix.variables,
            correlation_matrix: corr_matrix.matrix,
            method: format!("{:?}", corr_method),
        })
    }

    /// Generate pair plot data
    pub fn pair_plot_data(&self, columns: Option<Vec<String>>) -> Result<PairPlotData> {
        let cols = if let Some(c) = columns {
            c
        } else {
            self.numeric_column_names()
        };

        let mut data = HashMap::new();

        for col in &cols {
            let series = self.get_column(col)?;
            let values: Vec<f64> = series
                .f64()
                .map_err(|e| MatheMixxError::Polars(e))?
                .into_no_null_iter()
                .collect();
            data.insert(col.clone(), values);
        }

        Ok(PairPlotData {
            variables: cols,
            data,
        })
    }

    /// Generate violin plot data for a column
    pub fn violin_plot_data(&self, column: &str) -> Result<ViolinPlotData> {
        let series = self.get_column(column)?;
        let values = series
            .f64()
            .map_err(|e| MatheMixxError::Polars(e))?
            .into_no_null_iter()
            .collect::<Vec<_>>();

        if values.is_empty() {
            return Err(MatheMixxError::InvalidInput(format!(
                "Column '{}' has no valid numeric values",
                column
            )));
        }

        Ok(ViolinPlotData {
            variable: column.to_string(),
            values,
        })
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

fn calculate_std(values: &[f64]) -> Result<f64> {
    if values.is_empty() {
        return Err(MatheMixxError::InvalidInput(
            "Cannot calculate std of empty array".to_string(),
        ));
    }

    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let variance =
        values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (values.len() - 1) as f64;

    Ok(variance.sqrt())
}

fn calculate_median(sorted: &[f64]) -> f64 {
    let n = sorted.len();
    if n % 2 == 0 {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    } else {
        sorted[n / 2]
    }
}

fn calculate_percentile(sorted: &[f64], p: f64) -> f64 {
    let n = sorted.len();
    let rank = (p / 100.0) * (n - 1) as f64;
    let lower = rank.floor() as usize;
    let upper = rank.ceil() as usize;
    let fraction = rank - lower as f64;

    if lower == upper {
        sorted[lower]
    } else {
        sorted[lower] * (1.0 - fraction) + sorted[upper] * fraction
    }
}

/// Approximate normal quantile function (inverse CDF)
fn normal_quantile(p: f64) -> f64 {
    let normal = Normal::new(0.0, 1.0).unwrap();
    normal.inverse_cdf(p)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normal_quantile() {
        // Test standard normal quantiles
        assert!((normal_quantile(0.5) - 0.0).abs() < 0.01);
        assert!((normal_quantile(0.025) - (-1.96)).abs() < 0.01);
        assert!((normal_quantile(0.975) - 1.96).abs() < 0.01);
    }

    #[test]
    fn test_percentile_calculation() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(calculate_percentile(&data, 50.0), 3.0);
        assert!((calculate_percentile(&data, 25.0) - 2.0).abs() < 0.1);
        assert!((calculate_percentile(&data, 75.0) - 4.0).abs() < 0.1);
    }
}
