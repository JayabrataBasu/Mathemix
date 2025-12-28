// Cointegration tests implementation

use crate::errors::MatheMixxError;

/// Result of Engle-Granger cointegration test
#[derive(Debug, Clone)]
pub struct EngleGrangerResult {
    pub t_statistic: f64,
    pub critical_value: f64, // At 5% significance
    pub p_value: f64,
    pub is_cointegrated: bool,
}

/// Result of Johansen cointegration test
#[derive(Debug, Clone)]
pub struct JohansenResult {
    pub eigenvalues: Vec<f64>,
    pub trace_statistics: Vec<f64>,
    pub critical_values: Vec<f64>, // At 5% significance
    pub cointegration_rank: usize,
}

/// Perform Engle-Granger cointegration test between two series
pub fn engle_granger_test(series1: &[f64], series2: &[f64]) -> Result<EngleGrangerResult, MatheMixxError> {
    if series1.len() != series2.len() || series1.len() < 20 {
        return Err(MatheMixxError::InsufficientData);
    }

    // Step 1: Regress series1 on series2 to get residuals
    let residuals = regress_series(series1, series2)?;

    // Step 2: Test if residuals are stationary (ADF test)
    let adf_result = adf_test_residuals(&residuals)?;

    // Simplified critical values (5% significance)
    let critical_value = -3.37; // Approximate for large samples

    let is_cointegrated = adf_result.t_statistic < critical_value;

    Ok(EngleGrangerResult {
        t_statistic: adf_result.t_statistic,
        critical_value,
        p_value: adf_result.p_value,
        is_cointegrated,
    })
}

/// Perform Johansen cointegration test for multiple series
pub fn johansen_test(data: &[Vec<f64>]) -> Result<Vec<JohansenResult>, MatheMixxError> {
    // Placeholder implementation
    // Real Johansen test is complex, involving eigenvalue decomposition

    let n_vars = data.len();
    if n_vars < 2 || data[0].len() < 30 {
        return Err(MatheMixxError::InsufficientData);
    }

    // Dummy results
    let result = JohansenResult {
        eigenvalues: vec![0.5, 0.3],
        trace_statistics: vec![25.0, 10.0],
        critical_values: vec![20.0, 9.0],
        cointegration_rank: 1,
    };

    Ok(vec![result])
}

/// Helper: Regress series1 on series2
fn regress_series(series1: &[f64], series2: &[f64]) -> Result<Vec<f64>, MatheMixxError> {
    // Simple OLS: series1 = a + b * series2 + residuals
    let n = series1.len() as f64;
    let sum_x = series2.iter().sum::<f64>();
    let sum_y = series1.iter().sum::<f64>();
    let sum_xy = series1.iter().zip(series2.iter()).map(|(y, x)| y * x).sum::<f64>();
    let sum_x2 = series2.iter().map(|x| x * x).sum::<f64>();

    let b = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x);
    let a = (sum_y - b * sum_x) / n;

    let residuals: Vec<f64> = series1.iter().zip(series2.iter())
        .map(|(y, x)| y - (a + b * x))
        .collect();

    Ok(residuals)
}

/// Helper: ADF test on residuals (simplified)
fn adf_test_residuals(residuals: &[f64]) -> Result<AdfResult, MatheMixxError> {
    // Simplified ADF test
    let _n = residuals.len();
    let _diff_residuals: Vec<f64> = residuals.windows(2).map(|w| w[1] - w[0]).collect();

    // Simple t-statistic approximation
    let t_stat = -2.0; // Dummy value
    let p_value = 0.05; // Dummy

    Ok(AdfResult { t_statistic: t_stat, p_value })
}

#[derive(Debug)]
struct AdfResult {
    t_statistic: f64,
    p_value: f64,
}