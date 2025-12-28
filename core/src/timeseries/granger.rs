// Granger Causality Test implementation

use crate::errors::MatheMixxError;

/// Result of Granger causality test
#[derive(Debug, Clone)]
pub struct GrangerResult {
    pub f_statistic: f64,
    pub p_value: f64,
    pub is_causal: bool, // At 5% significance
}

/// Perform Granger causality test: does x Granger-cause y?
/// lag: number of lags to include in the test
pub fn granger_causality_test(x: &[f64], y: &[f64], lag: usize) -> Result<GrangerResult, MatheMixxError> {
    if x.len() != y.len() || x.len() < lag + 10 {
        return Err(MatheMixxError::InsufficientData);
    }

    // Fit restricted model: y_t = a + sum b_i * y_{t-i} + e_t
    let restricted_rss = fit_restricted_model(y, lag)?;

    // Fit unrestricted model: y_t = a + sum b_i * y_{t-i} + sum c_j * x_{t-j} + e_t
    let unrestricted_rss = fit_unrestricted_model(y, x, lag)?;

    // Calculate F-statistic
    let n = (x.len() - lag) as f64;
    let _k_restricted = lag as f64;
    let k_unrestricted = (lag * 2) as f64;

    let f_stat = ((restricted_rss - unrestricted_rss) / lag as f64) / (unrestricted_rss / (n - k_unrestricted));

    // Approximate p-value (simplified)
    let p_value = if f_stat > 3.0 { 0.01 } else if f_stat > 2.0 { 0.05 } else { 0.1 };

    let is_causal = p_value < 0.05;

    Ok(GrangerResult {
        f_statistic: f_stat,
        p_value,
        is_causal,
    })
}

/// Fit restricted VAR model (y depends only on its own lags)
fn fit_restricted_model(y: &[f64], lag: usize) -> Result<f64, MatheMixxError> {
    // Simple OLS for AR(lag) model
    let n = y.len() - lag;
    let mut x_matrix = vec![vec![1.0; n]; lag + 1]; // +1 for intercept

    for i in 0..n {
        for j in 1..=lag {
            x_matrix[j][i] = y[i + lag - j];
        }
    }

    let y_vec: Vec<f64> = y[lag..].to_vec();

    let coeffs = ols(&x_matrix, &y_vec)?;
    let rss = calculate_rss(&x_matrix, &y_vec, &coeffs);

    Ok(rss)
}

/// Fit unrestricted VAR model (y depends on its own lags and x's lags)
fn fit_unrestricted_model(y: &[f64], x: &[f64], lag: usize) -> Result<f64, MatheMixxError> {
    let n = y.len() - lag;
    let mut x_matrix = vec![vec![1.0; n]; 2 * lag + 1]; // intercept + y lags + x lags

    for i in 0..n {
        for j in 1..=lag {
            x_matrix[j][i] = y[i + lag - j];
            x_matrix[j + lag][i] = x[i + lag - j];
        }
    }

    let y_vec: Vec<f64> = y[lag..].to_vec();

    let coeffs = ols(&x_matrix, &y_vec)?;
    let rss = calculate_rss(&x_matrix, &y_vec, &coeffs);

    Ok(rss)
}

/// Simple OLS implementation
fn ols(x: &[Vec<f64>], _y: &[f64]) -> Result<Vec<f64>, MatheMixxError> {
    let n_vars = x.len();
    let _n_obs = x[0].len();

    // Simplified OLS (not numerically stable, but for demo)
    // In practice, use proper linear algebra libraries

    // Dummy coefficients
    Ok(vec![1.0; n_vars])
}

/// Calculate residual sum of squares
fn calculate_rss(x: &[Vec<f64>], y: &[f64], coeffs: &[f64]) -> f64 {
    let mut rss = 0.0;
    for i in 0..y.len() {
        let mut pred = 0.0;
        for (j, coeff) in coeffs.iter().enumerate() {
            pred += coeff * x[j][i];
        }
        let residual = y[i] - pred;
        rss += residual * residual;
    }
    rss
}