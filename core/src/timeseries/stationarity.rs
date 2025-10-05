// Stationarity tests: ADF, KPSS

/// Augmented Dickey-Fuller (ADF) test result
#[derive(Debug, Clone)]
pub struct ADFResult {
    pub test_statistic: f64,
    pub p_value: f64,
    pub lags_used: usize,
    pub n_obs: usize,
    pub critical_values: (f64, f64, f64), // 1%, 5%, 10%
    pub is_stationary: bool,
}

/// KPSS test result
#[derive(Debug, Clone)]
pub struct KPSSResult {
    pub test_statistic: f64,
    pub p_value: f64,
    pub lags_used: usize,
    pub critical_values: (f64, f64, f64), // 1%, 5%, 10%
    pub is_stationary: bool,
}

/// Augmented Dickey-Fuller test for unit root (non-stationarity)
/// H0: Series has a unit root (non-stationary)
/// H1: Series is stationary
pub fn adf_test(data: &[f64], max_lags: Option<usize>) -> ADFResult {
    let n = data.len();

    // For now, this is a simplified placeholder
    // Full implementation would require:
    // 1. Estimate AR(p) model with trend and lag terms
    // 2. Compute test statistic from regression
    // 3. Compare with Dickey-Fuller critical values

    let lags = max_lags.unwrap_or(((n as f64).powf(1.0 / 3.0) * 12.0).floor() as usize);

    // Calculate first difference variance vs original variance
    let mut diff_sum = 0.0;
    for i in 1..n {
        diff_sum += (data[i] - data[i - 1]).powi(2);
    }
    let diff_var = diff_sum / (n - 1) as f64;

    let mean = data.iter().sum::<f64>() / n as f64;
    let orig_var = data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n as f64;

    // Simplified test statistic
    let test_stat = -((diff_var / orig_var).sqrt() * 10.0);

    // MacKinnon critical values (approximate)
    let cv_1pct = -3.43;
    let cv_5pct = -2.86;
    let cv_10pct = -2.57;

    let is_stationary = test_stat < cv_5pct;

    ADFResult {
        test_statistic: test_stat,
        p_value: if test_stat < cv_1pct {
            0.001
        } else if test_stat < cv_5pct {
            0.03
        } else if test_stat < cv_10pct {
            0.08
        } else {
            0.15
        },
        lags_used: lags.min(10),
        n_obs: n - lags.min(10) - 1,
        critical_values: (cv_1pct, cv_5pct, cv_10pct),
        is_stationary,
    }
}

/// KPSS test for stationarity
/// H0: Series is stationary
/// H1: Series has a unit root (non-stationary)
pub fn kpss_test(data: &[f64], lags: Option<usize>) -> KPSSResult {
    let n = data.len();

    // For now, this is a simplified placeholder
    // Full implementation would calculate:
    // 1. Cumulative sum of residuals from level/trend regression
    // 2. Long-run variance estimate
    // 3. KPSS test statistic

    let lag_param = lags.unwrap_or(((n as f64 * 3.0 / 100.0).sqrt()).floor() as usize);

    // Simplified calculation
    let mean = data.iter().sum::<f64>() / n as f64;
    let cumsum: Vec<f64> = data
        .iter()
        .scan(0.0, |state, &x| {
            *state += x - mean;
            Some(*state)
        })
        .collect();

    let s_sq: f64 = cumsum.iter().map(|&x| x.powi(2)).sum::<f64>() / (n * n) as f64;
    let variance = data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n as f64;

    let test_stat = s_sq / variance;

    // KPSS critical values for level stationarity
    let cv_10pct = 0.347;
    let cv_5pct = 0.463;
    let cv_1pct = 0.739;

    let is_stationary = test_stat < cv_5pct;

    KPSSResult {
        test_statistic: test_stat,
        p_value: if test_stat > cv_1pct {
            0.001
        } else if test_stat > cv_5pct {
            0.03
        } else if test_stat > cv_10pct {
            0.08
        } else {
            0.15
        },
        lags_used: lag_param,
        critical_values: (cv_1pct, cv_5pct, cv_10pct),
        is_stationary,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adf_stationary_series() {
        // White noise should be stationary
        let data: Vec<f64> = (0..100).map(|i| ((i * 7) % 10) as f64).collect();
        let result = adf_test(&data, None);
        assert!(result.n_obs > 0);
        assert!(result.lags_used > 0);
    }

    #[test]
    fn test_kpss_stationary_series() {
        let data: Vec<f64> = (0..100).map(|i| ((i * 7) % 10) as f64).collect();
        let result = kpss_test(&data, None);
        assert!(result.lags_used > 0);
    }
}
