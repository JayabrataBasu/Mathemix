// Autocorrelation and partial autocorrelation functions

/// Calculate Autocorrelation Function (ACF)
/// Returns ACF values for lags 0 to nlags
pub fn acf(data: &[f64], nlags: usize) -> Vec<f64> {
    let n = data.len();
    if n == 0 || nlags >= n {
        return vec![];
    }

    // Calculate mean
    let mean = data.iter().sum::<f64>() / n as f64;

    // Calculate variance (lag 0 autocovariance)
    let variance = data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n as f64;

    if variance == 0.0 {
        return vec![1.0; nlags + 1];
    }

    let mut acf_values = vec![1.0]; // ACF at lag 0 is always 1

    // Calculate ACF for each lag
    for lag in 1..=nlags {
        let mut covariance = 0.0;
        for i in 0..(n - lag) {
            covariance += (data[i] - mean) * (data[i + lag] - mean);
        }
        covariance /= n as f64;
        acf_values.push(covariance / variance);
    }

    acf_values
}

/// Calculate Partial Autocorrelation Function (PACF)
/// Uses the Yule-Walker equations
pub fn pacf(data: &[f64], nlags: usize) -> Vec<f64> {
    let n = data.len();
    if n == 0 || nlags >= n {
        return vec![];
    }

    // First get ACF values
    let acf_values = acf(data, nlags);

    let mut pacf_values = vec![1.0]; // PACF at lag 0 is always 1

    if nlags == 0 {
        return pacf_values;
    }

    // PACF at lag 1 equals ACF at lag 1
    if nlags >= 1 {
        pacf_values.push(acf_values[1]);
    }

    // For lag >= 2, use Durbin-Levinson recursion
    let mut phi = vec![vec![0.0; nlags + 1]; nlags + 1];
    phi[1][1] = acf_values[1];

    for k in 2..=nlags {
        // Calculate numerator
        let mut numerator = acf_values[k];
        for j in 1..k {
            numerator -= phi[k - 1][j] * acf_values[k - j];
        }

        // Calculate denominator
        let mut denominator = 1.0;
        for j in 1..k {
            denominator -= phi[k - 1][j] * acf_values[j];
        }

        if denominator.abs() < 1e-10 {
            pacf_values.push(0.0);
            continue;
        }

        phi[k][k] = numerator / denominator;

        // Update phi values
        for j in 1..k {
            phi[k][j] = phi[k - 1][j] - phi[k][k] * phi[k - 1][k - j];
        }

        pacf_values.push(phi[k][k]);
    }

    pacf_values
}

/// Ljung-Box test for autocorrelation
/// Returns (test_statistic, p_value)
/// H0: No autocorrelation up to lag h
pub fn ljung_box_test(data: &[f64], lags: usize) -> (f64, f64) {
    let n = data.len() as f64;
    let acf_values = acf(data, lags);

    // Calculate Q statistic
    let mut q = 0.0;
    for k in 1..=lags {
        q += acf_values[k].powi(2) / (n - k as f64);
    }
    q *= n * (n + 2.0);

    // For p-value, we'd need chi-square distribution
    // For now, return the test statistic and a placeholder p-value
    // In full implementation, use statrs or similar for chi-square CDF
    let p_value = if q > 20.0 {
        0.001
    } else if q > 10.0 {
        0.05
    } else {
        0.5
    };

    (q, p_value)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_acf_constant_series() {
        let data = vec![5.0; 100];
        let acf_vals = acf(&data, 10);
        // For constant series, variance is 0, so ACF should be all 1s
        assert_eq!(acf_vals.len(), 11);
        assert_eq!(acf_vals[0], 1.0);
    }

    #[test]
    fn test_acf_white_noise() {
        // For white noise, ACF should be ~0 for all lags > 0
        // Using a simple pattern that resembles white noise
        let data: Vec<f64> = (0..100).map(|i| ((i * 7) % 10) as f64).collect();
        let acf_vals = acf(&data, 5);
        assert_eq!(acf_vals[0], 1.0); // Lag 0 is always 1
        assert_eq!(acf_vals.len(), 6);
    }

    #[test]
    fn test_pacf_length() {
        let data: Vec<f64> = (0..50).map(|i| i as f64).collect();
        let pacf_vals = pacf(&data, 10);
        assert_eq!(pacf_vals.len(), 11); // Includes lag 0
        assert_eq!(pacf_vals[0], 1.0); // PACF at lag 0 is always 1
    }

    #[test]
    fn test_ljung_box() {
        let data: Vec<f64> = (0..100).map(|i| (i as f64).sin()).collect();
        let (q, _p) = ljung_box_test(&data, 10);
        assert!(q > 0.0);
    }
}
