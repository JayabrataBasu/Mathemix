// Time series operations: lag, diff, moving averages, rolling statistics

/// Lag a time series by a specified number of periods
/// Returns a new series with NaN for the lagged positions
pub fn lag(data: &[f64], periods: usize) -> Vec<f64> {
    if periods == 0 {
        return data.to_vec();
    }

    let mut result = vec![f64::NAN; data.len()];
    for i in periods..data.len() {
        result[i] = data[i - periods];
    }
    result
}

/// Difference a time series (subtract previous value)
/// Returns a new series with NaN for the first `periods` positions
pub fn diff(data: &[f64], periods: usize) -> Vec<f64> {
    if periods == 0 {
        return data.to_vec();
    }

    let mut result = vec![f64::NAN; data.len()];
    for i in periods..data.len() {
        result[i] = data[i] - data[i - periods];
    }
    result
}

/// Simple Moving Average (SMA)
/// Returns a new series with NaN for positions where window is incomplete
pub fn sma(data: &[f64], window: usize) -> Vec<f64> {
    if window == 0 || window > data.len() {
        return vec![f64::NAN; data.len()];
    }

    rolling_mean(data, window)
}

/// Weighted Moving Average (WMA)
/// Weights are linearly decreasing: [n, n-1, ..., 2, 1]
pub fn wma(data: &[f64], window: usize) -> Vec<f64> {
    if window == 0 || window > data.len() {
        return vec![f64::NAN; data.len()];
    }

    let mut result = vec![f64::NAN; data.len()];

    // Calculate weights: [1, 2, 3, ..., window]
    let weights: Vec<f64> = (1..=window).map(|x| x as f64).collect();
    let weight_sum: f64 = weights.iter().sum();

    for i in (window - 1)..data.len() {
        let mut weighted_sum = 0.0;
        for j in 0..window {
            weighted_sum += data[i - (window - 1 - j)] * weights[j];
        }
        result[i] = weighted_sum / weight_sum;
    }

    result
}

/// Exponential Moving Average (EMA)
/// Alpha is the smoothing factor (0 < alpha <= 1)
/// Higher alpha gives more weight to recent observations
pub fn ema(data: &[f64], window: usize) -> Vec<f64> {
    if window == 0 || window > data.len() {
        return vec![f64::NAN; data.len()];
    }

    let mut result = vec![f64::NAN; data.len()];

    // Alpha = 2 / (window + 1) is common for EMA
    let alpha = 2.0 / (window as f64 + 1.0);

    // Initialize with first value
    result[0] = data[0];

    // Apply exponential smoothing
    for i in 1..data.len() {
        result[i] = alpha * data[i] + (1.0 - alpha) * result[i - 1];
    }

    // Set first window-1 values to NaN for consistency
    for i in 0..(window - 1).min(result.len()) {
        result[i] = f64::NAN;
    }

    result
}

/// Rolling mean (simple moving average)
pub fn rolling_mean(data: &[f64], window: usize) -> Vec<f64> {
    if window == 0 || window > data.len() {
        return vec![f64::NAN; data.len()];
    }

    let mut result = vec![f64::NAN; data.len()];

    for i in (window - 1)..data.len() {
        let start = i.saturating_sub(window - 1);
        let sum: f64 = data[start..=i].iter().sum();
        result[i] = sum / window as f64;
    }

    result
}

/// Rolling standard deviation
pub fn rolling_std(data: &[f64], window: usize) -> Vec<f64> {
    if window == 0 || window > data.len() || window < 2 {
        return vec![f64::NAN; data.len()];
    }

    let mut result = vec![f64::NAN; data.len()];

    for i in (window - 1)..data.len() {
        let start = i.saturating_sub(window - 1);
        let slice = &data[start..=i];
        let mean: f64 = slice.iter().sum::<f64>() / window as f64;
        let variance: f64 =
            slice.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (window - 1) as f64;
        result[i] = variance.sqrt();
    }

    result
}

/// Rolling minimum
pub fn rolling_min(data: &[f64], window: usize) -> Vec<f64> {
    if window == 0 || window > data.len() {
        return vec![f64::NAN; data.len()];
    }

    let mut result = vec![f64::NAN; data.len()];

    for i in (window - 1)..data.len() {
        let start = i.saturating_sub(window - 1);
        let min = data[start..=i].iter().fold(f64::INFINITY, |a, &b| a.min(b));
        result[i] = min;
    }

    result
}

/// Rolling maximum
pub fn rolling_max(data: &[f64], window: usize) -> Vec<f64> {
    if window == 0 || window > data.len() {
        return vec![f64::NAN; data.len()];
    }

    let mut result = vec![f64::NAN; data.len()];

    for i in (window - 1)..data.len() {
        let start = i.saturating_sub(window - 1);
        let max = data[start..=i]
            .iter()
            .fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        result[i] = max;
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lag() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let lagged = lag(&data, 1);
        assert!(lagged[0].is_nan());
        assert_eq!(lagged[1], 1.0);
        assert_eq!(lagged[2], 2.0);
        assert_eq!(lagged[4], 4.0);
    }

    #[test]
    fn test_diff() {
        let data = vec![1.0, 3.0, 6.0, 10.0, 15.0];
        let differenced = diff(&data, 1);
        assert!(differenced[0].is_nan());
        assert_eq!(differenced[1], 2.0);
        assert_eq!(differenced[2], 3.0);
        assert_eq!(differenced[3], 4.0);
        assert_eq!(differenced[4], 5.0);
    }

    #[test]
    fn test_sma() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let ma = sma(&data, 3);
        assert!(ma[0].is_nan());
        assert!(ma[1].is_nan());
        assert_eq!(ma[2], 2.0); // (1+2+3)/3
        assert_eq!(ma[3], 3.0); // (2+3+4)/3
        assert_eq!(ma[4], 4.0); // (3+4+5)/3
    }

    #[test]
    fn test_ema() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let ma = ema(&data, 3);
        assert!(ma[0].is_nan());
        assert!(ma[1].is_nan());
        // EMA starts from index window-1
        assert!(ma[2] > 0.0 && ma[2] < 6.0);
    }

    #[test]
    fn test_rolling_std() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let std = rolling_std(&data, 3);
        assert!(std[0].is_nan());
        assert!(std[1].is_nan());
        // Std of [1,2,3] = 1.0
        assert!((std[2] - 1.0).abs() < 0.0001);
    }

    #[test]
    fn test_rolling_min_max() {
        let data = vec![3.0, 1.0, 4.0, 1.0, 5.0];
        let min = rolling_min(&data, 3);
        let max = rolling_max(&data, 3);

        assert_eq!(min[2], 1.0); // min of [3,1,4]
        assert_eq!(max[2], 4.0); // max of [3,1,4]
        assert_eq!(min[4], 1.0); // min of [4,1,5]
        assert_eq!(max[4], 5.0); // max of [4,1,5]
    }
}
