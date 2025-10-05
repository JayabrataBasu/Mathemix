// Seasonal decomposition methods

use super::operations::sma;

/// Result of seasonal decomposition
#[derive(Debug, Clone)]
pub struct DecompositionResult {
    pub trend: Vec<f64>,
    pub seasonal: Vec<f64>,
    pub residual: Vec<f64>,
    pub observed: Vec<f64>,
}

/// Decomposition type
#[derive(Debug, Clone, Copy)]
pub enum DecompType {
    Additive,      // Y = T + S + R
    Multiplicative, // Y = T * S * R
}

/// Classical seasonal decomposition
/// period: seasonality period (e.g., 12 for monthly data with yearly seasonality)
pub fn seasonal_decompose(
    data: &[f64],
    period: usize,
    model: DecompType,
) -> Result<DecompositionResult, String> {
    let n = data.len();
    
    if period == 0 || period >= n {
        return Err("Invalid period for decomposition".to_string());
    }
    
    if n < 2 * period {
        return Err("Need at least 2 full periods for decomposition".to_string());
    }
    
    // Step 1: Calculate trend using centered moving average
    let trend = calculate_trend(data, period);
    
    // Step 2: Detrend the series
    let detrended: Vec<f64> = match model {
        DecompType::Additive => {
            data.iter()
                .zip(trend.iter())
                .map(|(&y, &t)| if t.is_nan() { f64::NAN } else { y - t })
                .collect()
        }
        DecompType::Multiplicative => {
            data.iter()
                .zip(trend.iter())
                .map(|(&y, &t)| if t.is_nan() || t == 0.0 { f64::NAN } else { y / t })
                .collect()
        }
    };
    
    // Step 3: Calculate seasonal component
    let seasonal = calculate_seasonal(&detrended, period, model);
    
    // Step 4: Calculate residuals
    let residual: Vec<f64> = match model {
        DecompType::Additive => {
            data.iter()
                .zip(trend.iter())
                .zip(seasonal.iter())
                .map(|((&y, &t), &s)| {
                    if t.is_nan() || s.is_nan() { f64::NAN } else { y - t - s }
                })
                .collect()
        }
        DecompType::Multiplicative => {
            data.iter()
                .zip(trend.iter())
                .zip(seasonal.iter())
                .map(|((&y, &t), &s)| {
                    if t.is_nan() || s.is_nan() || t == 0.0 || s == 0.0 {
                        f64::NAN
                    } else {
                        y / (t * s)
                    }
                })
                .collect()
        }
    };
    
    Ok(DecompositionResult {
        trend,
        seasonal,
        residual,
        observed: data.to_vec(),
    })
}

/// Calculate trend using centered moving average
fn calculate_trend(data: &[f64], period: usize) -> Vec<f64> {
    if period % 2 == 0 {
        // Even period: use 2x(period) moving average
        let ma1 = sma(data, period);
        // Center it with a 2-period moving average
        let mut trend = vec![f64::NAN; data.len()];
        for i in 1..ma1.len() {
            if !ma1[i].is_nan() && !ma1[i - 1].is_nan() {
                trend[i] = (ma1[i] + ma1[i - 1]) / 2.0;
            }
        }
        trend
    } else {
        // Odd period: simple centered moving average
        sma(data, period)
    }
}

/// Calculate seasonal component
fn calculate_seasonal(detrended: &[f64], period: usize, model: DecompType) -> Vec<f64> {
    let n = detrended.len();
    let mut seasonal_means = vec![0.0; period];
    let mut counts = vec![0; period];
    
    // Calculate average for each season
    for (i, &value) in detrended.iter().enumerate() {
        if !value.is_nan() {
            let season_idx = i % period;
            seasonal_means[season_idx] += value;
            counts[season_idx] += 1;
        }
    }
    
    for i in 0..period {
        if counts[i] > 0 {
            seasonal_means[i] /= counts[i] as f64;
        }
    }
    
    // Center the seasonal component
    match model {
        DecompType::Additive => {
            let mean: f64 = seasonal_means.iter().sum::<f64>() / period as f64;
            for s in &mut seasonal_means {
                *s -= mean;
            }
        }
        DecompType::Multiplicative => {
            let product: f64 = seasonal_means.iter().product();
            let geo_mean = product.powf(1.0 / period as f64);
            if geo_mean != 0.0 {
                for s in &mut seasonal_means {
                    *s /= geo_mean;
                }
            }
        }
    }
    
    // Repeat seasonal pattern for the entire series
    (0..n).map(|i| seasonal_means[i % period]).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_seasonal_decompose_additive() {
        // Create simple seasonal data: trend + seasonal
        let trend_data: Vec<f64> = (0..48).map(|i| i as f64).collect();
        let seasonal_pattern = vec![0.0, 1.0, 2.0, 1.0]; // period = 4
        let data: Vec<f64> = trend_data.iter()
            .enumerate()
            .map(|(i, &t)| t + seasonal_pattern[i % 4])
            .collect();
        
        let result = seasonal_decompose(&data, 4, DecompType::Additive);
        assert!(result.is_ok());
        
        let decomp = result.unwrap();
        assert_eq!(decomp.observed.len(), 48);
        assert_eq!(decomp.trend.len(), 48);
        assert_eq!(decomp.seasonal.len(), 48);
        assert_eq!(decomp.residual.len(), 48);
    }

    #[test]
    fn test_seasonal_decompose_invalid_period() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let result = seasonal_decompose(&data, 20, DecompType::Additive);
        assert!(result.is_err());
    }
}
