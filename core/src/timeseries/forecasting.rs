// Forecasting methods: exponential smoothing, Holt, Holt-Winters

/// Forecast result with confidence intervals
#[derive(Debug, Clone)]
pub struct ForecastResult {
    pub forecasts: Vec<f64>,
    pub lower_bound: Vec<f64>,  // Lower confidence interval
    pub upper_bound: Vec<f64>,  // Upper confidence interval
    pub confidence_level: f64,
}

/// Simple Exponential Smoothing (SES)
/// alpha: smoothing parameter (0 < alpha <= 1)
/// horizon: number of periods to forecast
pub fn simple_exp_smoothing(
    data: &[f64],
    alpha: f64,
    horizon: usize,
    confidence: f64,
) -> Result<ForecastResult, String> {
    if data.is_empty() {
        return Err("Data cannot be empty".to_string());
    }
    
    if alpha <= 0.0 || alpha > 1.0 {
        return Err("Alpha must be in (0, 1]".to_string());
    }
    
    // Fit the model
    let mut level = data[0];
    let mut fitted = vec![level];
    
    for &y in &data[1..] {
        level = alpha * y + (1.0 - alpha) * level;
        fitted.push(level);
    }
    
    // Calculate residuals and standard deviation
    let residuals: Vec<f64> = data.iter()
        .zip(fitted.iter())
        .map(|(&y, &f)| y - f)
        .collect();
    
    let n = residuals.len() as f64;
    let sigma = (residuals.iter().map(|&r| r * r).sum::<f64>() / n).sqrt();
    
    // Generate forecasts (constant for SES)
    let last_level = fitted.last().unwrap();
    let forecasts = vec![*last_level; horizon];
    
    // Calculate confidence intervals
    // For SES, variance grows with horizon: sigma^2 * [1 + sum(alpha^2*(1-alpha)^(2*i))]
    let z = if confidence >= 0.99 { 2.576 }
            else if confidence >= 0.95 { 1.96 }
            else { 1.645 };
    
    let mut lower_bound = Vec::with_capacity(horizon);
    let mut upper_bound = Vec::with_capacity(horizon);
    
    for h in 1..=horizon {
        let mut var_sum = 1.0;
        for j in 1..h {
            var_sum += alpha.powi(2) * (1.0 - alpha).powi(2 * j as i32);
        }
        let forecast_std = sigma * var_sum.sqrt();
        
        lower_bound.push(*last_level - z * forecast_std);
        upper_bound.push(*last_level + z * forecast_std);
    }
    
    Ok(ForecastResult {
        forecasts,
        lower_bound,
        upper_bound,
        confidence_level: confidence,
    })
}

/// Holt's Linear Trend Method
/// alpha: level smoothing parameter
/// beta: trend smoothing parameter
pub fn holt_linear(
    data: &[f64],
    alpha: f64,
    beta: f64,
    horizon: usize,
    confidence: f64,
) -> Result<ForecastResult, String> {
    if data.len() < 2 {
        return Err("Need at least 2 observations".to_string());
    }
    
    if alpha <= 0.0 || alpha > 1.0 || beta <= 0.0 || beta > 1.0 {
        return Err("Alpha and beta must be in (0, 1]".to_string());
    }
    
    // Initialize level and trend
    let mut level = data[0];
    let mut trend = data[1] - data[0];
    let mut fitted = vec![level];
    
    // Fit the model
    for &y in &data[1..] {
        let prev_level = level;
        level = alpha * y + (1.0 - alpha) * (prev_level + trend);
        trend = beta * (level - prev_level) + (1.0 - beta) * trend;
        fitted.push(prev_level + trend);
    }
    
    // Calculate residuals
    let residuals: Vec<f64> = data.iter()
        .zip(fitted.iter())
        .map(|(&y, &f)| y - f)
        .collect();
    
    let n = residuals.len() as f64;
    let sigma = (residuals.iter().map(|&r| r * r).sum::<f64>() / n).sqrt();
    
    // Generate forecasts
    let forecasts: Vec<f64> = (1..=horizon)
        .map(|h| level + h as f64 * trend)
        .collect();
    
    // Confidence intervals
    let z = if confidence >= 0.99 { 2.576 }
            else if confidence >= 0.95 { 1.96 }
            else { 1.645 };
    
    let lower_bound: Vec<f64> = forecasts.iter()
        .enumerate()
        .map(|(i, &f)| f - z * sigma * ((i + 1) as f64).sqrt())
        .collect();
    
    let upper_bound: Vec<f64> = forecasts.iter()
        .enumerate()
        .map(|(i, &f)| f + z * sigma * ((i + 1) as f64).sqrt())
        .collect();
    
    Ok(ForecastResult {
        forecasts,
        lower_bound,
        upper_bound,
        confidence_level: confidence,
    })
}

/// Holt-Winters Seasonal Method
/// alpha: level smoothing
/// beta: trend smoothing
/// gamma: seasonal smoothing
/// period: seasonal period
/// seasonal_type: "add" or "mult"
pub fn holt_winters(
    data: &[f64],
    alpha: f64,
    beta: f64,
    gamma: f64,
    period: usize,
    horizon: usize,
    seasonal_type: &str,
    confidence: f64,
) -> Result<ForecastResult, String> {
    if data.len() < 2 * period {
        return Err("Need at least 2 full seasonal periods".to_string());
    }
    
    if alpha <= 0.0 || alpha > 1.0 || beta <= 0.0 || beta > 1.0 || gamma <= 0.0 || gamma > 1.0 {
        return Err("Alpha, beta, and gamma must be in (0, 1]".to_string());
    }
    
    // Initialize components
    let mut level = data.iter().take(period).sum::<f64>() / period as f64;
    let mut trend = 0.0;
    let mut seasonal = initialize_seasonal(data, period, seasonal_type);
    
    // This is a simplified implementation
    // Full Holt-Winters would require more sophisticated initialization
    
    let mut fitted = Vec::with_capacity(data.len());
    
    for (t, &y) in data.iter().enumerate() {
        let season_idx = t % period;
        
        // Calculate fitted value
        let fitted_value = match seasonal_type {
            "mult" | "multiplicative" => (level + trend) * seasonal[season_idx],
            _ => level + trend + seasonal[season_idx],
        };
        fitted.push(fitted_value);
        
        // Update components
        let prev_level = level;
        
        match seasonal_type {
            "mult" | "multiplicative" => {
                if seasonal[season_idx] != 0.0 {
                    level = alpha * (y / seasonal[season_idx]) + (1.0 - alpha) * (prev_level + trend);
                }
            }
            _ => {
                level = alpha * (y - seasonal[season_idx]) + (1.0 - alpha) * (prev_level + trend);
            }
        }
        
        trend = beta * (level - prev_level) + (1.0 - beta) * trend;
        
        match seasonal_type {
            "mult" | "multiplicative" => {
                if level != 0.0 {
                    seasonal[season_idx] = gamma * (y / level) + (1.0 - gamma) * seasonal[season_idx];
                }
            }
            _ => {
                seasonal[season_idx] = gamma * (y - level) + (1.0 - gamma) * seasonal[season_idx];
            }
        }
    }
    
    // Calculate residuals
    let residuals: Vec<f64> = data.iter()
        .zip(fitted.iter())
        .map(|(&y, &f)| y - f)
        .collect();
    
    let n = residuals.len() as f64;
    let sigma = (residuals.iter().map(|&r| r * r).sum::<f64>() / n).sqrt();
    
    // Generate forecasts
    let forecasts: Vec<f64> = (1..=horizon)
        .map(|h| {
            let season_idx = (data.len() + h - 1) % period;
            match seasonal_type {
                "mult" | "multiplicative" => (level + h as f64 * trend) * seasonal[season_idx],
                _ => level + h as f64 * trend + seasonal[season_idx],
            }
        })
        .collect();
    
    // Confidence intervals
    let z = if confidence >= 0.99 { 2.576 }
            else if confidence >= 0.95 { 1.96 }
            else { 1.645 };
    
    let lower_bound: Vec<f64> = forecasts.iter()
        .enumerate()
        .map(|(i, &f)| f - z * sigma * ((i + 1) as f64).sqrt())
        .collect();
    
    let upper_bound: Vec<f64> = forecasts.iter()
        .enumerate()
        .map(|(i, &f)| f + z * sigma * ((i + 1) as f64).sqrt())
        .collect();
    
    Ok(ForecastResult {
        forecasts,
        lower_bound,
        upper_bound,
        confidence_level: confidence,
    })
}

/// Initialize seasonal components
fn initialize_seasonal(data: &[f64], period: usize, seasonal_type: &str) -> Vec<f64> {
    let mut seasonal = vec![0.0; period];
    let n_periods = data.len() / period;
    
    for s in 0..period {
        let mut sum = 0.0;
        for p in 0..n_periods {
            sum += data[p * period + s];
        }
        seasonal[s] = sum / n_periods as f64;
    }
    
    // Center the seasonal components
    let mean = seasonal.iter().sum::<f64>() / period as f64;
    
    match seasonal_type {
        "mult" | "multiplicative" => {
            if mean != 0.0 {
                for s in &mut seasonal {
                    *s /= mean;
                }
            }
        }
        _ => {
            for s in &mut seasonal {
                *s -= mean;
            }
        }
    }
    
    seasonal
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_exp_smoothing() {
        let data: Vec<f64> = vec![10.0, 12.0, 11.0, 13.0, 12.0, 14.0];
        let result = simple_exp_smoothing(&data, 0.3, 3, 0.95);
        assert!(result.is_ok());
        
        let forecast = result.unwrap();
        assert_eq!(forecast.forecasts.len(), 3);
        assert_eq!(forecast.lower_bound.len(), 3);
        assert_eq!(forecast.upper_bound.len(), 3);
        
        // Check confidence intervals make sense
        assert!(forecast.lower_bound[0] < forecast.forecasts[0]);
        assert!(forecast.upper_bound[0] > forecast.forecasts[0]);
    }

    #[test]
    fn test_holt_linear() {
        let data: Vec<f64> = (1..=20).map(|i| i as f64 * 2.0).collect();
        let result = holt_linear(&data, 0.3, 0.1, 5, 0.95);
        assert!(result.is_ok());
        
        let forecast = result.unwrap();
        assert_eq!(forecast.forecasts.len(), 5);
        
        // Forecasts should show increasing trend
        assert!(forecast.forecasts[1] > forecast.forecasts[0]);
    }

    #[test]
    fn test_holt_winters() {
        // Create seasonal data
        let seasonal_pattern = vec![100.0, 120.0, 110.0, 130.0];
        let data: Vec<f64> = (0..20)
            .map(|i| seasonal_pattern[i % 4] + (i / 4) as f64 * 5.0)
            .collect();
        
        let result = holt_winters(&data, 0.3, 0.1, 0.2, 4, 4, "add", 0.95);
        assert!(result.is_ok());
        
        let forecast = result.unwrap();
        assert_eq!(forecast.forecasts.len(), 4);
    }
}
