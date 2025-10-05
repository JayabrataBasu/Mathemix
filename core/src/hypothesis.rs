// Phase 5: Hypothesis Testing (Basic implementation)
use polars::prelude::*;
use serde::{Deserialize, Serialize};

use crate::errors::{MatheMixxError, Result};
use crate::DataSet;

/// T-test result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TTestResult {
    pub test_type: String,
    pub statistic: f64,
    pub degrees_of_freedom: f64,
    pub p_value: f64,
    pub mean1: f64,
    pub mean2: Option<f64>,
    pub confidence_interval: (f64, f64),
    pub significant: bool,
}

/// Chi-square test result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChiSquareResult {
    pub statistic: f64,
    pub degrees_of_freedom: usize,
    pub p_value: f64,
    pub significant: bool,
}

/// ANOVA result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnovaResult {
    pub f_statistic: f64,
    pub df_between: usize,
    pub df_within: usize,
    pub p_value: f64,
    pub significant: bool,
    pub group_means: Vec<(String, f64)>,
}

impl DataSet {
    /// One-sample t-test
    pub fn t_test_one_sample(
        &self,
        column: &str,
        population_mean: f64,
        alpha: f64,
    ) -> Result<TTestResult> {
        let series = self
            .dataframe()
            .column(column)?
            .as_series()
            .ok_or_else(|| MatheMixxError::UnsupportedOperation("Column is not a series".into()))?
            .cast(&DataType::Float64)?;
        let ca = series.f64()?;

        let mut values = Vec::new();
        for opt_val in ca.into_iter() {
            if let Some(val) = opt_val {
                values.push(val);
            }
        }

        let n = values.len();
        if n < 2 {
            return Err(MatheMixxError::InvalidInput(
                "Need at least 2 observations for t-test".into(),
            ));
        }

        let sample_mean = values.iter().sum::<f64>() / n as f64;
        let sample_var = values
            .iter()
            .map(|x| (x - sample_mean).powi(2))
            .sum::<f64>()
            / (n - 1) as f64;
        let sample_std = sample_var.sqrt();
        let se = sample_std / (n as f64).sqrt();

        let t_stat = (sample_mean - population_mean) / se;
        let df = (n - 1) as f64;
        let p_value = self.t_distribution_two_tailed(t_stat.abs(), df);

        let t_critical = self.t_critical_value(alpha / 2.0, df);
        let margin = t_critical * se;
        let ci = (sample_mean - margin, sample_mean + margin);

        Ok(TTestResult {
            test_type: "One-sample t-test".to_string(),
            statistic: t_stat,
            degrees_of_freedom: df,
            p_value,
            mean1: sample_mean,
            mean2: None,
            confidence_interval: ci,
            significant: p_value < alpha,
        })
    }

    /// Two-sample independent t-test
    pub fn t_test_two_sample(
        &self,
        col1: &str,
        col2: &str,
        alpha: f64,
        equal_var: bool,
    ) -> Result<TTestResult> {
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

        let mut values1 = Vec::new();
        let mut values2 = Vec::new();

        for opt_val in s1.f64()?.into_iter() {
            if let Some(val) = opt_val {
                values1.push(val);
            }
        }

        for opt_val in s2.f64()?.into_iter() {
            if let Some(val) = opt_val {
                values2.push(val);
            }
        }

        let n1 = values1.len();
        let n2 = values2.len();

        if n1 < 2 || n2 < 2 {
            return Err(MatheMixxError::InvalidInput(
                "Need at least 2 observations in each group".into(),
            ));
        }

        let mean1 = values1.iter().sum::<f64>() / n1 as f64;
        let mean2 = values2.iter().sum::<f64>() / n2 as f64;

        let var1 = values1.iter().map(|x| (x - mean1).powi(2)).sum::<f64>() / (n1 - 1) as f64;
        let var2 = values2.iter().map(|x| (x - mean2).powi(2)).sum::<f64>() / (n2 - 1) as f64;

        let (t_stat, df, se) = if equal_var {
            // Pooled variance
            let pooled_var =
                ((n1 - 1) as f64 * var1 + (n2 - 1) as f64 * var2) / (n1 + n2 - 2) as f64;
            let se = (pooled_var * (1.0 / n1 as f64 + 1.0 / n2 as f64)).sqrt();
            let df = (n1 + n2 - 2) as f64;
            let t = (mean1 - mean2) / se;
            (t, df, se)
        } else {
            // Welch's t-test
            let se = (var1 / n1 as f64 + var2 / n2 as f64).sqrt();
            let df = (var1 / n1 as f64 + var2 / n2 as f64).powi(2)
                / ((var1 / n1 as f64).powi(2) / (n1 - 1) as f64
                    + (var2 / n2 as f64).powi(2) / (n2 - 1) as f64);
            let t = (mean1 - mean2) / se;
            (t, df, se)
        };

        let p_value = self.t_distribution_two_tailed(t_stat.abs(), df);
        let t_critical = self.t_critical_value(alpha / 2.0, df);
        let margin = t_critical * se;
        let diff = mean1 - mean2;
        let ci = (diff - margin, diff + margin);

        Ok(TTestResult {
            test_type: if equal_var {
                "Two-sample t-test"
            } else {
                "Welch's t-test"
            }
            .to_string(),
            statistic: t_stat,
            degrees_of_freedom: df,
            p_value,
            mean1,
            mean2: Some(mean2),
            confidence_interval: ci,
            significant: p_value < alpha,
        })
    }

    /// Paired t-test
    pub fn t_test_paired(&self, col1: &str, col2: &str, alpha: f64) -> Result<TTestResult> {
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

        let mut differences = Vec::new();

        for (opt_v1, opt_v2) in s1.f64()?.into_iter().zip(s2.f64()?.into_iter()) {
            if let (Some(v1), Some(v2)) = (opt_v1, opt_v2) {
                differences.push(v1 - v2);
            }
        }

        let n = differences.len();
        if n < 2 {
            return Err(MatheMixxError::InvalidInput(
                "Need at least 2 paired observations".into(),
            ));
        }

        let mean_diff = differences.iter().sum::<f64>() / n as f64;
        let var_diff = differences
            .iter()
            .map(|x| (x - mean_diff).powi(2))
            .sum::<f64>()
            / (n - 1) as f64;
        let se = var_diff.sqrt() / (n as f64).sqrt();

        let t_stat = mean_diff / se;
        let df = (n - 1) as f64;
        let p_value = self.t_distribution_two_tailed(t_stat.abs(), df);

        let t_critical = self.t_critical_value(alpha / 2.0, df);
        let margin = t_critical * se;
        let ci = (mean_diff - margin, mean_diff + margin);

        Ok(TTestResult {
            test_type: "Paired t-test".to_string(),
            statistic: t_stat,
            degrees_of_freedom: df,
            p_value,
            mean1: mean_diff,
            mean2: None,
            confidence_interval: ci,
            significant: p_value < alpha,
        })
    }

    // Helper functions for statistical distributions (approximations)

    /// Two-tailed p-value for t-distribution (approximation)
    fn t_distribution_two_tailed(&self, t: f64, df: f64) -> f64 {
        // Approximate using normal distribution for large df
        if df > 30.0 {
            2.0 * self.normal_cdf(-t.abs())
        } else {
            // For small df, rough approximation
            // This should be replaced with proper t-distribution CDF
            let p = 1.0 / (1.0 + 0.05 * t * t / df);
            p.min(1.0).max(0.0)
        }
    }

    /// Critical value for t-distribution (approximation)
    fn t_critical_value(&self, alpha: f64, df: f64) -> f64 {
        // Rough approximation - should use proper t-distribution quantile
        if df > 30.0 {
            self.normal_quantile(1.0 - alpha)
        } else {
            // Approximate t critical value
            1.96 + 0.5 / df.sqrt()
        }
    }

    /// Normal CDF (standard normal)
    fn normal_cdf(&self, x: f64) -> f64 {
        0.5 * (1.0 + self.erf(x / std::f64::consts::SQRT_2))
    }

    /// Normal quantile (inverse CDF) - approximation
    fn normal_quantile(&self, p: f64) -> f64 {
        if p <= 0.0 {
            return f64::NEG_INFINITY;
        }
        if p >= 1.0 {
            return f64::INFINITY;
        }

        // Simplified approximation for common values
        if (p - 0.975).abs() < 0.001 {
            return 1.96;
        }
        if (p - 0.95).abs() < 0.001 {
            return 1.645;
        }
        if (p - 0.99).abs() < 0.001 {
            return 2.576;
        }

        // Very rough linear approximation
        1.96 // Default to 95% CI value
    }

    /// Error function (erf) - approximation
    fn erf(&self, x: f64) -> f64 {
        // Abramowitz and Stegun approximation
        let a1 = 0.254829592;
        let a2 = -0.284496736;
        let a3 = 1.421413741;
        let a4 = -1.453152027;
        let a5 = 1.061405429;
        let p = 0.3275911;

        let sign = if x < 0.0 { -1.0 } else { 1.0 };
        let x = x.abs();

        let t = 1.0 / (1.0 + p * x);
        let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();

        sign * y
    }
}
