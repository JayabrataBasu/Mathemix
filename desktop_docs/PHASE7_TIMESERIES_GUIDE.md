# Phase 7: Time Series Analysis - User Guide

## Overview

Phase 7 introduces comprehensive time series analysis capabilities to Mathemix, enabling you to analyze temporal data, test for stationarity, decompose series, and generate forecasts. This guide covers all features available in the desktop UI.

## Table of Contents

1. [Time Series Operations](#time-series-operations)
2. [Autocorrelation Analysis](#autocorrelation-analysis)
3. [Stationarity Testing](#stationarity-testing)
4. [Seasonal Decomposition](#seasonal-decomposition)
5. [Forecasting Methods](#forecasting-methods)
6. [Practical Examples](#practical-examples)
7. [Interpretation Guide](#interpretation-guide)

---

## Time Series Operations

### Basic Transformations

#### Lag Operator
Shifts the series backward by a specified number of periods.

**Use Cases:**
- Creating lagged predictors for regression
- Comparing current values with past values
- Feature engineering for machine learning

**Parameters:**
- **Lag Periods**: Number of periods to shift (default: 1)

**Example:** Monthly sales data lagged by 1 month to predict current sales from previous month.

---

#### Differencing
Computes the difference between consecutive observations.

**Use Cases:**
- Removing trends to achieve stationarity
- Analyzing period-to-period changes
- Preparing data for ARIMA modeling

**Parameters:**
- **Order**: Number of times to difference (1 = first difference, 2 = second difference)

**Example:** Stock prices → Daily returns (first difference)

---

### Moving Averages

#### Simple Moving Average (SMA)
Arithmetic mean of the last N observations.

**Use Cases:**
- Smoothing noisy data
- Identifying trends
- Technical analysis

**Parameters:**
- **Window Size**: Number of periods to average (e.g., 7 for weekly, 30 for monthly)

**Formula:** SMA_t = (x_t + x_{t-1} + ... + x_{t-n+1}) / n

---

#### Exponential Moving Average (EMA)
Weighted average giving more importance to recent observations.

**Use Cases:**
- Trend following with faster response to changes
- Signal smoothing with less lag than SMA
- Technical trading indicators

**Parameters:**
- **Span**: Specifies decay rate (higher = slower decay)

**Formula:** EMA_t = α × x_t + (1-α) × EMA_{t-1}

---

#### Weighted Moving Average (WMA)
Linear weights decreasing for older observations.

**Use Cases:**
- When recent data is more relevant
- Balancing between SMA and EMA
- Custom weighting schemes

**Parameters:**
- **Window Size**: Number of periods

---

### Rolling Statistics

Apply statistical functions over a sliding window.

**Available Functions:**
- **Rolling Mean**: Average over window
- **Rolling Std**: Standard deviation over window
- **Rolling Min/Max**: Minimum/maximum over window

**Use Cases:**
- Detecting outliers (values beyond rolling mean ± 2×std)
- Monitoring process variability
- Creating control charts

**Parameters:**
- **Window Size**: Number of periods in rolling window

---

## Autocorrelation Analysis

### Autocorrelation Function (ACF)

Measures linear relationship between observations at different lags.

**Use Cases:**
- Detecting seasonality patterns
- Identifying MA order for ARIMA
- Testing for randomness

**How to Read ACF Plot:**
- **Lag 0**: Always 1.0 (perfect correlation with itself)
- **Significant Spikes**: Values outside confidence bands indicate significant autocorrelation
- **Slow Decay**: Suggests non-stationarity
- **Sharp Cutoff**: Indicates MA(q) process

**Example Patterns:**
- **White Noise**: All lags ≈ 0 (within confidence bands)
- **AR(1)**: Exponential decay
- **Seasonal**: Spikes at seasonal lags (12, 24, 36 for monthly data)

---

### Partial Autocorrelation Function (PACF)

Correlation between observations at lag k, controlling for shorter lags.

**Use Cases:**
- Identifying AR order for ARIMA
- Understanding direct relationships between lags
- Model specification

**How to Read PACF Plot:**
- **Sharp Cutoff at Lag p**: Suggests AR(p) model
- **Gradual Decay**: Suggests MA component needed

**Example Patterns:**
- **AR(1)**: Spike at lag 1, then ≈ 0
- **AR(2)**: Spikes at lags 1-2, then ≈ 0

---

### Ljung-Box Test

Statistical test for autocorrelation in residuals.

**Use Cases:**
- Model diagnostics (residuals should be uncorrelated)
- Validating forecast model assumptions
- Testing for randomness

**Interpretation:**
- **Null Hypothesis**: No autocorrelation
- **p-value > 0.05**: Cannot reject H₀ (good - no autocorrelation)
- **p-value < 0.05**: Reject H₀ (autocorrelation present)

**Parameters:**
- **Lags**: Number of lags to test (default: 20)

---

## Stationarity Testing

### Augmented Dickey-Fuller (ADF) Test

Tests for presence of unit root (non-stationarity).

**Use Cases:**
- Checking if differencing is needed
- Validating ARIMA assumptions
- Time series preprocessing

**Interpretation:**
- **Null Hypothesis**: Series has unit root (non-stationary)
- **p-value < 0.05**: Reject H₀ → Series is **stationary**
- **p-value > 0.05**: Cannot reject H₀ → Series is **non-stationary**

**Test Statistic:**
- More negative = stronger evidence of stationarity
- Compare to critical values (-3.43 @ 1%, -2.86 @ 5%, -2.57 @ 10%)

**What to Do:**
- **Non-stationary**: Apply differencing or detrending
- **Stationary**: Proceed with modeling

---

### KPSS Test

Tests for stationarity around a deterministic trend.

**Use Cases:**
- Confirming ADF results
- Detecting trend stationarity
- Complementary stationarity check

**Interpretation:**
- **Null Hypothesis**: Series is stationary
- **p-value > 0.05**: Cannot reject H₀ → Series is **stationary**
- **p-value < 0.05**: Reject H₀ → Series is **non-stationary**

**Key Difference from ADF:**
- KPSS: H₀ = stationary (opposite of ADF)
- Use both tests together for robustness

**Decision Matrix:**
| ADF Result | KPSS Result | Conclusion |
|------------|-------------|------------|
| Stationary | Stationary | **Stationary** ✓ |
| Non-stationary | Non-stationary | **Non-stationary** - difference data |
| Stationary | Non-stationary | Trend stationary - detrend |
| Non-stationary | Stationary | Difference stationary - difference data |

---

## Seasonal Decomposition

Separates time series into trend, seasonal, and residual components.

### Additive Decomposition

**Model:** Y_t = T_t + S_t + R_t

**When to Use:**
- Seasonal variations are roughly constant over time
- Absolute magnitude of seasonality doesn't change with level

**Example:** Temperature data (seasonal variation ≈ constant)

---

### Multiplicative Decomposition

**Model:** Y_t = T_t × S_t × R_t

**When to Use:**
- Seasonal variations proportional to series level
- Percentage change in seasonality matters

**Example:** Retail sales (holiday peaks grow with business)

---

### Interpretation

**Trend Component:**
- Long-term direction (increasing, decreasing, stable)
- Smooth line representing underlying pattern

**Seasonal Component:**
- Repeating patterns (daily, weekly, monthly, quarterly)
- Fixed period (e.g., 12 for monthly data, 7 for daily)

**Residual Component:**
- Irregular fluctuations not explained by trend/seasonal
- Should resemble white noise if model is good
- Large residuals indicate model misspecification or outliers

**Parameters:**
- **Period**: Seasonal cycle length (12 = monthly, 4 = quarterly, 7 = daily)
- **Model**: 'additive' or 'multiplicative'

---

## Forecasting Methods

### Simple Exponential Smoothing (SES)

Best for series with **no trend or seasonality**.

**Use Cases:**
- Short-term forecasts of stable processes
- Inventory management for steady demand
- Level-only series

**Model:** 
- Forecast_t+h = L_t (constant for all horizons)
- L_t = α × Y_t + (1-α) × L_{t-1}

**Parameters:**
- **Alpha (α)**: Smoothing parameter (0-1)
  - α ≈ 0: More weight to history (smooth, slow to react)
  - α ≈ 1: More weight to recent (responsive, volatile)
  - Default: 0.3 (balanced)
- **Horizon**: Number of periods to forecast
- **Confidence Level**: For prediction intervals (default: 0.95)

**When to Use:**
- Mature products with stable demand
- When you only need short-term forecasts
- No clear trend or seasonality

---

### Holt's Linear Trend

For series with **trend but no seasonality**.

**Use Cases:**
- Sales growing/declining linearly
- Population projections
- Technology adoption curves

**Model:**
- Forecast_t+h = L_t + h × B_t
- L_t = α × Y_t + (1-α) × (L_{t-1} + B_{t-1})
- B_t = β × (L_t - L_{t-1}) + (1-β) × B_{t-1}

**Parameters:**
- **Alpha (α)**: Level smoothing (0-1, default: 0.3)
- **Beta (β)**: Trend smoothing (0-1, default: 0.1)
  - β ≈ 0: Trend changes slowly
  - β ≈ 1: Trend adapts quickly
- **Horizon**: Forecast periods
- **Confidence Level**: Default 0.95

**When to Use:**
- Clear upward or downward trend
- No seasonal patterns
- Linear growth/decline expected to continue

---

### Holt-Winters Method

For series with **trend AND seasonality**.

**Use Cases:**
- Retail sales (seasonal + growth)
- Energy demand (seasonal + trend)
- Tourism data (seasonal peaks + growth)

**Model:**
- Forecast_t+h = (L_t + h × B_t) × S_{t+h-m} (multiplicative)
- L_t = α × (Y_t / S_{t-m}) + (1-α) × (L_{t-1} + B_{t-1})
- B_t = β × (L_t - L_{t-1}) + (1-β) × B_{t-1}
- S_t = γ × (Y_t / L_t) + (1-γ) × S_{t-m}

**Parameters:**
- **Alpha (α)**: Level smoothing (default: 0.3)
- **Beta (β)**: Trend smoothing (default: 0.1)
- **Gamma (γ)**: Seasonal smoothing (0-1, default: 0.2)
  - γ ≈ 0: Stable seasonal pattern
  - γ ≈ 1: Seasonal pattern changes quickly
- **Period**: Seasonal cycle (12 = monthly, 4 = quarterly)
- **Horizon**: Forecast periods
- **Confidence Level**: Default 0.95

**When to Use:**
- Both trend and seasonality present
- Long-term forecasts needed
- Seasonal patterns expected to continue

---

### Choosing a Forecasting Method

| Characteristic | Method | Example |
|----------------|--------|---------|
| No trend, no seasonality | **SES** | Stable inventory demand |
| Trend, no seasonality | **Holt** | Linear sales growth |
| Trend + seasonality | **Holt-Winters** | Retail sales with holiday peaks |

---

## Practical Examples

### Example 1: Monthly Sales Analysis

**Scenario:** Analyze 3 years of monthly sales data.

**Steps:**
1. **Load Data**: Import CSV with date and sales columns
2. **Visualize**: Plot raw data to identify patterns
3. **Stationarity**: Run ADF and KPSS tests
   - If non-stationary → Apply first difference
4. **ACF/PACF**: Plot to identify potential ARIMA orders
5. **Decomposition**: Period = 12, model = multiplicative
   - Check if seasonal peaks grow with trend
6. **Forecast**: Use Holt-Winters (α=0.3, β=0.1, γ=0.2, period=12, horizon=6)
7. **Validate**: Check residuals with Ljung-Box test

**Expected Results:**
- Seasonal pattern repeating every 12 months
- Upward trend if business growing
- Forecast with confidence intervals

---

### Example 2: Stock Price Analysis

**Scenario:** Analyze daily stock prices.

**Steps:**
1. **Transform**: Calculate daily returns (first difference)
2. **Stationarity**: Test returns (should be stationary)
3. **ACF/PACF**: Check for autocorrelation in returns
4. **Volatility**: Rolling std (window=20) for volatility clustering
5. **Forecast**: Use SES for short-term price level (α=0.5)

**Notes:**
- Prices are typically non-stationary (random walk)
- Returns are usually stationary
- High α for stocks (markets change quickly)

---

### Example 3: Website Traffic Forecasting

**Scenario:** Daily website visitors with weekly seasonality.

**Steps:**
1. **Decomposition**: Period = 7, model = additive
   - Identify day-of-week effects
2. **Stationarity**: Check with ADF/KPSS
3. **Forecast**: Holt-Winters (period=7, horizon=14)
   - α = 0.4 (responsive to changes)
   - β = 0.1 (slow trend adaptation)
   - γ = 0.3 (moderate seasonal updates)

---

## Interpretation Guide

### Reading ACF/PACF Plots

**Confidence Bands (Dashed Red Lines):**
- 95% significance level
- Values outside bands are statistically significant

**Common Patterns:**

| ACF Pattern | PACF Pattern | Suggested Model |
|-------------|--------------|-----------------|
| Exponential decay | Spike at lag 1 | **AR(1)** |
| Spike at lag 1 | Exponential decay | **MA(1)** |
| Decay after lag 1 | Spikes at lags 1-2 | **AR(2)** |
| Spikes at lags 12, 24 | Spike at lag 12 | **Seasonal** (period=12) |

---

### Decomposition Plot Interpretation

**4-Panel Layout:**
1. **Observed**: Original data
2. **Trend**: Long-term movement
3. **Seasonal**: Repeating pattern
4. **Residual**: Unexplained variation

**Good Decomposition:**
- Residuals resemble white noise (no pattern)
- Trend is smooth
- Seasonal component is regular

**Poor Decomposition:**
- Residuals show patterns → Wrong period or model type
- Trend is jagged → May need smoothing or different method

---

### Forecast Plot Interpretation

**Components:**
- **Blue Line**: Historical data
- **Orange Dashed Line**: Point forecasts
- **Shaded Area**: Confidence interval (typically 95%)

**What to Check:**
- Forecasts follow expected pattern (trend/seasonality)
- Confidence intervals widen with horizon (uncertainty increases)
- Forecasts are reasonable given historical data

**Warning Signs:**
- Forecasts diverge drastically from history
- Very wide confidence intervals (poor model fit)
- Forecasts ignore obvious patterns

---

## Tips and Best Practices

### Data Preparation
1. **Handle Missing Values**: Interpolate or forward-fill before analysis
2. **Outliers**: Consider removing or transforming extreme values
3. **Sufficient Data**: Need at least 2 full seasonal cycles for decomposition

### Model Selection
1. **Start Simple**: Try SES before Holt-Winters
2. **Test Stationarity**: Always check before ARIMA-type modeling
3. **Validate**: Use holdout data to test forecast accuracy

### Parameter Tuning
1. **Alpha**: Start with 0.3, increase for volatile data
2. **Beta/Gamma**: Keep low (0.1-0.2) initially
3. **Period**: Must match actual seasonal cycle exactly

### Interpretation
1. **Multiple Tests**: Use both ADF and KPSS for stationarity
2. **Residual Checks**: Always examine residuals
3. **Domain Knowledge**: Statistics + business context = best forecasts

---

## Troubleshooting

### "Series is non-stationary"
**Solution:** Apply differencing (order=1) or detrending

### "Not enough data for decomposition"
**Solution:** Need at least 2×period observations (e.g., 24+ for period=12)

### "Forecasts are unrealistic"
**Solution:** 
- Check parameter values (α, β, γ)
- Try different model (SES vs Holt vs Holt-Winters)
- Verify data quality and period setting

### "Wide confidence intervals"
**Solution:**
- More historical data improves estimates
- Lower confidence level (e.g., 0.80 instead of 0.95)
- Consider model with better fit

---

## Keyboard Shortcuts

| Action | Shortcut |
|--------|----------|
| Run Analysis | `Ctrl+Enter` |
| Clear Results | `Ctrl+Shift+C` |
| Export Plot | `Ctrl+S` |
| New Analysis | `Ctrl+N` |

---

## Additional Resources

- **ACF/PACF Reference**: Box & Jenkins (1976) - Time Series Analysis
- **Stationarity**: Dickey & Fuller (1979) - Unit Root Tests
- **Forecasting**: Hyndman & Athanasopoulos - "Forecasting: Principles and Practice"

---

**Last Updated:** October 2025  
**Version:** 1.0  
**Phase:** 7 - Time Series Analysis
