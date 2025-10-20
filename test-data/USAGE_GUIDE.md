# Test Data Usage Guide

## Quick Start

All test datasets are located in the `test-data/` folder. Each file is designed to test specific features of Mathemix.

---

## 📊 Regression Testing

### 1. **regression_simple.csv**
**Use for:** Basic linear regression
- **Columns:** `study_hours`, `test_score`
- **Rows:** 100
- **Relationship:** Linear (y = 2.5x + 5 + noise)
- **How to use:**
  1. Open in Mathemix Desktop
  2. Set Dependent (Y): `test_score`
  3. Select Independent (X): `study_hours`
  4. Click "Run Regression"
  5. Expected: R² ≈ 0.95, positive coefficient for study_hours

### 2. **regression_multiple.csv**
**Use for:** Multiple linear regression
- **Columns:** `age`, `income`, `credit_score`, `house_price`
- **Rows:** 200
- **Relationships:** 
  - Income: Strong positive effect on price
  - Credit score: Positive effect on price
  - Age: Negative effect on price
- **How to use:**
  1. Set Dependent: `house_price`
  2. Select all 3 independents
  3. Click "Run Regression"
  4. Check diagnostic plots for model fit

### 3. **regression_housing.csv**
**Use for:** Complex regression analysis
- **Columns:** `square_feet`, `bedrooms`, `bathrooms`, `age_years`, `distance_to_city_km`, `price`
- **Rows:** 300
- **Ideal for:** Testing all Phase 6 diagnostic plots
- **How to use:**
  1. Set Dependent: `price`
  2. Select predictors: `square_feet`, `bedrooms`, `bathrooms`
  3. Run regression
  4. Click "📊 Diagnostic Plots" to see 6-panel diagnostics

---

## 📈 Visualization/Plotting Testing

### 4. **plots_distributions.csv**
**Use for:** Distribution plots (histogram, box plot, violin plot)
- **Columns:** 
  - `normal_dist` - Normal distribution (mean=100, sd=15)
  - `skewed_dist` - Right-skewed exponential distribution
  - `uniform_dist` - Uniform distribution (0-100)
  - `bimodal_dist` - Two-peaked distribution
  - `category` - Categorical variable (A, B, C, D)
  - `score` - Integer scores (0-100)
- **Rows:** 500
- **How to use:**
  1. Load the file
  2. Select any numeric column
  3. Try each plot type:
     - "Histogram + KDE" → Shows distribution with density curve
     - "Box Plot" → Shows quartiles and outliers
     - "Violin Plot" → Combines box plot with distribution

### 5. **plots_correlation.csv**
**Use for:** Correlation heatmap
- **Columns:** `variable_1` through `variable_5`
- **Rows:** 200
- **Correlations:**
  - var1 ↔ var2: Strong positive (r ≈ 0.8)
  - var1 ↔ var3: Moderate negative (r ≈ -0.6)
  - var2 ↔ var5: Moderate positive (r ≈ 0.5)
  - var4: Independent (no correlation)
- **How to use:**
  1. Load the file
  2. Click "Correlation Heatmap"
  3. Observe color-coded correlation matrix

---

## ⏱️ Time Series Analysis Testing

### 6. **timeseries_monthly_sales.csv**
**Best for:** Learning all time series features
- **Columns:** `date`, `sales`
- **Period:** 48 months (4 years)
- **Pattern:** Linear trend + annual seasonality
- **Ideal tests:**
  - **ACF/PACF:** Should show seasonal spikes at lag 12
  - **Stationarity:** Non-stationary (has trend)
  - **Decomposition:** period=12, model='additive'
  - **Forecasting:** Holt-Winters (α=0.3, β=0.1, γ=0.2, period=12, horizon=12)

**How to use:**
```
1. Load file → Time Series tab
2. Select column: sales
3. Plot ACF+PACF (nlags=24) → See seasonal pattern
4. Run both stationarity tests → Expect non-stationary
5. Decompose (period=12, additive) → Clear trend+seasonal
6. Forecast with Holt-Winters (12 months ahead)
```

### 7. **timeseries_daily_temperature.csv**
**Best for:** Strong seasonality testing
- **Columns:** `date`, `temperature_celsius`
- **Period:** 730 days (2 years)
- **Pattern:** Strong annual cycle (sine wave)
- **Ideal tests:**
  - **ACF:** Strong seasonality visible
  - **Decomposition:** period=365, model='additive'
  - **Forecasting:** Holt-Winters for seasonal forecast

### 8. **timeseries_stock_prices.csv**
**Best for:** Random walk / non-stationary data
- **Columns:** `date`, `close_price`, `volume`
- **Period:** 252 trading days (1 year)
- **Pattern:** Random walk with small upward drift
- **Ideal tests:**
  - **Stationarity:** ADF should show non-stationary
  - **First Difference:** Apply diff(1) to make stationary
  - **ACF on Returns:** Should show white noise
  - **Forecasting:** Use SES or Holt (no seasonality)

**How to use:**
```
1. Select column: close_price
2. ADF test → Non-stationary (p > 0.05)
3. Apply difference (order=1)
4. Test differenced data → Should be stationary
5. Forecast with Holt Linear (trend model)
```

### 9. **timeseries_website_traffic.csv**
**Best for:** Weekly seasonality
- **Columns:** `date`, `visitors`
- **Period:** 365 days
- **Pattern:** Weekly cycle (weekdays higher than weekends)
- **Ideal tests:**
  - **ACF:** Spikes at lags 7, 14, 21 (weekly pattern)
  - **Decomposition:** period=7, model='additive'
  - **Forecasting:** Holt-Winters with period=7

### 10. **timeseries_quarterly_revenue.csv**
**Best for:** Quarterly patterns
- **Columns:** `quarter`, `revenue`
- **Period:** 20 quarters (5 years)
- **Pattern:** Q4 highest (holiday season), Q1 lowest
- **Ideal tests:**
  - **Decomposition:** period=4, model='multiplicative'
  - **Forecasting:** Holt-Winters (period=4, horizon=4)
  - **ACF:** Seasonal spikes at lags 4, 8, 12

---

## 🎯 Testing Workflows

### Complete Regression Workflow
```
1. Open regression_housing.csv
2. Summarize → Check data statistics
3. Set Y = price, X = square_feet, bedrooms, bathrooms
4. Run Regression → Check coefficients, R²
5. Diagnostic Plots → Verify assumptions
6. Save plot → Export as PNG
```

### Complete Time Series Workflow
```
1. Open timeseries_monthly_sales.csv
2. Time Series tab → Select 'sales'
3. Plot ACF+PACF (nlags=24)
4. Run both stationarity tests
5. Decompose (period=12)
6. Forecast Holt-Winters (12 months)
7. Interpret results
```

### Complete Visualization Workflow
```
1. Open plots_distributions.csv
2. Select 'normal_dist'
3. Histogram + KDE → See distribution
4. Box Plot → Check for outliers
5. Switch to plots_correlation.csv
6. Correlation Heatmap → See relationships
7. Export plots
```

---

## 💡 Tips

### Expected Results

**Regression (regression_simple.csv):**
- Coefficient for study_hours: ~2.5
- Intercept: ~5.0
- R²: >0.90
- Residuals: Normally distributed

**Time Series (timeseries_monthly_sales.csv):**
- ADF test: Non-stationary (p > 0.05)
- Decomposition: Clear upward trend + seasonal peaks every 12 months
- Forecast: Should continue trend with seasonal variation

**Plots (plots_distributions.csv):**
- normal_dist: Bell-shaped histogram
- skewed_dist: Right-skewed with long tail
- bimodal_dist: Two distinct peaks

### Troubleshooting

**"Not enough data for decomposition"**
→ For monthly data, use period=12; for quarterly, use period=4; for weekly, use period=7

**"Regression failed"**
→ Make sure you selected at least one independent variable

**"Non-stationary series"**
→ This is expected for most raw time series. Apply differencing or use appropriate forecast method.

---

## 📝 Quick Reference

| File | Feature | Key Column(s) | Best Use |
|------|---------|---------------|----------|
| regression_simple | OLS | test_score | Learning regression |
| regression_multiple | OLS | house_price | Multiple predictors |
| regression_housing | OLS + Diagnostics | price | Full diagnostic suite |
| plots_distributions | Histograms, Box plots | normal_dist, etc. | Distribution visualization |
| plots_correlation | Heatmap | all variables | Correlation analysis |
| timeseries_monthly_sales | All TS features | sales | Complete TS workflow |
| timeseries_daily_temperature | Seasonality | temperature_celsius | Strong seasonal pattern |
| timeseries_stock_prices | Random walk | close_price | Non-stationary testing |
| timeseries_website_traffic | Weekly pattern | visitors | Short-term seasonality |
| timeseries_quarterly_revenue | Quarterly pattern | revenue | Quarterly analysis |

---

**Generated:** October 2025  
**Version:** 1.0  
**Location:** `test-data/USAGE_GUIDE.md`
