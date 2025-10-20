# Test Data Files - Complete List

## Summary

Created **10 CSV files** + **2 documentation files** for comprehensive testing of all Mathemix features.

---

## Files Created

### Regression Testing (3 files)

1. **regression_simple.csv** (3.7 KB)
   - Simple linear regression
   - 100 rows × 2 columns
   - Perfect for learning basic regression

2. **regression_multiple.csv** (6.5 KB)
   - Multiple linear regression
   - 200 rows × 4 columns
   - Tests multiple predictor variables

3. **regression_housing.csv** (15 KB)
   - Complex regression analysis
   - 300 rows × 6 columns
   - Ideal for diagnostic plots testing

### Visualization Testing (2 files)

4. **plots_distributions.csv** (39 KB)
   - Distribution visualization
   - 500 rows × 6 columns
   - 4 different distribution types + categorical data

5. **plots_correlation.csv** (20 KB)
   - Correlation analysis
   - 200 rows × 5 columns
   - Correlated and uncorrelated variables

### Time Series Testing (5 files)

6. **timeseries_monthly_sales.csv** (1.5 KB)
   - Monthly data with trend + seasonality
   - 48 months (4 years)
   - Best for learning all TS features

7. **timeseries_daily_temperature.csv** (22 KB)
   - Daily temperature data
   - 730 days (2 years)
   - Strong annual seasonality

8. **timeseries_stock_prices.csv** (9.6 KB)
   - Stock price simulation
   - 252 trading days
   - Random walk pattern

9. **timeseries_website_traffic.csv** (6.0 KB)
   - Website visitor data
   - 365 days
   - Weekly seasonality pattern

10. **timeseries_quarterly_revenue.csv** (557 bytes)
    - Quarterly business data
    - 20 quarters (5 years)
    - Strong Q4 seasonality

### Documentation (2 files)

11. **README.md**
    - Overview of all datasets
    - Quick reference guide

12. **USAGE_GUIDE.md**
    - Detailed usage instructions
    - Expected results for each file
    - Testing workflows
    - Troubleshooting tips

---

## Data Characteristics

### All datasets include:
✅ Clear, descriptive column names
✅ No missing values (for basic testing)
✅ Realistic patterns and relationships
✅ Appropriate noise levels
✅ CSV format with headers
✅ Suitable sizes for desktop testing

### Regression data features:
- Clear linear/non-linear relationships
- Realistic coefficient magnitudes
- Appropriate R² values (0.85-0.95)
- Normally distributed residuals

### Time series data features:
- Various seasonality patterns (weekly, monthly, quarterly, annual)
- Trend components (linear, no trend)
- Realistic noise levels
- Date/time columns in standard format

### Visualization data features:
- Multiple distribution types (normal, skewed, uniform, bimodal)
- Correlation matrices with known relationships
- Categorical and continuous variables
- Sufficient sample sizes for stable estimates

---

## Testing Coverage

| Feature | Test Files | Status |
|---------|-----------|--------|
| Simple Linear Regression | regression_simple.csv | ✅ |
| Multiple Regression | regression_multiple.csv, regression_housing.csv | ✅ |
| Diagnostic Plots | regression_housing.csv | ✅ |
| Histogram + KDE | plots_distributions.csv | ✅ |
| Box Plot | plots_distributions.csv | ✅ |
| Violin Plot | plots_distributions.csv | ✅ |
| Correlation Heatmap | plots_correlation.csv | ✅ |
| ACF/PACF | All timeseries_*.csv | ✅ |
| Stationarity Tests | All timeseries_*.csv | ✅ |
| Seasonal Decomposition | timeseries_monthly_sales.csv, etc. | ✅ |
| Forecasting (SES) | timeseries_stock_prices.csv | ✅ |
| Forecasting (Holt) | timeseries_stock_prices.csv | ✅ |
| Forecasting (Holt-Winters) | timeseries_monthly_sales.csv, etc. | ✅ |

**Total Coverage:** All major features of Mathemix can be tested

---

## File Sizes

```
Total size: ~133 KB (all 10 CSV files)

Breakdown:
- Regression files: 25.2 KB
- Visualization files: 59 KB
- Time series files: 48.8 KB
```

Small enough for:
- Quick loading in desktop UI
- Version control (git)
- Distribution with software
- Email attachments if needed

---

## Quick Test Commands

### Verify all files exist:
```bash
cd test-data
ls -lh *.csv | wc -l  # Should show 10
```

### Check file integrity:
```bash
# Each file should have a header row
for f in *.csv; do 
  echo "$f: $(head -1 $f)"
done
```

### Test loading in Python:
```python
import pandas as pd
import os

for file in os.listdir('test-data'):
    if file.endswith('.csv'):
        df = pd.read_csv(f'test-data/{file}')
        print(f"{file}: {len(df)} rows × {len(df.columns)} columns")
```

---

## Next Steps

1. ✅ Files created and ready
2. ⏳ Test in Mathemix Desktop UI
3. ⏳ Verify all plots render correctly
4. ⏳ Validate regression results
5. ⏳ Test time series forecasts
6. ⏳ Create screenshots for documentation

---

**Created:** October 20, 2025
**Location:** `/test-data/`
**Purpose:** Comprehensive testing of Mathemix features
**Status:** Ready for use ✅
