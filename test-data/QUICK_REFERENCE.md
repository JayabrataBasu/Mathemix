# Test Data Quick Reference Card

## 📂 Location
`/test-data/` folder (13 files total)

---

## 🎯 Quick File Selection Guide

**Want to test...** → **Use this file:**

### Regression
- Basic regression → `regression_simple.csv`
- Multiple predictors → `regression_multiple.csv` or `regression_housing.csv`
- All diagnostic plots → `regression_housing.csv` ⭐

### Plots
- Histograms/Box plots → `plots_distributions.csv`
- Correlation heatmap → `plots_correlation.csv`

### Time Series
- Learn all TS features → `timeseries_monthly_sales.csv` ⭐ BEST FOR LEARNING
- Strong seasonality → `timeseries_daily_temperature.csv`
- Non-stationary data → `timeseries_stock_prices.csv`
- Weekly patterns → `timeseries_website_traffic.csv`
- Quarterly data → `timeseries_quarterly_revenue.csv`

---

## 🚀 5-Minute Test Workflow

### Test Regression (2 min)
```
1. Open: regression_housing.csv
2. Dependent: price
3. Independent: square_feet, bedrooms
4. Click: Run Regression
5. Click: 📊 Diagnostic Plots
```

### Test Time Series (3 min)
```
1. Open: timeseries_monthly_sales.csv
2. Tab: Time Series
3. Select: sales column
4. Click: Plot ACF + PACF
5. Click: Run Both Tests (stationarity)
6. Decompose: period=12, additive
7. Forecast: Holt-Winters, horizon=12
```

---

## 📊 Expected Results Cheatsheet

| File | Key Test | Expected Outcome |
|------|----------|------------------|
| regression_simple | OLS | R² > 0.90, positive coef |
| timeseries_monthly_sales | ACF | Spikes at lag 12, 24, 36 |
| timeseries_monthly_sales | ADF | Non-stationary (p > 0.05) |
| timeseries_monthly_sales | Decompose | Clear trend + seasonal |
| timeseries_stock_prices | ADF | Non-stationary |
| timeseries_stock_prices | KPSS | Non-stationary |
| plots_distributions | Histogram | 4 different shapes |
| plots_correlation | Heatmap | Strong correlations visible |

---

## 💡 Pro Tips

1. **Start with monthly_sales.csv** - Best for learning time series
2. **Use housing.csv** - Best for regression diagnostics
3. **Check README.md** - Quick overview of all files
4. **See USAGE_GUIDE.md** - Detailed instructions
5. **Files in git** - Won't be ignored (added to .gitignore exception)

---

## 🔍 File Naming Convention

```
[feature]_[description].csv

Examples:
- regression_simple.csv     → Regression with simple pattern
- timeseries_monthly_sales.csv → Time series with monthly frequency
- plots_distributions.csv   → For plotting various distributions
```

---

## 📝 Documentation Files

1. **README.md** - Overview of datasets
2. **USAGE_GUIDE.md** - Detailed usage with examples
3. **FILES_CREATED.md** - Complete file list with sizes
4. **QUICK_REFERENCE.md** - This file

---

**Total:** 10 CSV files (133 KB) + 3 docs  
**Status:** ✅ Ready to use  
**Updated:** October 20, 2025
