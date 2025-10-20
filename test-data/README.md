# Test Data Files for Mathemix

This folder contains synthetic datasets for testing various features of Mathemix.

## Files Overview

### Regression Testing
- `regression_simple.csv` - Simple linear regression (1 predictor)
- `regression_multiple.csv` - Multiple linear regression (3 predictors)
- `regression_housing.csv` - Housing price prediction dataset

### Visualization/Plotting
- `plots_distributions.csv` - Data for histogram, box plot, violin plot testing
- `plots_correlation.csv` - Data for correlation heatmap testing

### Time Series Analysis
- `timeseries_monthly_sales.csv` - Monthly sales data with trend and seasonality
- `timeseries_daily_temperature.csv` - Daily temperature data with seasonality
- `timeseries_stock_prices.csv` - Simulated stock price data (random walk)
- `timeseries_website_traffic.csv` - Weekly website traffic pattern
- `timeseries_quarterly_revenue.csv` - Quarterly revenue with strong seasonality

## Data Characteristics

All datasets are synthetically generated with:
- Realistic patterns and relationships
- Appropriate noise levels
- Clear headers
- No missing values (for basic testing)
- Suitable size for desktop application testing

## Usage

1. Launch Mathemix Desktop: `python run_desktop.py`
2. File → Open CSV
3. Navigate to `test-data/` folder
4. Select the appropriate file for the feature you want to test

---
Generated: October 2025
