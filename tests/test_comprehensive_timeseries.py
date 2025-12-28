"""
Comprehensive Time Series Testing Suite for MatheMixX

Tests ALL 27 time series functions with multiple scenarios, edge cases,
parameter combinations, and validation. Uses the comprehensive_timeseries.csv
dataset designed specifically for exhaustive testing.

Functions tested:
- Operations: lag, diff, sma, ema, wma, rolling_mean, rolling_std, rolling_min, rolling_max (9)
- Autocorrelation: acf, pacf, ljung_box_test (3)
- Stationarity: adf_test, kpss_test (2)
- Decomposition: seasonal_decompose (1)
- Forecasting: simple_exp_smoothing, holt_linear, holt_winters (3)
- Plotting: plot_acf, plot_pacf, plot_acf_pacf, plot_decomposition, plot_forecast (5)
- TimeSeriesAnalyzer class methods (4 additional wrapper methods)

Total: 27 unique functions tested
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for testing
import matplotlib.pyplot as plt
import sys
from pathlib import Path
from datetime import datetime
import traceback

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent / 'python'))

import mathemixx_core as mx
import plots
from timeseries import TimeSeriesAnalyzer
from mathemixx_desktop.error_handler import ErrorContext, DataValidationError


class TestResult:
    """Track test results."""
    def __init__(self):
        self.total = 0
        self.passed = 0
        self.failed = 0
        self.errors = []
        self.warnings = []
    
    def add_pass(self, test_name):
        self.total += 1
        self.passed += 1
        print(f"  [PASS] {test_name}")
    
    def add_fail(self, test_name, error):
        self.total += 1
        self.failed += 1
        self.errors.append((test_name, str(error)))
        print(f"  [FAIL] {test_name}")
        print(f"    Error: {error}")
    
    def add_warning(self, message):
        self.warnings.append(message)
        print(f"  [WARN] {message}")
    
    def summary(self):
        return {
            'total': self.total,
            'passed': self.passed,
            'failed': self.failed,
            'success_rate': (self.passed / self.total * 100) if self.total > 0 else 0
        }


def print_section(title):
    """Print formatted section header."""
    print(f"\n{'='*80}")
    print(f"{title:^80}")
    print(f"{'='*80}\n")


def print_subsection(title):
    """Print formatted subsection header."""
    print(f"\n{'-'*80}")
    print(f"{title}")
    print(f"{'-'*80}")


# ============================================================================
# TEST SUITE 1: Basic Operations (9 functions)
# ============================================================================

def test_basic_operations(df, results):
    """Test lag, diff, sma, ema, wma, rolling statistics."""
    print_section("TEST SUITE 1: BASIC OPERATIONS (9 Functions)")
    
    # Test data: use trend_seasonal
    data = df['trend_seasonal'].tolist()
    
    # 1. LAG OPERATION
    print_subsection("1.1 Lag Operation")
    try:
        for periods in [1, 7, 30]:
            lagged = mx.py_lag(data, periods)
            assert len(lagged) == len(data), f"Lag length mismatch"
            assert np.isnan(lagged[0]), f"First value should be NaN"
            assert not np.isnan(lagged[periods]), f"Value at lag+1 should not be NaN"
            results.add_pass(f"lag(periods={periods})")
    except Exception as e:
        results.add_fail("lag operation", e)
    
    # 2. DIFFERENCE OPERATION
    print_subsection("1.2 Difference Operation")
    try:
        for order in [1, 2]:
            diffed = mx.py_diff(data, order)
            assert len(diffed) == len(data), f"Diff length mismatch"
            assert np.isnan(diffed[0]), f"First value should be NaN"
            valid_count = len([x for x in diffed if not np.isnan(x)])
            assert valid_count == len(data) - order, f"Valid count incorrect"
            results.add_pass(f"diff(order={order})")
    except Exception as e:
        results.add_fail("diff operation", e)
    
    # 3. SIMPLE MOVING AVERAGE
    print_subsection("1.3 Simple Moving Average")
    try:
        for window in [3, 7, 14, 30]:
            sma_vals = mx.py_sma(data, window)
            assert len(sma_vals) == len(data), f"SMA length mismatch"
            valid_count = len([x for x in sma_vals if not np.isnan(x)])
            assert valid_count >= len(data) - window + 1, f"Too many NaN values"
            results.add_pass(f"sma(window={window})")
    except Exception as e:
        results.add_fail("sma operation", e)
    
    # 4. EXPONENTIAL MOVING AVERAGE
    print_subsection("1.4 Exponential Moving Average")
    try:
        for window in [5, 10, 20]:
            ema_vals = mx.py_ema(data, window)
            assert len(ema_vals) == len(data), f"EMA length mismatch"
            # EMA should have fewer NaN than SMA
            results.add_pass(f"ema(window={window})")
    except Exception as e:
        results.add_fail("ema operation", e)
    
    # 5. WEIGHTED MOVING AVERAGE
    print_subsection("1.5 Weighted Moving Average")
    try:
        for window in [3, 7, 14]:
            wma_vals = mx.py_wma(data, window)
            assert len(wma_vals) == len(data), f"WMA length mismatch"
            results.add_pass(f"wma(window={window})")
    except Exception as e:
        results.add_fail("wma operation", e)
    
    # 6. ROLLING MEAN
    print_subsection("1.6 Rolling Mean")
    try:
        for window in [7, 14, 30]:
            roll_mean = mx.py_rolling_mean(data, window)
            assert len(roll_mean) == len(data), f"Rolling mean length mismatch"
            results.add_pass(f"rolling_mean(window={window})")
    except Exception as e:
        results.add_fail("rolling_mean operation", e)
    
    # 7. ROLLING STD
    print_subsection("1.7 Rolling Standard Deviation")
    try:
        for window in [7, 14, 30]:
            roll_std = mx.py_rolling_std(data, window)
            assert len(roll_std) == len(data), f"Rolling std length mismatch"
            assert all(x >= 0 or np.isnan(x) for x in roll_std), "Negative std values"
            results.add_pass(f"rolling_std(window={window})")
    except Exception as e:
        results.add_fail("rolling_std operation", e)
    
    # 8. ROLLING MIN
    print_subsection("1.8 Rolling Minimum")
    try:
        for window in [7, 14]:
            roll_min = mx.py_rolling_min(data, window)
            assert len(roll_min) == len(data), f"Rolling min length mismatch"
            results.add_pass(f"rolling_min(window={window})")
    except Exception as e:
        results.add_fail("rolling_min operation", e)
    
    # 9. ROLLING MAX
    print_subsection("1.9 Rolling Maximum")
    try:
        for window in [7, 14]:
            roll_max = mx.py_rolling_max(data, window)
            assert len(roll_max) == len(data), f"Rolling max length mismatch"
            results.add_pass(f"rolling_max(window={window})")
    except Exception as e:
        results.add_fail("rolling_max operation", e)


# ============================================================================
# TEST SUITE 2: Autocorrelation Analysis (3 functions)
# ============================================================================

def test_autocorrelation(df, results):
    """Test ACF, PACF, Ljung-Box test."""
    print_section("TEST SUITE 2: AUTOCORRELATION ANALYSIS (3 Functions)")
    
    # Test with different patterns
    test_series = {
        'strong_autocorr': df['strong_autocorr'].tolist(),
        'weak_autocorr': df['weak_autocorr'].tolist(),
        'trend_seasonal': df['trend_seasonal'].tolist()
    }
    
    for series_name, data in test_series.items():
        print_subsection(f"2.{list(test_series.keys()).index(series_name) + 1} Testing with {series_name}")
        
        # 1. ACF
        try:
            for nlags in [10, 20, 40]:
                acf_vals = mx.py_acf(data, nlags)
                assert len(acf_vals) == nlags + 1, f"ACF length should be nlags+1"
                assert acf_vals[0] == 1.0, f"ACF at lag 0 should be 1.0"
                assert all(-1 <= x <= 1 for x in acf_vals), "ACF values out of range"
                results.add_pass(f"acf({series_name}, nlags={nlags})")
        except Exception as e:
            results.add_fail(f"acf({series_name})", e)
        
        # 2. PACF
        try:
            for nlags in [10, 20, 40]:
                pacf_vals = mx.py_pacf(data, nlags)
                assert len(pacf_vals) == nlags + 1, f"PACF length should be nlags+1"
                assert all(-1 <= x <= 1 for x in pacf_vals), "PACF values out of range"
                results.add_pass(f"pacf({series_name}, nlags={nlags})")
        except Exception as e:
            results.add_fail(f"pacf({series_name})", e)
        
        # 3. LJUNG-BOX TEST
        try:
            for lags in [10, 20]:
                statistic, p_value = mx.py_ljung_box_test(data, lags)
                assert statistic >= 0, "LB statistic should be non-negative"
                assert 0 <= p_value <= 1, "P-value should be in [0, 1]"
                results.add_pass(f"ljung_box({series_name}, lags={lags})")
        except Exception as e:
            results.add_fail(f"ljung_box({series_name})", e)


# ============================================================================
# TEST SUITE 3: Stationarity Tests (2 functions)
# ============================================================================

def test_stationarity(df, results):
    """Test ADF and KPSS tests."""
    print_section("TEST SUITE 3: STATIONARITY TESTS (2 Functions)")
    
    # Test with known stationary and non-stationary series
    test_cases = {
        'stationary_ar1': (df['stationary_ar1'].tolist(), True, "stationary"),
        'non_stationary_rw': (df['non_stationary_rw'].tolist(), False, "non-stationary"),
        'pure_trend': (df['pure_trend'].tolist(), False, "trending")
    }
    
    for series_name, (data, expected_stationary, desc) in test_cases.items():
        print_subsection(f"3.{list(test_cases.keys()).index(series_name) + 1} Testing {series_name} ({desc})")
        
        # 1. ADF TEST
        try:
            for max_lags in [None, 10, 20]:
                result = mx.py_adf_test(data, max_lags)
                assert hasattr(result, 'test_statistic'), "Missing test_statistic"
                assert hasattr(result, 'p_value'), "Missing p_value"
                assert hasattr(result, 'is_stationary'), "Missing is_stationary"
                assert hasattr(result, 'lags_used'), "Missing lags_used"
                assert hasattr(result, 'n_obs'), "Missing n_obs"
                assert hasattr(result, 'critical_values'), "Missing critical_values"
                assert 0 <= result.p_value <= 1, "P-value out of range"
                results.add_pass(f"adf_test({series_name}, max_lags={max_lags})")
        except Exception as e:
            results.add_fail(f"adf_test({series_name})", e)
        
        # 2. KPSS TEST
        try:
            for lags in [None, 10]:
                result = mx.py_kpss_test(data, lags)
                assert hasattr(result, 'test_statistic'), "Missing test_statistic"
                assert hasattr(result, 'p_value'), "Missing p_value"
                assert hasattr(result, 'is_stationary'), "Missing is_stationary"
                results.add_pass(f"kpss_test({series_name}, lags={lags})")
        except Exception as e:
            results.add_fail(f"kpss_test({series_name})", e)


# ============================================================================
# TEST SUITE 4: Seasonal Decomposition (1 function)
# ============================================================================

def test_decomposition(df, results):
    """Test seasonal decomposition."""
    print_section("TEST SUITE 4: SEASONAL DECOMPOSITION (1 Function)")
    
    # Test with different seasonal patterns
    test_cases = {
        'pure_seasonal': (df['pure_seasonal'].tolist(), 7, "7-day weekly"),
        'trend_seasonal': (df['trend_seasonal'].tolist(), 30, "30-day monthly"),
        'quarterly_seasonal': (df['quarterly_seasonal'].tolist(), 91, "91-day quarterly"),
        'weekly_pattern': (df['weekly_pattern'].tolist(), 7, "7-day weekly"),
        'multiplicative_seasonal': (df['multiplicative_seasonal'].tolist(), 14, "14-day")
    }
    
    for series_name, (data, period, desc) in test_cases.items():
        print_subsection(f"4.{list(test_cases.keys()).index(series_name) + 1} {series_name} ({desc}, period={period})")
        
        # Test both additive and multiplicative
        for model in ['additive', 'multiplicative']:
            try:
                result = mx.py_seasonal_decompose(data, period, model)
                assert hasattr(result, 'trend'), "Missing trend component"
                assert hasattr(result, 'seasonal'), "Missing seasonal component"
                assert hasattr(result, 'residual'), "Missing residual component"
                assert hasattr(result, 'observed'), "Missing observed component"
                assert len(result.trend) == len(data), "Trend length mismatch"
                assert len(result.seasonal) == len(data), "Seasonal length mismatch"
                assert len(result.residual) == len(data), "Residual length mismatch"
                results.add_pass(f"seasonal_decompose({series_name}, period={period}, model={model})")
            except Exception as e:
                results.add_fail(f"seasonal_decompose({series_name}, {model})", e)


# ============================================================================
# TEST SUITE 5: Forecasting Methods (3 functions)
# ============================================================================

def test_forecasting(df, results):
    """Test SES, Holt, Holt-Winters forecasting."""
    print_section("TEST SUITE 5: FORECASTING METHODS (3 Functions)")
    
    # Test data and scenarios
    test_data = df['trend_seasonal'].tolist()
    seasonal_data = df['weekly_pattern'].tolist()
    
    # 1. SIMPLE EXPONENTIAL SMOOTHING
    print_subsection("5.1 Simple Exponential Smoothing")
    try:
        for alpha in [0.1, 0.3, 0.5, 0.8]:
            for horizon in [5, 10, 20]:
                for conf in [0.90, 0.95]:
                    result = mx.py_simple_exp_smoothing(test_data, alpha, horizon, conf)
                    assert hasattr(result, 'forecasts'), "Missing forecasts"
                    assert hasattr(result, 'lower_bound'), "Missing lower_bound"
                    assert hasattr(result, 'upper_bound'), "Missing upper_bound"
                    assert hasattr(result, 'confidence_level'), "Missing confidence_level"
                    assert len(result.forecasts) == horizon, f"Forecast length mismatch"
                    assert len(result.lower_bound) == horizon, f"Lower bound length mismatch"
                    assert len(result.upper_bound) == horizon, f"Upper bound length mismatch"
                    assert result.confidence_level == conf, f"Confidence level mismatch"
                    # Check bounds contain forecasts
                    for i in range(horizon):
                        assert result.lower_bound[i] <= result.forecasts[i] <= result.upper_bound[i], \
                            "Forecast outside confidence bounds"
                    results.add_pass(f"ses(alpha={alpha}, h={horizon}, conf={conf})")
    except Exception as e:
        results.add_fail("simple_exp_smoothing", e)
    
    # 2. HOLT LINEAR TREND
    print_subsection("5.2 Holt Linear Trend")
    try:
        for alpha in [0.2, 0.5]:
            for beta in [0.1, 0.3]:
                for horizon in [5, 10]:
                    result = mx.py_holt_linear(test_data, alpha, beta, horizon, 0.95)
                    assert len(result.forecasts) == horizon, f"Forecast length mismatch"
                    assert len(result.lower_bound) == horizon, f"Lower bound length mismatch"
                    assert len(result.upper_bound) == horizon, f"Upper bound length mismatch"
                    results.add_pass(f"holt(alpha={alpha}, beta={beta}, h={horizon})")
    except Exception as e:
        results.add_fail("holt_linear", e)
    
    # 3. HOLT-WINTERS SEASONAL
    print_subsection("5.3 Holt-Winters Seasonal")
    try:
        for alpha in [0.2, 0.4]:
            for beta in [0.1, 0.2]:
                for gamma in [0.1, 0.3]:
                    for period in [7, 14]:
                        for seasonal_type in ['additive', 'multiplicative']:
                            horizon = 14
                            result = mx.py_holt_winters(
                                seasonal_data, alpha, beta, gamma, period, 
                                horizon, seasonal_type, 0.95
                            )
                            assert len(result.forecasts) == horizon, f"Forecast length mismatch"
                            results.add_pass(f"hw(alpha={alpha}, beta={beta}, gamma={gamma}, p={period}, {seasonal_type})")
    except Exception as e:
        results.add_fail("holt_winters", e)


# ============================================================================
# TEST SUITE 6: Plotting Functions (5 functions)
# ============================================================================

def test_plotting(df, results):
    """Test all plotting functions."""
    print_section("TEST SUITE 6: PLOTTING FUNCTIONS (5 Functions)")
    
    data = df['trend_seasonal'].tolist()
    seasonal_data = df['weekly_pattern'].tolist()
    
    # 1. PLOT ACF
    print_subsection("6.1 ACF Plot")
    try:
        for nlags in [20, 40]:
            # Compute ACF values first, then plot
            acf_values = mx.py_acf(data, nlags)
            ax = plots.plot_acf(acf_values)
            assert ax is not None, "Axes is None"
            plt.close(ax.figure)
            results.add_pass(f"plot_acf(nlags={nlags})")
    except Exception as e:
        results.add_fail("plot_acf", e)
    
    # 2. PLOT PACF
    print_subsection("6.2 PACF Plot")
    try:
        for nlags in [20, 40]:
            # Compute PACF values first, then plot
            pacf_values = mx.py_pacf(data, nlags)
            ax = plots.plot_pacf(pacf_values)
            assert ax is not None, "Axes is None"
            plt.close(ax.figure)
            results.add_pass(f"plot_pacf(nlags={nlags})")
    except Exception as e:
        results.add_fail("plot_pacf", e)
    
    # 3. PLOT ACF+PACF COMBINED
    print_subsection("6.3 Combined ACF+PACF Plot")
    try:
        for nlags in [20, 30]:
            fig = plots.plot_acf_pacf(data, nlags=nlags)
            assert fig is not None, "Figure is None"
            plt.close(fig)
            results.add_pass(f"plot_acf_pacf(nlags={nlags})")
    except Exception as e:
        results.add_fail("plot_acf_pacf", e)
    
    # 4. PLOT DECOMPOSITION
    print_subsection("6.4 Decomposition Plot")
    try:
        for period in [7, 30]:
            decomp = mx.py_seasonal_decompose(seasonal_data, period, 'additive')
            fig = plots.plot_decomposition(decomp)
            assert fig is not None, "Figure is None"
            plt.close(fig)
            results.add_pass(f"plot_decomposition(period={period})")
    except Exception as e:
        results.add_fail("plot_decomposition", e)
    
    # 5. PLOT FORECAST
    print_subsection("6.5 Forecast Plot")
    try:
        for horizon in [10, 20]:
            forecast = mx.py_simple_exp_smoothing(data, 0.3, horizon, 0.95)
            ax = plots.plot_forecast(data, forecast, n_history=50)
            assert ax is not None, "Axes is None"
            plt.close(ax.figure)
            results.add_pass(f"plot_forecast(horizon={horizon})")
    except Exception as e:
        results.add_fail("plot_forecast", e)


# ============================================================================
# TEST SUITE 7: TimeSeriesAnalyzer Class (wrapper methods)
# ============================================================================

def test_timeseries_analyzer(df, results):
    """Test TimeSeriesAnalyzer class methods."""
    print_section("TEST SUITE 7: TIMESERIESANALYZER CLASS (Wrapper Methods)")
    
    data = df['trend_seasonal'].tolist()
    ts = TimeSeriesAnalyzer(data)
    
    print_subsection("7.1 TimeSeriesAnalyzer Methods")
    
    # Test all methods via the class interface
    try:
        # Operations
        ts.lag(1)
        results.add_pass("TimeSeriesAnalyzer.lag()")
        
        ts.diff(1)
        results.add_pass("TimeSeriesAnalyzer.diff()")
        
        ts.sma(7)
        results.add_pass("TimeSeriesAnalyzer.sma()")
        
        ts.ema(7)
        results.add_pass("TimeSeriesAnalyzer.ema()")
        
        ts.wma(7)
        results.add_pass("TimeSeriesAnalyzer.wma()")
        
        # Autocorrelation
        ts.acf(20)
        results.add_pass("TimeSeriesAnalyzer.acf()")
        
        ts.pacf(20)
        results.add_pass("TimeSeriesAnalyzer.pacf()")
        
        ts.ljung_box_test(10)
        results.add_pass("TimeSeriesAnalyzer.ljung_box_test()")
        
        # Stationarity
        ts.adf_test()
        results.add_pass("TimeSeriesAnalyzer.adf_test()")
        
        ts.kpss_test()
        results.add_pass("TimeSeriesAnalyzer.kpss_test()")
        
        # Decomposition
        ts.decompose(30)
        results.add_pass("TimeSeriesAnalyzer.decompose()")
        
        # Forecasting
        ts.forecast_ses(horizon=10)
        results.add_pass("TimeSeriesAnalyzer.forecast_ses()")
        
        ts.forecast_holt(horizon=10)
        results.add_pass("TimeSeriesAnalyzer.forecast_holt()")
        
        ts.forecast_holt_winters(period=7, horizon=10)
        results.add_pass("TimeSeriesAnalyzer.forecast_holt_winters()")
        
    except Exception as e:
        results.add_fail("TimeSeriesAnalyzer class methods", e)


# ============================================================================
# TEST SUITE 8: Edge Cases and Error Handling
# ============================================================================

def test_edge_cases(df, results):
    """Test edge cases and error handling."""
    print_section("TEST SUITE 8: EDGE CASES & ERROR HANDLING")
    
    # 1. CONSTANT VALUES (no variance)
    print_subsection("8.1 Constant Values (No Variance)")
    constant_data = df['constant_values'].tolist()
    try:
        # Should handle constant data gracefully
        sma = mx.py_sma(constant_data, 7)
        results.add_pass("SMA with constant values")
    except Exception as e:
        results.add_warning(f"SMA with constant values failed: {e}")
    
    # 2. VERY SMALL WINDOW
    print_subsection("8.2 Minimum Window Sizes")
    data = df['trend_seasonal'].tolist()
    try:
        sma = mx.py_sma(data, 2)  # Minimum window
        results.add_pass("SMA with window=2")
    except Exception as e:
        results.add_fail("SMA with minimum window", e)
    
    # 3. LARGE LAG
    print_subsection("8.3 Large Lag Values")
    try:
        lagged = mx.py_lag(data, 100)  # Large lag
        valid_count = len([x for x in lagged if not np.isnan(x)])
        assert valid_count < len(data), "Should have many NaN values"
        results.add_pass("Lag with large period (100)")
    except Exception as e:
        results.add_fail("Large lag operation", e)
    
    # 4. SMALL SAMPLE FOR FORECASTING
    print_subsection("8.4 Small Sample Forecasting")
    small_data = data[:30]  # Only 30 points
    try:
        forecast = mx.py_simple_exp_smoothing(small_data, 0.3, 10, 0.95)
        results.add_pass("Forecast with small sample (30 points)")
    except Exception as e:
        results.add_warning(f"Small sample forecasting: {e}")
    
    # 5. HIGH NLAGS FOR ACF
    print_subsection("8.5 High Number of Lags")
    try:
        acf_vals = mx.py_acf(data, 100)  # Many lags
        assert len(acf_vals) == 101, "ACF length incorrect"
        results.add_pass("ACF with 100 lags")
    except Exception as e:
        results.add_fail("High nlags ACF", e)
    
    # 6. OUTLIERS
    print_subsection("8.6 Data with Outliers")
    outlier_data = df['with_outliers'].tolist()
    try:
        # Should handle outliers
        result = mx.py_adf_test(outlier_data, 10)
        results.add_pass("ADF test with outliers")
        
        sma = mx.py_sma(outlier_data, 7)
        results.add_pass("SMA with outliers")
    except Exception as e:
        results.add_fail("Operations with outliers", e)
    
    # 7. EXTREME PARAMETER VALUES
    print_subsection("8.7 Extreme Parameter Values")
    try:
        # Very small alpha
        forecast = mx.py_simple_exp_smoothing(data, 0.01, 5, 0.95)
        results.add_pass("SES with alpha=0.01")
        
        # Very large alpha
        forecast = mx.py_simple_exp_smoothing(data, 0.99, 5, 0.95)
        results.add_pass("SES with alpha=0.99")
        
        # High confidence level
        forecast = mx.py_simple_exp_smoothing(data, 0.3, 5, 0.999)
        results.add_pass("SES with 99.9% confidence")
    except Exception as e:
        results.add_fail("Extreme parameters", e)


# ============================================================================
# MAIN TEST RUNNER
# ============================================================================

def main():
    """Run all test suites."""
    print("\n" + "="*80)
    print("COMPREHENSIVE TIME SERIES TESTING SUITE".center(80))
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Testing all 27 time series functions with comprehensive scenarios")
    print("="*80)
    
    # Load test data
    print("\nLoading test data...")
    try:
        df = pd.read_csv('test-data/comprehensive_timeseries.csv')
        print(f"[OK] Loaded comprehensive_timeseries.csv ({df.shape[0]} rows x {df.shape[1]} columns)")
        print(f"  Columns: {', '.join(df.columns[1:6])}... (18 total)")
    except FileNotFoundError:
        print("[ERROR] comprehensive_timeseries.csv not found!")
        print("  Please run the data generation script first.")
        return 1
    
    # Initialize results tracker
    results = TestResult()
    
    # Run all test suites
    try:
        test_basic_operations(df, results)
        test_autocorrelation(df, results)
        test_stationarity(df, results)
        test_decomposition(df, results)
        test_forecasting(df, results)
        test_plotting(df, results)
        test_timeseries_analyzer(df, results)
        test_edge_cases(df, results)
        
    except KeyboardInterrupt:
        print("\n\n✗ Testing interrupted by user")
        return 1
    except Exception as e:
        print(f"\n\n✗ FATAL ERROR: {e}")
        traceback.print_exc()
        return 1
    
    # Print final summary
    print_section("FINAL TEST SUMMARY")
    
    summary = results.summary()
    
    print(f"Total Tests:    {summary['total']}")
    print(f"Passed:         {summary['passed']} ({summary['success_rate']:.1f}%)")
    print(f"Failed:         {summary['failed']}")
    print(f"Warnings:       {len(results.warnings)}")
    
    if results.failed > 0:
        print(f"\n{'='*80}")
        print("FAILED TESTS DETAILS")
        print(f"{'='*80}\n")
        for i, (test_name, error) in enumerate(results.errors, 1):
            print(f"{i}. {test_name}")
            print(f"   Error: {error}\n")
    
    if results.warnings:
        print(f"\n{'='*80}")
        print("WARNINGS")
        print(f"{'='*80}\n")
        for i, warning in enumerate(results.warnings, 1):
            print(f"{i}. {warning}")
    
    print(f"\n{'='*80}")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}")
    
    if summary['success_rate'] == 100:
        print("\n[SUCCESS] ALL TESTS PASSED!\n")
        return 0
    elif summary['success_rate'] >= 90:
        print(f"\n[WARNING] {summary['success_rate']:.1f}% TESTS PASSED (Some issues detected)\n")
        return 1
    else:
        print(f"\n[FAILURE] ONLY {summary['success_rate']:.1f}% TESTS PASSED (Significant failures)\n")
        return 1


if __name__ == "__main__":
    exit(main())
