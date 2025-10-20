#!/usr/bin/env python3
"""
Mathemix Test Data Verification Script

This script systematically tests all CSV files in the test-data folder
against all Mathemix features and reports success/failure with detailed errors.

Usage: python verify_test_data.py
"""

import sys
import os
from pathlib import Path
import traceback
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Color codes for terminal output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_header(text):
    """Print a formatted header."""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*80}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text.center(80)}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*80}{Colors.END}\n")

def print_section(text):
    """Print a section header."""
    print(f"\n{Colors.BOLD}{text}{Colors.END}")
    print(f"{'-'*80}")

def print_success(text):
    """Print success message."""
    print(f"{Colors.GREEN}[OK] {text}{Colors.END}")

def print_failure(text):
    """Print failure message."""
    print(f"{Colors.RED}[FAIL] {text}{Colors.END}")

def print_warning(text):
    """Print warning message."""
    print(f"{Colors.YELLOW}[WARN] {text}{Colors.END}")

def print_info(text):
    """Print info message."""
    print(f"{Colors.BLUE}[INFO] {text}{Colors.END}")


class TestResults:
    """Store and display test results."""
    
    def __init__(self):
        self.total = 0
        self.passed = 0
        self.failed = 0
        self.errors = []
    
    def add_pass(self, test_name):
        self.total += 1
        self.passed += 1
        print_success(f"{test_name}")
    
    def add_fail(self, test_name, error):
        self.total += 1
        self.failed += 1
        self.errors.append((test_name, error))
        print_failure(f"{test_name}")
        print(f"  {Colors.RED}Error: {error}{Colors.END}")
    
    def print_summary(self):
        """Print final summary."""
        print_header("TEST SUMMARY")
        
        print(f"Total Tests: {self.total}")
        print(f"{Colors.GREEN}Passed: {self.passed}{Colors.END}")
        print(f"{Colors.RED}Failed: {self.failed}{Colors.END}")
        print(f"Success Rate: {(self.passed/self.total*100) if self.total > 0 else 0:.1f}%\n")
        
        if self.errors:
            print_section("FAILED TESTS DETAILS")
            for i, (test_name, error) in enumerate(self.errors, 1):
                print(f"\n{i}. {Colors.RED}{test_name}{Colors.END}")
                print(f"   Error: {error}")


def test_imports():
    """Test if all required modules can be imported."""
    print_section("1. Testing Imports")
    results = TestResults()
    
    # Test pandas and numpy
    try:
        import pandas as pd
        results.add_pass("Import pandas")
    except ImportError as e:
        results.add_fail("Import pandas", str(e))
        return results
    
    try:
        import numpy as np
        results.add_pass("Import numpy")
    except ImportError as e:
        results.add_fail("Import numpy", str(e))
        return results
    
    # Test mathemixx_core
    try:
        import mathemixx_core as mx
        results.add_pass("Import mathemixx_core")
    except ImportError as e:
        results.add_fail("Import mathemixx_core", str(e))
        print_warning("Core library not installed. Run: pip install -e .")
        return results
    
    # Test Phase 7 bindings
    try:
        if hasattr(mx, 'py_lag'):
            results.add_pass("Phase 7 bindings available (py_lag)")
        else:
            results.add_fail("Phase 7 bindings", "py_lag not found - package needs reinstall")
    except Exception as e:
        results.add_fail("Phase 7 bindings check", str(e))
    
    # Test plots module
    try:
        sys.path.insert(0, str(Path(__file__).parent / "python"))
        import plots
        results.add_pass("Import plots module")
    except ImportError as e:
        results.add_fail("Import plots module", str(e))
    
    # Test timeseries module
    try:
        from python.timeseries import TimeSeriesAnalyzer
        results.add_pass("Import timeseries module")
    except ImportError as e:
        results.add_fail("Import timeseries module", str(e))
    
    return results


def test_data_loading():
    """Test loading all CSV files."""
    print_section("2. Testing Data Loading")
    results = TestResults()
    
    import pandas as pd
    
    test_data_dir = Path(__file__).parent / "test-data"
    
    if not test_data_dir.exists():
        results.add_fail("test-data folder", "Folder not found")
        return results
    
    csv_files = sorted(test_data_dir.glob("*.csv"))
    
    if not csv_files:
        results.add_fail("CSV files", "No CSV files found in test-data/")
        return results
    
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            if len(df) > 0:
                results.add_pass(f"Load {csv_file.name} ({len(df)} rows × {len(df.columns)} cols)")
            else:
                results.add_fail(f"Load {csv_file.name}", "Empty dataframe")
        except Exception as e:
            results.add_fail(f"Load {csv_file.name}", str(e))
    
    return results


def test_regression_features():
    """Test regression functionality with test data."""
    print_section("3. Testing Regression Features")
    results = TestResults()
    
    try:
        import pandas as pd
        import mathemixx_core as mx
        
        test_data_dir = Path(__file__).parent / "test-data"
        
        # Test simple regression
        try:
            df = pd.read_csv(test_data_dir / "regression_simple.csv")
            csv_path = str(test_data_dir / "regression_simple.csv")
            dataset = mx.DataSet.from_csv(csv_path)
            
            # Get columns
            cols = df.columns.tolist()
            if len(cols) >= 2:
                result = dataset.regress_ols(cols[1], [cols[0]], True)
                
                if result.r_squared() > 0.5:
                    results.add_pass(f"Simple regression (R²={result.r_squared():.3f})")
                else:
                    results.add_fail("Simple regression", f"Low R² = {result.r_squared():.3f}")
            else:
                results.add_fail("Simple regression", "Not enough columns")
        except Exception as e:
            results.add_fail("Simple regression", str(e))
        
        # Test multiple regression
        try:
            df = pd.read_csv(test_data_dir / "regression_multiple.csv")
            csv_path = str(test_data_dir / "regression_multiple.csv")
            dataset = mx.DataSet.from_csv(csv_path)
            
            cols = df.columns.tolist()
            if len(cols) >= 4:
                # Last column is dependent, first 3 are independent
                result = dataset.regress_ols(cols[-1], cols[:-1], True)
                
                if result.nobs() == len(df):
                    results.add_pass(f"Multiple regression (n={result.nobs()}, R²={result.r_squared():.3f})")
                else:
                    results.add_fail("Multiple regression", "Observation count mismatch")
            else:
                results.add_fail("Multiple regression", "Not enough columns")
        except Exception as e:
            results.add_fail("Multiple regression", str(e))
        
        # Test housing regression (complex)
        try:
            csv_path = str(test_data_dir / "regression_housing.csv")
            dataset = mx.DataSet.from_csv(csv_path)
            df = pd.read_csv(csv_path)
            
            cols = df.columns.tolist()
            # price is dependent, use first 3 predictors
            result = dataset.regress_ols('price', ['square_feet', 'bedrooms', 'bathrooms'], True)
            
            coef_table = result.table()
            if len(coef_table) > 0:
                results.add_pass(f"Housing regression ({len(coef_table)} coefficients)")
            else:
                results.add_fail("Housing regression", "No coefficients returned")
        except Exception as e:
            results.add_fail("Housing regression", str(e))
        
        # Test summarize
        try:
            csv_path = str(test_data_dir / "regression_simple.csv")
            dataset = mx.DataSet.from_csv(csv_path)
            summary = dataset.summarize()
            
            if len(summary) > 0:
                results.add_pass(f"Dataset summarize ({len(summary)} variables)")
            else:
                results.add_fail("Dataset summarize", "Empty summary")
        except Exception as e:
            results.add_fail("Dataset summarize", str(e))
    
    except ImportError as e:
        results.add_fail("Regression tests", f"Import error: {e}")
    
    return results


def test_visualization_features():
    """Test visualization/plotting functionality."""
    print_section("4. Testing Visualization Features")
    results = TestResults()
    
    try:
        import pandas as pd
        import mathemixx_core as mx
        import matplotlib.pyplot as plt
        sys.path.insert(0, str(Path(__file__).parent / "python"))
        import plots
        
        test_data_dir = Path(__file__).parent / "test-data"
        
        # Test distributions plots
        try:
            df = pd.read_csv(test_data_dir / "plots_distributions.csv")
            csv_path = str(test_data_dir / "plots_distributions.csv")
            dataset = mx.DataSet.from_csv(csv_path)
            
            # Test boxplot
            fig, ax = plt.subplots()
            plots.plot_boxplot(dataset, 'normal_dist', ax=ax)
            plt.close(fig)
            results.add_pass("Box plot")
        except Exception as e:
            results.add_fail("Box plot", str(e))
        
        # Test histogram
        try:
            df = pd.read_csv(test_data_dir / "plots_distributions.csv")
            csv_path = str(test_data_dir / "plots_distributions.csv")
            dataset = mx.DataSet.from_csv(csv_path)
            
            fig, ax = plt.subplots()
            plots.plot_histogram(dataset, 'normal_dist', kde=True, ax=ax)
            plt.close(fig)
            results.add_pass("Histogram with KDE")
        except Exception as e:
            results.add_fail("Histogram with KDE", str(e))
        
        # Test violin plot
        try:
            fig, ax = plt.subplots()
            plots.plot_violin(dataset, 'normal_dist', ax=ax)
            plt.close(fig)
            results.add_pass("Violin plot")
        except Exception as e:
            results.add_fail("Violin plot", str(e))
        
        # Test correlation heatmap
        try:
            csv_path = str(test_data_dir / "plots_correlation.csv")
            dataset = mx.DataSet.from_csv(csv_path)
            
            fig, ax = plt.subplots()
            plots.plot_correlation_heatmap(dataset, method='pearson', ax=ax)
            plt.close(fig)
            results.add_pass("Correlation heatmap")
        except Exception as e:
            results.add_fail("Correlation heatmap", str(e))
    
    except ImportError as e:
        results.add_fail("Visualization tests", f"Import error: {e}")
    
    return results


def test_timeseries_features():
    """Test time series analysis functionality."""
    print_section("5. Testing Time Series Features")
    results = TestResults()
    
    try:
        import pandas as pd
        import numpy as np
        import mathemixx_core as mx
        
        test_data_dir = Path(__file__).parent / "test-data"
        
        # Load monthly sales data
        df = pd.read_csv(test_data_dir / "timeseries_monthly_sales.csv")
        data = df['sales'].tolist()
        
        # Check if Phase 7 bindings are available
        if not hasattr(mx, 'py_lag'):
            print_warning("Phase 7 bindings not available - skipping time series tests")
            print_info("To fix: Restart Python and run: pip uninstall mathemixx-core -y && pip install -e .")
            return results
        
        # Test lag
        try:
            lagged = mx.py_lag(data, 1)
            if len(lagged) == len(data):
                results.add_pass(f"Lag operation ({len(lagged)} values)")
            else:
                results.add_fail("Lag operation", f"Expected {len(data)}, got {len(lagged)}")
        except Exception as e:
            results.add_fail("Lag operation", str(e))
        
        # Test diff
        try:
            diffed = mx.py_diff(data, 1)
            if len(diffed) == len(data):
                results.add_pass(f"Difference operation ({len(diffed)} values)")
            else:
                results.add_fail("Difference operation", f"Expected {len(data)}, got {len(diffed)}")
        except Exception as e:
            results.add_fail("Difference operation", str(e))
        
        # Test SMA
        try:
            sma_vals = mx.py_sma(data, 3)
            if len(sma_vals) > 0:
                results.add_pass(f"Simple Moving Average ({len(sma_vals)} values)")
            else:
                results.add_fail("Simple Moving Average", "No values returned")
        except Exception as e:
            results.add_fail("Simple Moving Average", str(e))
        
        # Test EMA
        try:
            ema_vals = mx.py_ema(data, 3)
            if len(ema_vals) == len(data):
                results.add_pass(f"Exponential Moving Average ({len(ema_vals)} values)")
            else:
                results.add_fail("Exponential Moving Average", f"Expected {len(data)}, got {len(ema_vals)}")
        except Exception as e:
            results.add_fail("Exponential Moving Average", str(e))
        
        # Test ACF
        try:
            acf_vals = mx.py_acf(data, 10)
            if len(acf_vals) == 11:  # lags 0 to 10
                results.add_pass(f"ACF calculation ({len(acf_vals)} lags)")
            else:
                results.add_fail("ACF calculation", f"Expected 11 lags, got {len(acf_vals)}")
        except Exception as e:
            results.add_fail("ACF calculation", str(e))
        
        # Test PACF
        try:
            pacf_vals = mx.py_pacf(data, 10)
            if len(pacf_vals) == 11:
                results.add_pass(f"PACF calculation ({len(pacf_vals)} lags)")
            else:
                results.add_fail("PACF calculation", f"Expected 11 lags, got {len(pacf_vals)}")
        except Exception as e:
            results.add_fail("PACF calculation", str(e))
        
        # Test Ljung-Box
        try:
            lb_result = mx.py_ljung_box_test(data, 10)
            if isinstance(lb_result, tuple) and len(lb_result) == 2:
                statistic, p_value = lb_result
                results.add_pass(f"Ljung-Box test (stat={statistic:.3f}, p={p_value:.3f})")
            else:
                results.add_fail("Ljung-Box test", f"Unexpected result type: {type(lb_result)}")
        except Exception as e:
            results.add_fail("Ljung-Box test", str(e))
        
        # Test ADF
        try:
            adf_result = mx.py_adf_test(data, 10)
            if hasattr(adf_result, 'test_statistic') and hasattr(adf_result, 'is_stationary'):
                status = "stationary" if adf_result.is_stationary else "non-stationary"
                results.add_pass(f"ADF test ({status}, p={adf_result.p_value:.3f})")
            else:
                results.add_fail("ADF test", "Missing required attributes")
        except Exception as e:
            results.add_fail("ADF test", str(e))
        
        # Test KPSS
        try:
            kpss_result = mx.py_kpss_test(data, 10)
            if hasattr(kpss_result, 'test_statistic') and hasattr(kpss_result, 'is_stationary'):
                status = "stationary" if kpss_result.is_stationary else "non-stationary"
                results.add_pass(f"KPSS test ({status}, p={kpss_result.p_value:.3f})")
            else:
                results.add_fail("KPSS test", "Missing required attributes")
        except Exception as e:
            results.add_fail("KPSS test", str(e))
        
        # Test decomposition
        try:
            decomp = mx.py_seasonal_decompose(data, 12, 'additive')
            if hasattr(decomp, 'trend') and hasattr(decomp, 'seasonal'):
                results.add_pass(f"Seasonal decomposition (trend length={len(decomp.trend)})")
            else:
                results.add_fail("Seasonal decomposition", "Missing components")
        except Exception as e:
            results.add_fail("Seasonal decomposition", str(e))
        
        # Test Simple Exponential Smoothing
        try:
            forecast = mx.py_simple_exp_smoothing(data, 0.3, 10, 0.95)
            if hasattr(forecast, 'forecasts') and len(forecast.forecasts) == 10:
                results.add_pass(f"Simple Exp Smoothing (10 forecasts, CI={forecast.confidence_level})")
            else:
                results.add_fail("Simple Exp Smoothing", "Incorrect forecast length")
        except Exception as e:
            results.add_fail("Simple Exp Smoothing", str(e))
        
        # Test Holt Linear
        try:
            forecast = mx.py_holt_linear(data, 0.3, 0.1, 10, 0.95)
            if hasattr(forecast, 'forecasts') and len(forecast.forecasts) == 10:
                results.add_pass(f"Holt Linear (10 forecasts)")
            else:
                results.add_fail("Holt Linear", "Incorrect forecast length")
        except Exception as e:
            results.add_fail("Holt Linear", str(e))
        
        # Test Holt-Winters
        try:
            forecast = mx.py_holt_winters(data, 0.3, 0.1, 0.2, 12, 10, 'additive', 0.95)
            if hasattr(forecast, 'forecasts') and len(forecast.forecasts) == 10:
                results.add_pass(f"Holt-Winters (10 forecasts, period=12)")
            else:
                results.add_fail("Holt-Winters", "Incorrect forecast length")
        except Exception as e:
            results.add_fail("Holt-Winters", str(e))
        
        # Test with stock prices (random walk)
        try:
            df_stock = pd.read_csv(test_data_dir / "timeseries_stock_prices.csv")
            stock_data = df_stock['close_price'].tolist()
            
            adf_stock = mx.py_adf_test(stock_data, 10)
            if not adf_stock.is_stationary:
                results.add_pass("Stock prices non-stationary (expected)")
            else:
                print_warning("Stock prices showed as stationary (unexpected but possible)")
                results.add_pass("Stock prices stationarity test")
        except Exception as e:
            results.add_fail("Stock prices test", str(e))
    
    except ImportError as e:
        results.add_fail("Time series tests", f"Import error: {e}")
    except FileNotFoundError as e:
        results.add_fail("Time series tests", f"File not found: {e}")
    
    return results


def test_timeseries_plots():
    """Test time series plotting functionality."""
    print_section("6. Testing Time Series Plots")
    results = TestResults()
    
    try:
        import pandas as pd
        import mathemixx_core as mx
        import matplotlib.pyplot as plt
        sys.path.insert(0, str(Path(__file__).parent / "python"))
        import plots
        
        test_data_dir = Path(__file__).parent / "test-data"
        df = pd.read_csv(test_data_dir / "timeseries_monthly_sales.csv")
        data = df['sales'].tolist()
        
        if not hasattr(mx, 'py_acf'):
            print_warning("Phase 7 bindings not available - skipping TS plot tests")
            return results
        
        # Test ACF plot
        try:
            acf_vals = mx.py_acf(data, 20)
            fig, ax = plt.subplots()
            plots.plot_acf(acf_vals, ax=ax)
            plt.close(fig)
            results.add_pass("ACF plot")
        except Exception as e:
            results.add_fail("ACF plot", str(e))
        
        # Test PACF plot
        try:
            pacf_vals = mx.py_pacf(data, 20)
            fig, ax = plt.subplots()
            plots.plot_pacf(pacf_vals, ax=ax)
            plt.close(fig)
            results.add_pass("PACF plot")
        except Exception as e:
            results.add_fail("PACF plot", str(e))
        
        # Test ACF+PACF combined
        try:
            fig = plots.plot_acf_pacf(data, nlags=20)
            plt.close(fig)
            results.add_pass("ACF+PACF combined plot")
        except Exception as e:
            results.add_fail("ACF+PACF combined plot", str(e))
        
        # Test decomposition plot
        try:
            decomp = mx.py_seasonal_decompose(data, 12, 'additive')
            fig = plots.plot_decomposition(decomp)
            plt.close(fig)
            results.add_pass("Decomposition plot")
        except Exception as e:
            results.add_fail("Decomposition plot", str(e))
        
        # Test forecast plot
        try:
            forecast = mx.py_simple_exp_smoothing(data, 0.3, 10, 0.95)
            ax = plots.plot_forecast(data, forecast, n_history=30)
            plt.close(ax.figure)
            results.add_pass("Forecast plot")
        except Exception as e:
            results.add_fail("Forecast plot", str(e))
    
    except ImportError as e:
        results.add_fail("Time series plot tests", f"Import error: {e}")
    except FileNotFoundError as e:
        results.add_fail("Time series plot tests", f"File not found: {e}")
    
    return results


def main():
    """Main test runner."""
    print_header("MATHEMIX TEST DATA VERIFICATION")
    
    start_time = datetime.now()
    print_info(f"Started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print_info(f"Python: {sys.version.split()[0]}")
    print_info(f"Working directory: {Path.cwd()}")
    
    all_results = []
    
    # Run all test suites
    all_results.append(("Imports", test_imports()))
    all_results.append(("Data Loading", test_data_loading()))
    all_results.append(("Regression", test_regression_features()))
    all_results.append(("Visualization", test_visualization_features()))
    all_results.append(("Time Series", test_timeseries_features()))
    all_results.append(("TS Plots", test_timeseries_plots()))
    
    # Print overall summary
    print_header("OVERALL RESULTS")
    
    total_passed = sum(r.passed for _, r in all_results)
    total_failed = sum(r.failed for _, r in all_results)
    total_tests = sum(r.total for _, r in all_results)
    
    print(f"\n{Colors.BOLD}Test Suite Breakdown:{Colors.END}")
    print(f"{'Suite':<20} {'Total':<8} {'Passed':<8} {'Failed':<8} {'Rate':<8}")
    print("-" * 60)
    
    for suite_name, results in all_results:
        rate = f"{(results.passed/results.total*100) if results.total > 0 else 0:.1f}%"
        passed_str = f"{Colors.GREEN}{results.passed}{Colors.END}" if results.passed > 0 else "0"
        failed_str = f"{Colors.RED}{results.failed}{Colors.END}" if results.failed > 0 else "0"
        print(f"{suite_name:<20} {results.total:<8} {passed_str:<15} {failed_str:<15} {rate:<8}")
    
    print("-" * 60)
    success_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0
    passed_str = f"{Colors.GREEN}{total_passed}{Colors.END}"
    failed_str = f"{Colors.RED}{total_failed}{Colors.END}" if total_failed > 0 else "0"
    print(f"{'TOTAL':<20} {total_tests:<8} {passed_str:<15} {failed_str:<15} {success_rate:.1f}%")
    
    # Print all errors
    all_errors = []
    for suite_name, results in all_results:
        for test_name, error in results.errors:
            all_errors.append((suite_name, test_name, error))
    
    if all_errors:
        print_header("FAILED TESTS DETAILS")
        for i, (suite, test, error) in enumerate(all_errors, 1):
            print(f"\n{i}. [{suite}] {Colors.RED}{test}{Colors.END}")
            print(f"   {Colors.YELLOW}Error:{Colors.END} {error}")
    
    # Final status
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    print_header("TEST RUN COMPLETE")
    print_info(f"Completed at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print_info(f"Duration: {duration:.2f} seconds")
    
    if total_failed == 0:
        print(f"\n{Colors.BOLD}{Colors.GREEN}🎉 ALL TESTS PASSED! 🎉{Colors.END}\n")
        return 0
    else:
        print(f"\n{Colors.BOLD}{Colors.RED}⚠ {total_failed} TEST(S) FAILED{Colors.END}\n")
        
        # Print recommendations
        if any("Phase 7" in str(e) for _, _, e in all_errors):
            print_section("RECOMMENDATIONS")
            print_warning("Phase 7 bindings not available.")
            print_info("To fix:")
            print("  1. Close all Python processes")
            print("  2. Run: pip uninstall mathemixx-core -y")
            print("  3. Run: pip install -e .")
            print("  4. Run this script again\n")
        
        return 1


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print(f"\n\n{Colors.YELLOW}Test interrupted by user{Colors.END}")
        sys.exit(130)
    except Exception as e:
        print(f"\n{Colors.RED}FATAL ERROR:{Colors.END} {e}")
        traceback.print_exc()
        sys.exit(1)
