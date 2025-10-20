"""
Phase 7: Time Series Analysis - Comprehensive Integration Tests
Tests all time series functionality including operations, ACF/PACF, 
stationarity tests, decomposition, and forecasting.
"""

import sys
import os
import pytest
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import the compiled module and Python wrapper
import mathemixx_core as core
from python.timeseries import (
    TimeSeriesAnalyzer,
    lag, diff, sma, ema, wma,
    acf, pacf,
    adf_test, kpss_test,
    seasonal_decompose,
    forecast_ses, forecast_holt, forecast_holt_winters
)

# Import rolling operations and ljung_box directly from core
rolling_mean = core.py_rolling_mean
rolling_std = core.py_rolling_std
rolling_min = core.py_rolling_min
rolling_max = core.py_rolling_max
ljung_box_test = core.py_ljung_box_test
simple_exp_smoothing = forecast_ses
holt_linear = forecast_holt
holt_winters = forecast_holt_winters


class TestTimeSeriesOperations:
    """Test basic time series operations."""
    
    def test_lag_operation(self):
        """Test lag operation."""
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        
        # Lag 1
        lagged = lag(data, 1)
        assert len(lagged) == 4
        assert lagged[0] == 2.0
        assert lagged[-1] == 5.0
        
        # Lag 2
        lagged = lag(data, 2)
        assert len(lagged) == 3
        assert lagged[0] == 3.0
    
    def test_diff_operation(self):
        """Test differencing operation."""
        data = [1.0, 3.0, 6.0, 10.0, 15.0]
        
        # First difference
        diff_data = diff(data, 1)
        assert len(diff_data) == 4
        assert all(abs(d - 2.0) < 1e-10 for d in diff_data[:2])  # First two diffs
        
        # Second difference
        diff2_data = diff(data, 2)
        assert len(diff2_data) == 3
    
    def test_moving_averages(self):
        """Test simple, exponential, and weighted moving averages."""
        data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        
        # SMA
        sma_vals = sma(data, 3)
        assert len(sma_vals) == 8
        assert abs(sma_vals[0] - 2.0) < 1e-10  # (1+2+3)/3
        assert abs(sma_vals[-1] - 9.0) < 1e-10  # (8+9+10)/3
        
        # EMA
        ema_vals = ema(data, 3)
        assert len(ema_vals) == 10
        assert ema_vals[0] == 1.0  # First value unchanged
        
        # WMA
        wma_vals = wma(data, 3)
        assert len(wma_vals) == 8
        assert wma_vals[0] > sma_vals[0]  # WMA gives more weight to recent
    
    def test_rolling_operations(self):
        """Test rolling statistics."""
        data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        
        # Rolling mean
        r_mean = rolling_mean(data, 3)
        assert len(r_mean) == 8
        assert abs(r_mean[0] - 2.0) < 1e-10
        
        # Rolling std
        r_std = rolling_std(data, 3)
        assert len(r_std) == 8
        assert r_std[0] > 0  # Should have variance
        
        # Rolling min/max
        r_min = rolling_min(data, 3)
        r_max = rolling_max(data, 3)
        assert len(r_min) == 8
        assert len(r_max) == 8
        assert r_min[0] == 1.0
        assert r_max[0] == 3.0


class TestAutocorrelation:
    """Test autocorrelation functions."""
    
    def test_acf_white_noise(self):
        """Test ACF on white noise."""
        np.random.seed(42)
        data = np.random.randn(100).tolist()
        
        acf_vals = acf(data, 10)
        assert len(acf_vals) == 11  # Lags 0 to 10
        assert abs(acf_vals[0] - 1.0) < 1e-10  # Lag 0 should be 1
        
        # Most ACF values should be close to 0 for white noise
        assert all(abs(v) < 0.3 for v in acf_vals[1:])
    
    def test_pacf_white_noise(self):
        """Test PACF on white noise."""
        np.random.seed(42)
        data = np.random.randn(100).tolist()
        
        pacf_vals = pacf(data, 10)
        assert len(pacf_vals) == 11
        assert abs(pacf_vals[0] - 1.0) < 1e-10  # Lag 0 should be 1
    
    def test_acf_ar1_process(self):
        """Test ACF on AR(1) process."""
        # Generate AR(1) process: x_t = 0.7 * x_{t-1} + e_t
        np.random.seed(42)
        n = 200
        data = [0.0]
        for _ in range(n - 1):
            data.append(0.7 * data[-1] + np.random.randn() * 0.5)
        
        acf_vals = acf(data, 20)
        # AR(1) should show exponential decay in ACF
        assert acf_vals[1] > acf_vals[5]  # Decay pattern
        assert acf_vals[5] > acf_vals[10]
    
    def test_ljung_box_test(self):
        """Test Ljung-Box test for autocorrelation."""
        # White noise should not reject null hypothesis
        np.random.seed(42)
        white_noise = np.random.randn(100).tolist()
        
        result = ljung_box_test(white_noise, 10)
        assert result['statistic'] >= 0
        assert 0 <= result['p_value'] <= 1
        assert result['degrees_of_freedom'] == 10


class TestStationarityTests:
    """Test stationarity tests."""
    
    def test_adf_stationary(self):
        """Test ADF on stationary series."""
        np.random.seed(42)
        stationary_data = np.random.randn(100).tolist()
        
        result = adf_test(stationary_data, max_lags=10)
        assert hasattr(result, 'test_statistic')
        assert hasattr(result, 'p_value')
        assert hasattr(result, 'is_stationary')
        assert 0 <= result.p_value <= 1
        
        # Should likely be stationary
        assert result.is_stationary is True
    
    def test_adf_nonstationary(self):
        """Test ADF on non-stationary series (random walk)."""
        np.random.seed(42)
        random_walk = np.cumsum(np.random.randn(100)).tolist()
        
        result = adf_test(random_walk, max_lags=10)
        # Random walk should be non-stationary
        assert result.is_stationary is False
    
    def test_kpss_stationary(self):
        """Test KPSS on stationary series."""
        np.random.seed(42)
        stationary_data = np.random.randn(100).tolist()
        
        result = kpss_test(stationary_data, nlags=10)
        assert hasattr(result, 'test_statistic')
        assert hasattr(result, 'p_value')
        assert hasattr(result, 'is_stationary')
        assert 0 <= result.p_value <= 1
        
        # Should be stationary (KPSS null hypothesis is stationarity)
        assert result.is_stationary is True
    
    def test_kpss_nonstationary(self):
        """Test KPSS on non-stationary series."""
        # Linear trend
        trend_data = [float(i) + np.random.randn() * 0.1 for i in range(100)]
        
        result = kpss_test(trend_data, nlags=10)
        # Should detect non-stationarity
        assert result.is_stationary is False


class TestDecomposition:
    """Test seasonal decomposition."""
    
    def test_additive_decomposition(self):
        """Test additive seasonal decomposition."""
        # Create synthetic seasonal data
        n = 120
        t = np.arange(n)
        trend = 0.05 * t + 10
        seasonal = 5 * np.sin(2 * np.pi * t / 12)
        noise = np.random.randn(n) * 0.5
        data = (trend + seasonal + noise).tolist()
        
        result = seasonal_decompose(data, period=12, model='additive')
        
        assert hasattr(result, 'trend')
        assert hasattr(result, 'seasonal')
        assert hasattr(result, 'residual')
        assert hasattr(result, 'observed')
        
        assert len(result.trend) == n
        assert len(result.seasonal) == n
        assert len(result.residual) == n
        assert len(result.observed) == n
        
        # Check that decomposition is reasonable
        # Trend should be roughly linear
        trend_diff = np.diff(result.trend[12:-12])  # Skip NaN regions
        assert np.std(trend_diff) < 1.0  # Should be relatively smooth
    
    def test_multiplicative_decomposition(self):
        """Test multiplicative seasonal decomposition."""
        n = 120
        t = np.arange(n)
        trend = 0.05 * t + 10
        seasonal = 1 + 0.2 * np.sin(2 * np.pi * t / 12)
        noise = 1 + np.random.randn(n) * 0.05
        data = (trend * seasonal * noise).tolist()
        
        result = seasonal_decompose(data, period=12, model='multiplicative')
        
        assert len(result.trend) == n
        assert len(result.seasonal) == n
        assert len(result.residual) == n


class TestForecasting:
    """Test forecasting methods."""
    
    def test_simple_exponential_smoothing(self):
        """Test Simple Exponential Smoothing."""
        # Generate level data
        np.random.seed(42)
        data = [10.0 + np.random.randn() for _ in range(50)]
        
        result = simple_exp_smoothing(data, alpha=0.3, horizon=10, confidence_level=0.95)
        
        assert hasattr(result, 'forecasts')
        assert hasattr(result, 'lower_bound')
        assert hasattr(result, 'upper_bound')
        assert hasattr(result, 'confidence_level')
        
        assert len(result.forecasts) == 10
        assert len(result.lower_bound) == 10
        assert len(result.upper_bound) == 10
        assert result.confidence_level == 0.95
        
        # Forecasts should be reasonable
        assert all(abs(f - 10.0) < 5.0 for f in result.forecasts)
        
        # Bounds should contain forecast
        for i in range(10):
            assert result.lower_bound[i] <= result.forecasts[i] <= result.upper_bound[i]
    
    def test_holt_linear(self):
        """Test Holt's Linear Trend method."""
        # Generate trending data
        data = [10.0 + 0.5 * i + np.random.randn() * 0.5 for i in range(50)]
        
        result = holt_linear(data, alpha=0.3, beta=0.1, horizon=10, confidence_level=0.95)
        
        assert len(result.forecasts) == 10
        assert len(result.lower_bound) == 10
        assert len(result.upper_bound) == 10
        
        # Should capture trend
        forecast_diffs = np.diff(result.forecasts)
        assert all(d > 0 for d in forecast_diffs)  # Should be increasing
    
    def test_holt_winters(self):
        """Test Holt-Winters method."""
        # Generate seasonal + trend data
        n = 60
        t = np.arange(n)
        trend = 0.1 * t + 10
        seasonal = 3 * np.sin(2 * np.pi * t / 12)
        data = (trend + seasonal + np.random.randn(n) * 0.5).tolist()
        
        result = holt_winters(
            data, 
            alpha=0.3, 
            beta=0.1, 
            gamma=0.2, 
            period=12, 
            horizon=12,
            confidence_level=0.95
        )
        
        assert len(result.forecasts) == 12
        assert len(result.lower_bound) == 12
        assert len(result.upper_bound) == 12
        
        # Forecasts should show seasonality
        # Check if forecasts vary (not constant)
        assert np.std(result.forecasts) > 0.5


class TestTimeSeriesAnalyzer:
    """Test TimeSeriesAnalyzer class (OOP interface)."""
    
    def test_analyzer_creation(self):
        """Test creating TimeSeriesAnalyzer."""
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        analyzer = TimeSeriesAnalyzer(data)
        
        assert analyzer.data == data
        assert len(analyzer) == 5
    
    def test_analyzer_operations(self):
        """Test analyzer operations."""
        data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        analyzer = TimeSeriesAnalyzer(data)
        
        # Test lag
        lagged = analyzer.lag(1)
        assert len(lagged) == 9
        
        # Test diff
        diffed = analyzer.diff(1)
        assert len(diffed) == 9
        
        # Test moving averages
        sma_vals = analyzer.sma(3)
        assert len(sma_vals) == 8
        
        ema_vals = analyzer.ema(3)
        assert len(ema_vals) == 10
    
    def test_analyzer_autocorrelation(self):
        """Test analyzer autocorrelation methods."""
        np.random.seed(42)
        data = np.random.randn(100).tolist()
        analyzer = TimeSeriesAnalyzer(data)
        
        acf_vals = analyzer.acf(10)
        assert len(acf_vals) == 11
        
        pacf_vals = analyzer.pacf(10)
        assert len(pacf_vals) == 11
        
        lb_result = analyzer.ljung_box_test(10)
        assert 'statistic' in lb_result
    
    def test_analyzer_stationarity(self):
        """Test analyzer stationarity tests."""
        np.random.seed(42)
        data = np.random.randn(100).tolist()
        analyzer = TimeSeriesAnalyzer(data)
        
        adf_result = analyzer.adf_test(10)
        assert hasattr(adf_result, 'is_stationary')
        
        kpss_result = analyzer.kpss_test(10)
        assert hasattr(kpss_result, 'is_stationary')
    
    def test_analyzer_decomposition(self):
        """Test analyzer decomposition."""
        n = 120
        t = np.arange(n)
        data = (10 + 0.05 * t + 5 * np.sin(2 * np.pi * t / 12) + np.random.randn(n) * 0.5).tolist()
        analyzer = TimeSeriesAnalyzer(data)
        
        result = analyzer.decompose(period=12, model='additive')
        assert len(result.trend) == n
        assert len(result.seasonal) == n
    
    def test_analyzer_forecasting(self):
        """Test analyzer forecasting methods."""
        data = [10.0 + np.random.randn() for _ in range(50)]
        analyzer = TimeSeriesAnalyzer(data)
        
        # SES
        ses_result = analyzer.forecast_ses(alpha=0.3, horizon=10)
        assert len(ses_result.forecasts) == 10
        
        # Holt
        holt_result = analyzer.forecast_holt(alpha=0.3, beta=0.1, horizon=10)
        assert len(holt_result.forecasts) == 10
        
        # Holt-Winters
        n = 60
        seasonal_data = [10 + 3 * np.sin(2 * np.pi * i / 12) for i in range(n)]
        analyzer2 = TimeSeriesAnalyzer(seasonal_data)
        hw_result = analyzer2.forecast_holt_winters(
            alpha=0.3, beta=0.1, gamma=0.2, period=12, horizon=12
        )
        assert len(hw_result.forecasts) == 12


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_data(self):
        """Test with empty data."""
        with pytest.raises(Exception):
            lag([], 1)
    
    def test_insufficient_data(self):
        """Test with insufficient data."""
        data = [1.0, 2.0]
        
        # Should handle gracefully
        with pytest.raises(Exception):
            acf(data, 10)  # Not enough data for 10 lags
    
    def test_invalid_parameters(self):
        """Test with invalid parameters."""
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        
        # Invalid lag
        with pytest.raises(Exception):
            lag(data, -1)
        
        # Invalid window size
        with pytest.raises(Exception):
            sma(data, 0)


if __name__ == "__main__":
    print("Running Phase 7 Time Series Tests...")
    pytest.main([__file__, "-v", "--tb=short"])
