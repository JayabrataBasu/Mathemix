"""
Time Series Analysis Module for MatheMixX

Provides comprehensive time series analysis capabilities including:
- Basic operations (lag, diff, moving averages, rolling statistics)
- Autocorrelation analysis (ACF, PACF, Ljung-Box test)
- Stationarity tests (ADF, KPSS)
- Seasonal decomposition (additive/multiplicative)
- Forecasting methods (Simple Exponential Smoothing, Holt, Holt-Winters)
"""

from typing import Optional, Tuple, List
import mathemixx_core as core


class TimeSeriesAnalyzer:
    """
    High-level interface for time series analysis.
    
    Example:
        >>> import pandas as pd
        >>> from timeseries import TimeSeriesAnalyzer
        >>> 
        >>> data = pd.read_csv('sales.csv')['sales'].tolist()
        >>> ts = TimeSeriesAnalyzer(data)
        >>> 
        >>> # Check stationarity
        >>> adf_result = ts.adf_test()
        >>> print(f"Is stationary: {adf_result.is_stationary}")
        >>> 
        >>> # Decompose
        >>> decomp = ts.decompose(period=12)
        >>> 
        >>> # Forecast
        >>> forecast = ts.forecast_holt_winters(period=12, horizon=12)
    """
    
    def __init__(self, data: List[float]):
        """Initialize with time series data."""
        self.data = data
    
    # Basic Operations
    def lag(self, periods: int = 1) -> List[float]:
        """Lag the series by specified periods."""
        return core.py_lag(self.data, periods)
    
    def diff(self, periods: int = 1) -> List[float]:
        """Difference the series."""
        return core.py_diff(self.data, periods)
    
    def sma(self, window: int) -> List[float]:
        """Simple Moving Average."""
        return core.py_sma(self.data, window)
    
    def ema(self, window: int) -> List[float]:
        """Exponential Moving Average."""
        return core.py_ema(self.data, window)
    
    def wma(self, window: int) -> List[float]:
        """Weighted Moving Average."""
        return core.py_wma(self.data, window)
    
    def rolling_mean(self, window: int) -> List[float]:
        """Rolling mean."""
        return core.py_rolling_mean(self.data, window)
    
    def rolling_std(self, window: int) -> List[float]:
        """Rolling standard deviation."""
        return core.py_rolling_std(self.data, window)
    
    def rolling_min(self, window: int) -> List[float]:
        """Rolling minimum."""
        return core.py_rolling_min(self.data, window)
    
    def rolling_max(self, window: int) -> List[float]:
        """Rolling maximum."""
        return core.py_rolling_max(self.data, window)
    
    # Autocorrelation
    def acf(self, nlags: int = 20) -> List[float]:
        """
        Autocorrelation Function.
        
        Args:
            nlags: Number of lags to compute
            
        Returns:
            List of ACF values for lags 0 to nlags
        """
        return core.py_acf(self.data, nlags)
    
    def pacf(self, nlags: int = 20) -> List[float]:
        """
        Partial Autocorrelation Function.
        
        Args:
            nlags: Number of lags to compute
            
        Returns:
            List of PACF values for lags 0 to nlags
        """
        return core.py_pacf(self.data, nlags)
    
    def ljung_box_test(self, lags: int = 10) -> Tuple[float, float]:
        """
        Ljung-Box test for autocorrelation.
        
        Returns:
            (test_statistic, p_value)
        """
        return core.py_ljung_box_test(self.data, lags)
    
    # Stationarity Tests
    def adf_test(self, max_lags: Optional[int] = None):
        """
        Augmented Dickey-Fuller test for stationarity.
        
        H0: Series has a unit root (non-stationary)
        H1: Series is stationary
        
        Returns:
            ADFResult with test_statistic, p_value, and is_stationary
        """
        return core.py_adf_test(self.data, max_lags)
    
    def kpss_test(self, lags: Optional[int] = None):
        """
        KPSS test for stationarity.
        
        H0: Series is stationary
        H1: Series has a unit root (non-stationary)
        
        Returns:
            KPSSResult with test_statistic, p_value, and is_stationary
        """
        return core.py_kpss_test(self.data, lags)
    
    # Decomposition
    def decompose(self, period: int, model: str = "additive"):
        """
        Seasonal decomposition.
        
        Args:
            period: Seasonal period (e.g., 12 for monthly data with yearly seasonality)
            model: "additive" or "multiplicative"
            
        Returns:
            DecompositionResult with trend, seasonal, residual, and observed components
        """
        return core.py_seasonal_decompose(self.data, period, model)
    
    # Forecasting
    def forecast_ses(
        self,
        alpha: float = 0.3,
        horizon: int = 12,
        confidence: float = 0.95,
        confidence_level: Optional[float] = None,
    ):
        """
        Simple Exponential Smoothing forecast.
        
        Args:
            alpha: Smoothing parameter (0 < alpha <= 1)
            horizon: Number of periods to forecast
            confidence: Confidence level (e.g., 0.95 for 95%)
            
        Returns:
            ForecastResult with forecasts, lower_bound, upper_bound
        """
        conf = confidence_level if confidence_level is not None else confidence
        return core.py_simple_exp_smoothing(self.data, alpha, horizon, conf)
    
    def forecast_holt(
        self,
        alpha: float = 0.3,
        beta: float = 0.1,
        horizon: int = 12,
        confidence: float = 0.95,
        confidence_level: Optional[float] = None,
    ):
        """
        Holt's Linear Trend forecast.
        
        Args:
            alpha: Level smoothing parameter
            beta: Trend smoothing parameter
            horizon: Number of periods to forecast
            confidence: Confidence level
            
        Returns:
            ForecastResult with forecasts, lower_bound, upper_bound
        """
        conf = confidence_level if confidence_level is not None else confidence
        return core.py_holt_linear(self.data, alpha, beta, horizon, conf)
    
    def forecast_holt_winters(
        self,
        period: int = 12,
        alpha: float = 0.3,
        beta: float = 0.1,
        gamma: float = 0.2,
        horizon: int = 12,
        seasonal_type: str = "additive",
        confidence: float = 0.95,
        confidence_level: Optional[float] = None,
    ):
        """
        Holt-Winters seasonal forecast.
        
        Args:
            period: Seasonal period
            alpha: Level smoothing parameter
            beta: Trend smoothing parameter
            gamma: Seasonal smoothing parameter
            horizon: Number of periods to forecast
            seasonal_type: "additive" or "multiplicative"
            confidence: Confidence level
            
        Returns:
            ForecastResult with forecasts, lower_bound, upper_bound
        """
        conf = confidence_level if confidence_level is not None else confidence
        return core.py_holt_winters(
            self.data, alpha, beta, gamma, period, horizon, seasonal_type, conf
        )


# Convenience functions (functional API)

def lag(data: List[float], periods: int = 1) -> List[float]:
    """Lag a time series by specified periods."""
    return core.py_lag(data, periods)


def diff(data: List[float], periods: int = 1) -> List[float]:
    """Difference a time series."""
    return core.py_diff(data, periods)


def sma(data: List[float], window: int) -> List[float]:
    """Simple Moving Average."""
    return core.py_sma(data, window)


def ema(data: List[float], window: int) -> List[float]:
    """Exponential Moving Average."""
    return core.py_ema(data, window)


def wma(data: List[float], window: int) -> List[float]:
    """Weighted Moving Average."""
    return core.py_wma(data, window)


def acf(data: List[float], nlags: int = 20) -> List[float]:
    """Autocorrelation Function."""
    return core.py_acf(data, nlags)


def pacf(data: List[float], nlags: int = 20) -> List[float]:
    """Partial Autocorrelation Function."""
    return core.py_pacf(data, nlags)


def adf_test(data: List[float], max_lags: Optional[int] = None):
    """Augmented Dickey-Fuller test for stationarity."""
    return core.py_adf_test(data, max_lags)


def kpss_test(data: List[float], lags: Optional[int] = None):
    """KPSS test for stationarity."""
    return core.py_kpss_test(data, lags)


def seasonal_decompose(data: List[float], period: int, model: str = "additive"):
    """Seasonal decomposition."""
    return core.py_seasonal_decompose(data, period, model)


def forecast_ses(
    data: List[float],
    alpha: float = 0.3,
    horizon: int = 12,
    confidence: float = 0.95,
    confidence_level: Optional[float] = None,
):
    """Simple Exponential Smoothing forecast."""
    conf = confidence_level if confidence_level is not None else confidence
    return core.py_simple_exp_smoothing(data, alpha, horizon, conf)


def forecast_holt(
    data: List[float],
    alpha: float = 0.3,
    beta: float = 0.1,
    horizon: int = 12,
    confidence: float = 0.95,
    confidence_level: Optional[float] = None,
):
    """Holt's Linear Trend forecast."""
    conf = confidence_level if confidence_level is not None else confidence
    return core.py_holt_linear(data, alpha, beta, horizon, conf)


def forecast_holt_winters(
    data: List[float],
    period: int = 12,
    alpha: float = 0.3,
    beta: float = 0.1,
    gamma: float = 0.2,
    horizon: int = 12,
    seasonal_type: str = "additive",
    confidence: float = 0.95,
    confidence_level: Optional[float] = None,
):
    """Holt-Winters seasonal forecast."""
    conf = confidence_level if confidence_level is not None else confidence
    return core.py_holt_winters(
        data, alpha, beta, gamma, period, horizon, seasonal_type, conf
    )
