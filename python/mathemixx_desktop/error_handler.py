"""
Enhanced error handling for MatheMixX Desktop.

Provides detailed, user-friendly error messages with specific causes and solutions.
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Any


class DataValidationError(Exception):
    """Raised when data validation fails with specific reason."""
    pass


class ErrorContext:
    """Context manager for providing detailed error messages."""
    
    @staticmethod
    def validate_data_loaded(data, context: str = "this operation") -> None:
        """Validate that data is loaded."""
        if data is None or (hasattr(data, '__len__') and len(data) == 0):
            raise DataValidationError(
                f"No data loaded for {context}.\n\n"
                f"Please load a CSV file first using the 'Load CSV' button."
            )
    
    @staticmethod
    def validate_column_exists(df: pd.DataFrame, column: str, purpose: str = "analysis") -> None:
        """Validate that a column exists in the dataframe."""
        if column not in df.columns:
            available = ', '.join(df.columns[:10])
            if len(df.columns) > 10:
                available += f"... ({len(df.columns)} total)"
            raise DataValidationError(
                f"Column '{column}' not found in dataset.\n\n"
                f"Available columns: {available}\n\n"
                f"Required for: {purpose}"
            )
    
    @staticmethod
    def validate_numeric_column(df: pd.DataFrame, column: str) -> None:
        """Validate that a column contains numeric data."""
        if column not in df.columns:
            ErrorContext.validate_column_exists(df, column)
        
        if not pd.api.types.is_numeric_dtype(df[column]):
            dtype = df[column].dtype
            raise DataValidationError(
                f"Column '{column}' must contain numeric data.\n\n"
                f"Current type: {dtype}\n\n"
                f"Please select a numeric column or convert the data to numeric format."
            )
    
    @staticmethod
    def validate_sufficient_data(data: List[float], min_points: int, operation: str) -> None:
        """Validate that there are enough data points."""
        if len(data) < min_points:
            raise DataValidationError(
                f"Insufficient data for {operation}.\n\n"
                f"Required: At least {min_points} data points\n"
                f"Available: {len(data)} data points\n\n"
                f"Please load more data or choose a different operation."
            )
    
    @staticmethod
    def validate_window_size(data_length: int, window: int, operation: str) -> None:
        """Validate window size for moving averages and rolling operations."""
        if window <= 0:
            raise DataValidationError(
                f"Window size must be positive.\n\n"
                f"Current value: {window}\n"
                f"Required: window > 0"
            )
        
        if window > data_length:
            raise DataValidationError(
                f"Window size too large for {operation}.\n\n"
                f"Window size: {window}\n"
                f"Data points: {data_length}\n\n"
                f"The window must be smaller than or equal to the number of data points.\n"
                f"Try reducing the window size."
            )
    
    @staticmethod
    def validate_period(period: int, data_length: int, operation: str) -> None:
        """Validate seasonal period."""
        if period <= 1:
            raise DataValidationError(
                f"Period must be greater than 1 for {operation}.\n\n"
                f"Current value: {period}\n"
                f"Common periods: 12 (monthly with yearly seasonality), 4 (quarterly), 7 (daily with weekly pattern)"
            )
        
        if period > data_length // 2:
            raise DataValidationError(
                f"Period too large for available data in {operation}.\n\n"
                f"Period: {period}\n"
                f"Data points: {data_length}\n\n"
                f"For reliable seasonal decomposition, you need at least 2 complete cycles.\n"
                f"Try reducing the period or loading more data."
            )
    
    @staticmethod
    def validate_parameter_range(value: float, param_name: str, min_val: Optional[float] = None, 
                                 max_val: Optional[float] = None, inclusive: bool = True) -> None:
        """Validate that a parameter is within valid range."""
        if min_val is not None:
            if inclusive and value < min_val:
                raise DataValidationError(
                    f"Parameter '{param_name}' is too small.\n\n"
                    f"Current value: {value}\n"
                    f"Minimum allowed: {min_val}\n\n"
                    f"Please adjust the parameter value."
                )
            elif not inclusive and value <= min_val:
                raise DataValidationError(
                    f"Parameter '{param_name}' must be greater than {min_val}.\n\n"
                    f"Current value: {value}\n\n"
                    f"Please adjust the parameter value."
                )
        
        if max_val is not None:
            if inclusive and value > max_val:
                raise DataValidationError(
                    f"Parameter '{param_name}' is too large.\n\n"
                    f"Current value: {value}\n"
                    f"Maximum allowed: {max_val}\n\n"
                    f"Please adjust the parameter value."
                )
            elif not inclusive and value >= max_val:
                raise DataValidationError(
                    f"Parameter '{param_name}' must be less than {max_val}.\n\n"
                    f"Current value: {value}\n\n"
                    f"Please adjust the parameter value."
                )
    
    @staticmethod
    def validate_no_nan_inf(data: List[float], context: str = "data") -> None:
        """Validate that data doesn't contain NaN or Inf values."""
        arr = np.array(data)
        nan_count = np.isnan(arr).sum()
        inf_count = np.isinf(arr).sum()
        
        if nan_count > 0 or inf_count > 0:
            issues = []
            if nan_count > 0:
                issues.append(f"{nan_count} NaN (missing) values")
            if inf_count > 0:
                issues.append(f"{inf_count} infinite values")
            
            raise DataValidationError(
                f"Invalid values detected in {context}.\n\n"
                f"Issues found: {', '.join(issues)}\n\n"
                f"Please clean the data by:\n"
                f"- Removing rows with missing values\n"
                f"- Filling missing values with mean/median\n"
                f"- Checking for calculation errors that produce infinity"
            )
    
    @staticmethod
    def validate_variance(data: List[float], operation: str) -> None:
        """Validate that data has variance (not all same values)."""
        arr = np.array(data)
        # Remove NaN for variance check
        clean_arr = arr[~np.isnan(arr)]
        
        if len(clean_arr) == 0:
            raise DataValidationError(
                f"No valid data points for {operation}.\n\n"
                f"All values are NaN (missing)."
            )
        
        if np.var(clean_arr) == 0:
            unique_val = clean_arr[0]
            raise DataValidationError(
                f"No variance in data for {operation}.\n\n"
                f"All {len(clean_arr)} data points have the same value: {unique_val}\n\n"
                f"This operation requires data with variation.\n"
                f"Please check your data or select a different column."
            )
    
    @staticmethod
    def format_regression_error(error: Exception, dependent: str, independents: List[str]) -> str:
        """Format regression-specific error messages."""
        error_str = str(error).lower()
        
        # Singular matrix / perfect multicollinearity
        if 'singular' in error_str or 'multicollinearity' in error_str:
            return (
                f"Regression failed: Multicollinearity detected.\n\n"
                f"Dependent variable: {dependent}\n"
                f"Independent variables: {', '.join(independents)}\n\n"
                f"Possible causes:\n"
                f"- One or more independent variables are perfectly correlated\n"
                f"- One variable is a linear combination of others\n"
                f"- Insufficient variation in the data\n\n"
                f"Solutions:\n"
                f"- Remove one of the correlated variables\n"
                f"- Check for duplicate or derived columns\n"
                f"- Ensure sufficient data variation"
            )
        
        # Not enough observations
        if 'insufficient' in error_str or 'observations' in error_str:
            return (
                f"Regression failed: Insufficient data.\n\n"
                f"You need more observations than variables.\n"
                f"Variables: {len(independents) + 1} (including intercept)\n\n"
                f"Solutions:\n"
                f"- Load more data rows\n"
                f"- Use fewer independent variables"
            )
        
        # Generic regression error
        return (
            f"Regression failed.\n\n"
            f"Dependent: {dependent}\n"
            f"Independent: {', '.join(independents)}\n\n"
            f"Error: {error}\n\n"
            f"Common causes:\n"
            f"- Non-numeric data in selected columns\n"
            f"- Missing values (NaN) in the data\n"
            f"- Perfect correlation between variables\n"
            f"- Insufficient data points"
        )
    
    @staticmethod
    def format_timeseries_error(error: Exception, operation: str, **params) -> str:
        """Format time series-specific error messages."""
        error_str = str(error).lower()
        
        # Empty or insufficient data
        if 'empty' in error_str or 'length' in error_str or 'size' in error_str:
            min_points = params.get('min_points', 'unknown')
            return (
                f"{operation} failed: Insufficient data.\n\n"
                f"Minimum required: {min_points} data points\n\n"
                f"Solution: Load a dataset with more observations."
            )
        
        # Period-related errors
        if 'period' in error_str:
            period = params.get('period', 'unknown')
            return (
                f"{operation} failed: Invalid period.\n\n"
                f"Period: {period}\n\n"
                f"The period must be:\n"
                f"- Greater than 1\n"
                f"- Less than half the data length\n"
                f"- Appropriate for your data frequency (e.g., 12 for monthly)"
            )
        
        # Stationarity test errors
        if operation in ['ADF Test', 'KPSS Test', 'Ljung-Box Test']:
            return (
                f"{operation} failed.\n\n"
                f"Error: {error}\n\n"
                f"Possible causes:\n"
                f"- Insufficient data points (need at least 10-20)\n"
                f"- All values are identical (no variance)\n"
                f"- Data contains NaN or infinite values\n\n"
                f"Solutions:\n"
                f"- Check data quality\n"
                f"- Ensure sufficient observations\n"
                f"- Remove or fill missing values"
            )
        
        # Forecasting errors
        if 'forecast' in operation.lower():
            return (
                f"{operation} failed.\n\n"
                f"Error: {error}\n\n"
                f"Common causes:\n"
                f"- Smoothing parameters out of range (must be 0 < α,β,γ ≤ 1)\n"
                f"- Period doesn't match data seasonality\n"
                f"- Insufficient historical data\n"
                f"- Data contains NaN or infinite values\n\n"
                f"Solutions:\n"
                f"- Adjust smoothing parameters\n"
                f"- Verify period matches data pattern\n"
                f"- Clean data of missing/invalid values"
            )
        
        # Generic time series error
        return (
            f"{operation} failed.\n\n"
            f"Error: {error}\n\n"
            f"Please check:\n"
            f"- Data is loaded and numeric\n"
            f"- Parameters are within valid ranges\n"
            f"- Sufficient data points for the operation"
        )
    
    @staticmethod
    def format_plot_error(error: Exception, plot_type: str) -> str:
        """Format plotting error messages."""
        error_str = str(error).lower()
        
        if 'empty' in error_str or 'no data' in error_str:
            return (
                f"{plot_type} failed: No data to plot.\n\n"
                f"Please select valid columns with numeric data."
            )
        
        if 'dimension' in error_str or 'shape' in error_str:
            return (
                f"{plot_type} failed: Data dimension mismatch.\n\n"
                f"Error: {error}\n\n"
                f"This plot type requires specific data structure.\n"
                f"Please check that selected columns are compatible."
            )
        
        return (
            f"{plot_type} failed.\n\n"
            f"Error: {error}\n\n"
            f"Please check:\n"
            f"- Selected columns contain numeric data\n"
            f"- Data has no missing values\n"
            f"- Column selection is appropriate for this plot type"
        )


__all__ = ['ErrorContext', 'DataValidationError']
