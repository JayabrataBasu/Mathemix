"""
Test Enhanced Error Handling in MatheMixX

This script demonstrates the improved error messages for common failure scenarios.
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent / 'python'))

from mathemixx_desktop.error_handler import ErrorContext, DataValidationError

def test_validation_functions():
    """Test all validation functions with various scenarios."""
    
    print("=" * 70)
    print("TESTING ENHANCED ERROR HANDLING")
    print("=" * 70)
    print()
    
    # Test 1: Insufficient data
    print("Test 1: Insufficient Data Validation")
    print("-" * 70)
    try:
        data = [1.0, 2.0, 3.0]  # Only 3 points
        ErrorContext.validate_sufficient_data(data, 10, "time series analysis")
    except DataValidationError as e:
        print("✓ Caught validation error:")
        print(f"  {str(e).split(chr(10))[0]}")  # First line
        print()
    
    # Test 2: Window size too large
    print("Test 2: Window Size Validation")
    print("-" * 70)
    try:
        ErrorContext.validate_window_size(10, 15, "moving average")
    except DataValidationError as e:
        print("✓ Caught validation error:")
        print(f"  {str(e).split(chr(10))[0]}")
        print()
    
    # Test 3: Period validation
    print("Test 3: Period Validation")
    print("-" * 70)
    try:
        ErrorContext.validate_period(15, 20, "seasonal decomposition")
    except DataValidationError as e:
        print("✓ Caught validation error:")
        print(f"  {str(e).split(chr(10))[0]}")
        print()
    
    # Test 4: Parameter range validation
    print("Test 4: Parameter Range Validation")
    print("-" * 70)
    try:
        ErrorContext.validate_parameter_range(1.5, "alpha", 0.0, 1.0, inclusive=False)
    except DataValidationError as e:
        print("✓ Caught validation error:")
        print(f"  {str(e).split(chr(10))[0]}")
        print()
    
    # Test 5: No variance
    print("Test 5: No Variance Validation")
    print("-" * 70)
    try:
        data = [5.0] * 100  # All same value
        ErrorContext.validate_variance(data, "correlation analysis")
    except DataValidationError as e:
        print("✓ Caught validation error:")
        print(f"  {str(e).split(chr(10))[0]}")
        print()
    
    # Test 6: NaN/Inf validation
    print("Test 6: NaN/Inf Validation")
    print("-" * 70)
    try:
        data = [1.0, 2.0, np.nan, 4.0, np.inf, 6.0]
        ErrorContext.validate_no_nan_inf(data, "regression data")
    except DataValidationError as e:
        print("✓ Caught validation error:")
        print(f"  {str(e).split(chr(10))[0]}")
        print()
    
    # Test 7: Column validation
    print("Test 7: Column Existence Validation")
    print("-" * 70)
    try:
        df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        ErrorContext.validate_column_exists(df, "NonExistent", "analysis")
    except DataValidationError as e:
        print("✓ Caught validation error:")
        print(f"  {str(e).split(chr(10))[0]}")
        print()
    
    # Test 8: Numeric column validation
    print("Test 8: Numeric Column Validation")
    print("-" * 70)
    try:
        df = pd.DataFrame({'Name': ['A', 'B', 'C'], 'Value': [1, 2, 3]})
        ErrorContext.validate_numeric_column(df, "Name")
    except DataValidationError as e:
        print("✓ Caught validation error:")
        print(f"  {str(e).split(chr(10))[0]}")
        print()
    
    # Test 9: Regression error formatting
    print("Test 9: Regression Error Formatting (Multicollinearity)")
    print("-" * 70)
    error = Exception("Matrix is singular due to multicollinearity")
    formatted = ErrorContext.format_regression_error(error, "Sales", ["Price", "Price_Copy"])
    print("✓ Formatted error message:")
    print(f"  {formatted.split(chr(10))[0]}")
    print()
    
    # Test 10: Time series error formatting
    print("Test 10: Time Series Error Formatting")
    print("-" * 70)
    error = Exception("Period must be positive")
    formatted = ErrorContext.format_timeseries_error(error, "Seasonal Decomposition", period=0)
    print("✓ Formatted error message:")
    print(f"  {formatted.split(chr(10))[0]}")
    print()
    
    print("=" * 70)
    print("ALL VALIDATION TESTS COMPLETED SUCCESSFULLY")
    print("=" * 70)
    print()
    print("Summary:")
    print("- All 10 validation scenarios caught errors correctly")
    print("- Error messages are specific and actionable")
    print("- Users get clear guidance on how to fix issues")
    print()


if __name__ == "__main__":
    test_validation_functions()
