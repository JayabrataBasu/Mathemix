"""Quick test to verify Phase 7 bindings are available."""
import sys
import os

# Ensure we're using the local build
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'target', 'release'))

try:
    import mathemixx_core as core
    
    print("Testing Phase 7 Time Series Bindings:")
    print("=" * 50)
    
    # Test data
    data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    
    # Test basic operations
    print("\n1. Testing basic operations...")
    try:
        lagged = core.py_lag(data, 1)
        print(f"✓ py_lag: {len(lagged)} values")
    except AttributeError as e:
        print(f"✗ py_lag not found: {e}")
    
    try:
        diffed = core.py_diff(data, 1)
        print(f"✓ py_diff: {len(diffed)} values")
    except AttributeError as e:
        print(f"✗ py_diff not found: {e}")
    
    # Test moving averages
    print("\n2. Testing moving averages...")
    for func_name in ['py_sma', 'py_ema', 'py_wma']:
        try:
            func = getattr(core, func_name)
            result = func(data, 3)
            print(f"✓ {func_name}: {len(result)} values")
        except AttributeError:
            print(f"✗ {func_name} not found")
    
    # Test rolling operations
    print("\n3. Testing rolling operations...")
    for func_name in ['py_rolling_mean', 'py_rolling_std', 'py_rolling_min', 'py_rolling_max']:
        try:
            func = getattr(core, func_name)
            result = func(data, 3)
            print(f"✓ {func_name}: {len(result)} values")
        except AttributeError:
            print(f"✗ {func_name} not found")
    
    # Test autocorrelation
    print("\n4. Testing autocorrelation...")
    for func_name in ['py_acf', 'py_pacf', 'py_ljung_box_test']:
        try:
            func = getattr(core, func_name)
            if func_name == 'py_ljung_box_test':
                result = func(data, 5)
                print(f"✓ {func_name}: statistic={result['statistic']:.4f}")
            else:
                result = func(data, 5)
                print(f"✓ {func_name}: {len(result)} values")
        except AttributeError:
            print(f"✗ {func_name} not found")
    
    # Test stationarity
    print("\n5. Testing stationarity tests...")
    for func_name in ['py_adf_test', 'py_kpss_test']:
        try:
            func = getattr(core, func_name)
            if func_name == 'py_adf_test':
                result = func(data, 5)
            else:
                result = func(data, 5)
            print(f"✓ {func_name}: statistic={result.test_statistic:.4f}")
        except AttributeError:
            print(f"✗ {func_name} not found")
    
    # Test decomposition
    print("\n6. Testing decomposition...")
    try:
        result = core.py_seasonal_decompose(data * 2, 4, 'additive')
        print(f"✓ py_seasonal_decompose: trend length={len(result.trend)}")
    except AttributeError:
        print("✗ py_seasonal_decompose not found")
    
    # Test forecasting
    print("\n7. Testing forecasting...")
    for func_name in ['py_simple_exp_smoothing', 'py_holt_linear', 'py_holt_winters']:
        try:
            func = getattr(core, func_name)
            if func_name == 'py_simple_exp_smoothing':
                result = func(data, 0.3, 5, 0.95)
            elif func_name == 'py_holt_linear':
                result = func(data, 0.3, 0.1, 5, 0.95)
            else:
                result = func(data * 3, 0.3, 0.1, 0.2, 3, 5, 'additive', 0.95)
            print(f"✓ {func_name}: {len(result.forecasts)} forecasts")
        except AttributeError:
            print(f"✗ {func_name} not found")
    
    print("\n" + "=" * 50)
    print("Phase 7 bindings test complete!")
    
except ImportError as e:
    print(f"ERROR: Could not import mathemixx_core: {e}")
    print("\nPlease run: cargo build --release")
    sys.exit(1)
