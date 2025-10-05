#!/usr/bin/env python3
"""Quick test to verify desktop app imports work correctly."""
import sys
from pathlib import Path

print("Testing imports...")

# Test 1: Import mathemixx_core bindings (from installed wheel)
# DO NOT add python/ to path yet - import the installed wheel first
try:
    import mathemixx_core as mx
    print("✓ mathemixx_core imported successfully from installed wheel")
    print(f"  Module file: {mx.__file__}")
except ImportError as e:
    print(f"✗ Failed to import mathemixx_core: {e}")
    print("  Make sure the wheel is installed")
    sys.exit(1)

# NOW add the python directory for the desktop app
sys.path.insert(0, str(Path(__file__).parent / "python"))

# Test 2: Load sample data
try:
    dataset = mx.DataSet.from_csv("data/sample_regression.csv")
    print(f"✓ Loaded dataset with columns: {dataset.column_names()}")
except Exception as e:
    print(f"✗ Failed to load dataset: {e}")
    sys.exit(1)

# Test 3: Run regression
try:
    result = dataset.regress_ols("y", ["x1", "x2"], robust=True)
    print(f"✓ Regression completed:")
    print(f"  R² = {result.r_squared():.6f}")
    print(f"  Coefficients: {len(result.coefficients)} parameters")
except Exception as e:
    print(f"✗ Regression failed: {e}")
    sys.exit(1)

# Test 4: Import desktop app
try:
    from mathemixx_desktop import launch
    print("✓ Desktop app module imported successfully")
except ImportError as e:
    print(f"✗ Failed to import desktop app: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("All integration tests passed!")
print("=" * 60)
print("\nTo launch the desktop app, run:")
print("  python run_desktop.py")
print("or double-click:")
print("  run_mathemixx.bat")
