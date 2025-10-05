#!/usr/bin/env python3
"""Test the mathemixx_core Python bindings"""

import mathemixx_core as mx
import sys

def test_load_data():
    """Test loading CSV data"""
    print("Testing data loading...")
    dataset = mx.DataSet.from_csv("data/sample_regression.csv")
    print(f"✓ Loaded dataset from CSV")
    print(f"  Columns: {dataset.column_names()}")
    print(f"  Numeric columns: {dataset.numeric_columns()}")
    return dataset

def test_summary(dataset):
    """Test summary statistics"""
    print("\nTesting summary statistics...")
    summary = dataset.summarize()
    print(f"✓ Computed summary statistics for {len(summary)} variables:")
    for row in summary:
        print(f"  {row.variable}: mean={row.mean:.4f}, sd={row.sd:.4f}, min={row.min:.4f}, max={row.max:.4f}")

def test_regression(dataset):
    """Test OLS regression"""
    print("\nTesting OLS regression...")
    result = dataset.regress_ols("y", ["x1", "x2"], robust=True)
    
    print(f"✓ Regression completed:")
    print(f"  Dependent variable: {result.dependent}")
    print(f"  R-squared: {result.r_squared():.6f}")
    print(f"  Adjusted R-squared: {result.adj_r_squared():.6f}")
    print(f"  Observations: {result.nobs()}")
    
    print(f"\n  Coefficients:")
    for row in result.table():
        print(f"    {row.variable:12s}: β={row.coefficient:8.4f}, SE={row.std_error:.4f}, "
              f"t={row.t_value:6.3f}, p={row.p_value:.4f}")
    
    # Check if robust standard errors are available
    robust_se = result.robust_std_errors()
    if robust_se:
        print(f"\n  Robust standard errors available: {list(robust_se.keys())}")
    
    return result

def test_export(result):
    """Test exporting results"""
    print("\nTesting export functionality...")
    
    # Export to JSON
    json_str = result.to_json()
    print(f"✓ Exported to JSON ({len(json_str)} bytes)")
    
    # Export to CSV
    result.export_csv("test_output.csv")
    print(f"✓ Exported to CSV: test_output.csv")
    
    # Export to TeX
    result.export_tex("test_output.tex")
    print(f"✓ Exported to TeX: test_output.tex")

def main():
    print("=" * 60)
    print("MatheMixX Python Bindings Test")
    print("=" * 60)
    
    try:
        dataset = test_load_data()
        test_summary(dataset)
        result = test_regression(dataset)
        test_export(result)
        
        print("\n" + "=" * 60)
        print("✓ All tests passed successfully!")
        print("=" * 60)
        return 0
        
    except Exception as e:
        print(f"\n✗ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
