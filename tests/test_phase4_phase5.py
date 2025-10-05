"""
Comprehensive test script for Phase 4 (Data Manipulation) and Phase 5 (Statistical Methods)
"""
import mathemixx_core as mx
import sys
import traceback

def print_section(title):
    """Print a formatted section header"""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)

def test_column_info(ds):
    """Test Phase 4: Column information"""
    print_section("Phase 4: Column Information")
    try:
        info = ds.column_info()
        print(f"✓ Found {len(info)} columns:")
        for col in info:
            print(f"  - {col.name}: {col.dtype} ({col.total_count - col.null_count} non-null, {col.null_count} null)")
        
        # Test is_numeric_column
        print("\n✓ Numeric column checks:")
        for col in info:
            is_num = ds.is_numeric_column(col.name)
            print(f"  - {col.name}: {'numeric' if is_num else 'non-numeric'}")
        
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        traceback.print_exc()
        return False

def test_data_manipulation(ds):
    """Test Phase 4: Data manipulation operations"""
    print_section("Phase 4: Data Manipulation")
    try:
        # Test select_numeric
        numeric_ds = ds.select_numeric()
        print(f"✓ select_numeric: {len(numeric_ds.column_names())} numeric columns")
        
        # Test rename_column
        col_names = ds.column_names()
        if len(col_names) > 0:
            old_name = col_names[0]
            renamed = ds.rename_column(old_name, f"{old_name}_renamed")
            print(f"✓ rename_column: '{old_name}' -> '{old_name}_renamed'")
            print(f"  New columns: {renamed.column_names()[:5]}...")
        
        # Test add_column_transform (use first numeric column)
        numeric_cols = [c for c in col_names if ds.is_numeric_column(c)]
        if len(numeric_cols) > 0:
            test_col = numeric_cols[0]
            print(f"\n✓ add_column_transform tests (using '{test_col}'):")
            transforms = [
                ("log", f"{test_col}_log"),
                ("square", f"{test_col}_squared"),
                ("sqrt", f"{test_col}_sqrt"),
                ("standardize", f"{test_col}_std"),
                ("center", f"{test_col}_centered"),
            ]
            for trans_type, target in transforms:
                transformed = ds.add_column_transform(test_col, target, trans_type)
                print(f"  - {trans_type}: created column '{target}'")
        
        # Test filter_rows
        if len(numeric_cols) > 0:
            test_col = numeric_cols[0]
            print(f"\n✓ filter_rows tests (using '{test_col}'):")
            conditions = [
                ("gt", 5.0, ">5.0"),
                ("lt", 7.0, "<7.0"),
                ("ge", 6.0, ">=6.0"),
                ("le", 4.0, "<=4.0"),
            ]
            for cond, value, desc in conditions:
                filtered = ds.filter_rows(test_col, cond, value)
                print(f"  - {test_col} {desc}: {filtered.n_rows()} rows")
        
        # Test drop_columns
        if len(ds.column_names()) > 2:
            cols_to_drop = [ds.column_names()[0]]
            dropped = ds.drop_columns(cols_to_drop)
            print(f"\n✓ drop_columns: dropped {cols_to_drop}, {len(dropped.column_names())} columns remaining")
        
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        traceback.print_exc()
        return False

def test_correlation(ds):
    """Test Phase 5: Correlation analysis"""
    print_section("Phase 5: Correlation Analysis")
    try:
        # Test Pearson correlation
        corr_pearson = ds.correlation(None, "pearson")
        print(f"✓ Pearson correlation matrix:")
        print(f"  Variables: {corr_pearson.variables}")
        print(f"  Matrix shape: {len(corr_pearson.variables)} x {len(corr_pearson.variables)}")
        if len(corr_pearson.variables) >= 2:
            col1, col2 = corr_pearson.variables[0], corr_pearson.variables[1]
            print(f"  Sample: corr({col1}, {col2}) = {corr_pearson.matrix[0][1]:.4f}")
        
        # Test Spearman correlation
        corr_spearman = ds.correlation(None, "spearman")
        print(f"\n✓ Spearman correlation matrix:")
        print(f"  Variables: {corr_spearman.variables}")
        if len(corr_spearman.variables) >= 2:
            col1, col2 = corr_spearman.variables[0], corr_spearman.variables[1]
            print(f"  Sample: corr({col1}, {col2}) = {corr_spearman.matrix[0][1]:.4f}")
        
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        traceback.print_exc()
        return False

def test_enhanced_summary(ds):
    """Test Phase 5: Enhanced summary statistics"""
    print_section("Phase 5: Enhanced Summary Statistics")
    try:
        summaries = ds.enhanced_summary(None)
        print(f"✓ Enhanced summary for {len(summaries)} columns:\n")
        
        for summ in summaries[:3]:  # Show first 3 columns
            print(f"  Variable: {summ.variable}")
            print(f"    Mean:     {summ.mean:.4f}")
            print(f"    Median:   {summ.median:.4f}")
            print(f"    Std Dev:  {summ.std:.4f}")
            print(f"    Min:      {summ.min:.4f}")
            print(f"    Q1:       {summ.q25:.4f}")
            print(f"    Q3:       {summ.q75:.4f}")
            print(f"    Max:      {summ.max:.4f}")
            print(f"    Skewness: {summ.skewness:.4f}")
            print(f"    Kurtosis: {summ.kurtosis:.4f}")
            print()
        
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        traceback.print_exc()
        return False

def test_frequency_table(ds):
    """Test Phase 5: Frequency tables"""
    print_section("Phase 5: Frequency Tables")
    try:
        # Test on first column
        col_name = ds.column_names()[0]
        freq_table = ds.frequency_table(col_name, 10)
        
        print(f"✓ Frequency table for '{freq_table.variable}' (top {len(freq_table.rows)} values):")
        print(f"  Total count: {freq_table.total_count}\n")
        
        for row in freq_table.rows[:5]:  # Show top 5
            print(f"  {row.value}: {row.count} ({row.percentage:.2f}%)")
        
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        traceback.print_exc()
        return False

def test_hypothesis_tests(ds):
    """Test Phase 5: Hypothesis testing (t-tests)"""
    print_section("Phase 5: Hypothesis Testing (t-tests)")
    try:
        numeric_cols = [col for col in ds.column_names() if ds.is_numeric_column(col)]
        
        if len(numeric_cols) >= 2:
            col1, col2 = numeric_cols[0], numeric_cols[1]
            
            # One-sample t-test
            print(f"✓ One-sample t-test on '{col1}' (μ₀ = 5.0):")
            result1 = ds.t_test_one_sample(col1, 5.0, 0.05)
            print(f"  Test type:     {result1.test_type}")
            print(f"  t-statistic:   {result1.statistic:.4f}")
            print(f"  p-value:       {result1.p_value:.4f}")
            print(f"  df:            {result1.degrees_of_freedom:.0f}")
            print(f"  Mean:          {result1.mean1:.4f}")
            print(f"  95% CI:        [{result1.ci_lower:.4f}, {result1.ci_upper:.4f}]")
            print(f"  Significant:   {result1.significant}")
            
            # Two-sample t-test
            print(f"\n✓ Two-sample t-test: '{col1}' vs '{col2}':")
            result2 = ds.t_test_two_sample(col1, col2, 0.05, True)
            print(f"  Test type:     {result2.test_type}")
            print(f"  t-statistic:   {result2.statistic:.4f}")
            print(f"  p-value:       {result2.p_value:.4f}")
            print(f"  df:            {result2.degrees_of_freedom:.0f}")
            print(f"  Mean 1:        {result2.mean1:.4f}")
            print(f"  Mean 2:        {result2.mean2:.4f}")
            print(f"  95% CI:        [{result2.ci_lower:.4f}, {result2.ci_upper:.4f}]")
            print(f"  Significant:   {result2.significant}")
            
            # Paired t-test
            print(f"\n✓ Paired t-test: '{col1}' vs '{col2}':")
            result3 = ds.t_test_paired(col1, col2, 0.05)
            print(f"  Test type:     {result3.test_type}")
            print(f"  t-statistic:   {result3.statistic:.4f}")
            print(f"  p-value:       {result3.p_value:.4f}")
            print(f"  df:            {result3.degrees_of_freedom:.0f}")
            print(f"  Mean diff:     {result3.mean1:.4f}")
            print(f"  95% CI:        [{result3.ci_lower:.4f}, {result3.ci_upper:.4f}]")
            print(f"  Significant:   {result3.significant}")
            
            return True
        else:
            print("✗ Need at least 2 numeric columns for t-tests")
            return False
            
    except Exception as e:
        print(f"✗ Error: {e}")
        traceback.print_exc()
        return False

def main():
    """Main test runner"""
    print("\n" + "="*70)
    print("  MATHEMIX - PHASE 4 & 5 COMPREHENSIVE TEST")
    print("="*70)
    
    # Load sample dataset
    print("\nLoading sample dataset...")
    try:
        # Load the iris dataset (example.csv)
        ds = mx.load_csv("data/example.csv")
        print(f"✓ Loaded: {ds.n_rows()} rows × {ds.n_cols()} columns")
        print(f"  Columns: {ds.column_names()[:5]}...")
    except Exception as e:
        print(f"✗ Could not load data/example.csv: {e}")
        print("\nPlease ensure data/example.csv exists in the Mathemix folder")
        return 1
    
    # Run all tests
    results = {
        "Column Info": test_column_info(ds),
        "Data Manipulation": test_data_manipulation(ds),
        "Correlation": test_correlation(ds),
        "Enhanced Summary": test_enhanced_summary(ds),
        "Frequency Table": test_frequency_table(ds),
        "Hypothesis Tests": test_hypothesis_tests(ds),
    }
    
    # Summary
    print_section("TEST SUMMARY")
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}: {test_name}")
    
    print(f"\n  Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Phase 4 & 5 implementation successful!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Review errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
