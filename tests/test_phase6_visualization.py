"""
Comprehensive test script for Phase 6: Data Visualization
"""
import sys
import mathemixx_core as mx
from python import plots
import matplotlib.pyplot as plt

def print_section(title):
    """Print a formatted section header"""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)

def test_regression_diagnostics():
    """Test Phase 6: Regression diagnostic plots"""
    print_section("Phase 6: Regression Diagnostic Plots")
    try:
        # Load data
        ds = mx.load_csv("data/example.csv")
        
        # Run regression
        print("Running regression: petal_length ~ sepal_length + sepal_width")
        result = ds.regress_ols("petal_length", ["sepal_length", "sepal_width"], None)
        
        # Test diagnostic data methods
        print("\n✓ Testing diagnostic data methods:")
        
        resid_fitted = result.residual_fitted_data()
        print(f"  - residual_fitted_data: {len(resid_fitted.fitted_values)} points")
        
        qq_data = result.qq_plot_data()
        print(f"  - qq_plot_data: {len(qq_data.theoretical_quantiles)} quantiles")
        
        scale_loc = result.scale_location_data()
        print(f"  - scale_location_data: {len(scale_loc.sqrt_abs_residuals)} points")
        
        resid_lev = result.residuals_leverage_data()
        print(f"  - residuals_leverage_data: {len(resid_lev.leverage)} points")
        print(f"    Max Cook's distance: {max(resid_lev.cooks_distance):.4f}")
        
        resid_hist = result.residual_histogram_data(None)
        print(f"  - residual_histogram_data: {len(resid_hist.residuals)} residuals, {resid_hist.bins} bins")
        
        # Create plots
        print("\n✓ Creating diagnostic plots...")
        
        print("  - Residual vs Fitted plot")
        plots.plot_residual_fitted(result)
        plt.savefig("test_residual_fitted.png", dpi=150)
        plt.close()
        
        print("  - Q-Q plot")
        plots.plot_qq(result)
        plt.savefig("test_qq_plot.png", dpi=150)
        plt.close()
        
        print("  - Scale-Location plot")
        plots.plot_scale_location(result)
        plt.savefig("test_scale_location.png", dpi=150)
        plt.close()
        
        print("  - Residuals vs Leverage plot")
        plots.plot_residuals_leverage(result)
        plt.savefig("test_residuals_leverage.png", dpi=150)
        plt.close()
        
        print("  - Residual histogram")
        plots.plot_residual_histogram(result)
        plt.savefig("test_residual_histogram.png", dpi=150)
        plt.close()
        
        print("  - Complete diagnostic suite")
        plots.plot_diagnostic_suite(result)
        plt.savefig("test_diagnostic_suite.png", dpi=150)
        plt.close()
        
        print("\n✓ All diagnostic plots saved successfully!")
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_general_plots():
    """Test Phase 6: General statistical plots"""
    print_section("Phase 6: General Statistical Plots")
    try:
        # Load data
        ds = mx.load_csv("data/example.csv")
        
        # Test plot data methods
        print("\n✓ Testing plot data methods:")
        
        box_data = ds.box_plot_data("sepal_length")
        print(f"  - box_plot_data for '{box_data.variable}':")
        print(f"    Min: {box_data.min:.2f}, Q1: {box_data.q1:.2f}, Median: {box_data.median:.2f}")
        print(f"    Q3: {box_data.q3:.2f}, Max: {box_data.max:.2f}, Mean: {box_data.mean:.2f}")
        print(f"    Outliers: {len(box_data.outliers)}")
        
        hist_data = ds.histogram_data("petal_length", None)
        print(f"\n  - histogram_data for '{hist_data.variable}':")
        print(f"    Values: {len(hist_data.values)}, Bins: {hist_data.bins}")
        
        heatmap_data = ds.heatmap_data(None, "pearson")
        print(f"\n  - heatmap_data ({heatmap_data.method}):")
        print(f"    Variables: {heatmap_data.variables}")
        print(f"    Matrix size: {len(heatmap_data.correlation_matrix)}x{len(heatmap_data.correlation_matrix[0])}")
        
        violin_data = ds.violin_plot_data("petal_width")
        print(f"\n  - violin_plot_data for '{violin_data.variable}':")
        print(f"    Values: {len(violin_data.values)}")
        
        # Create plots
        print("\n✓ Creating statistical plots...")
        
        print("  - Box plot")
        plots.plot_boxplot(ds, "sepal_length")
        plt.savefig("test_boxplot.png", dpi=150)
        plt.close()
        
        print("  - Histogram with KDE")
        plots.plot_histogram(ds, "petal_length", kde=True)
        plt.savefig("test_histogram.png", dpi=150)
        plt.close()
        
        print("  - Correlation heatmap")
        plots.plot_correlation_heatmap(ds, method='pearson')
        plt.savefig("test_heatmap.png", dpi=150)
        plt.close()
        
        print("  - Violin plot")
        plots.plot_violin(ds, "petal_width")
        plt.savefig("test_violin.png", dpi=150)
        plt.close()
        
        print("\n✓ All statistical plots saved successfully!")
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_plot_customization():
    """Test plot customization and export features"""
    print_section("Phase 6: Plot Customization & Export")
    try:
        print("\n✓ Testing plot style customization:")
        
        # Test different styles
        styles = ['whitegrid', 'darkgrid', 'white', 'dark']
        for style in styles:
            plots.set_plot_style(style=style, font_scale=1.1)
            print(f"  - Set style: {style}")
        
        # Reset to default
        plots.set_plot_style()
        
        print("\n✓ Testing export to different formats:")
        
        # Create a simple plot
        ds = mx.load_csv("data/example.csv")
        plots.plot_boxplot(ds, "sepal_width")
        
        # Export to different formats
        formats = ['png', 'pdf', 'svg']
        for fmt in formats:
            filename = f"test_export.{fmt}"
            plots.save_plot(filename, dpi=150)
            print(f"  - Exported to {fmt.upper()}")
        
        plt.close()
        
        print("\n✓ Plot customization features working!")
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test runner"""
    print("\n" + "="*70)
    print("  MATHEMIX - PHASE 6 COMPREHENSIVE VISUALIZATION TEST")
    print("="*70)
    
    # Run all tests
    results = {
        "Regression Diagnostics": test_regression_diagnostics(),
        "General Statistical Plots": test_general_plots(),
        "Plot Customization": test_plot_customization(),
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
        print("\n🎉 All tests passed! Phase 6 implementation successful!")
        print("\nGenerated plot files:")
        print("  - test_residual_fitted.png")
        print("  - test_qq_plot.png")
        print("  - test_scale_location.png")
        print("  - test_residuals_leverage.png")
        print("  - test_residual_histogram.png")
        print("  - test_diagnostic_suite.png")
        print("  - test_boxplot.png")
        print("  - test_histogram.png")
        print("  - test_heatmap.png")
        print("  - test_violin.png")
        print("  - test_export.png/pdf/svg")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Review errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
