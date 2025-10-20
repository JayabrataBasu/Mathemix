# MatheMixX Desktop - Statistical Analysis Application

**A high-performance desktop statistical application with Rust-powered computation and a modern PySide6 UI**

## 🎯 Overview

MatheMixX is a Stata-like desktop application for statistical analysis, featuring:
- **Rust Core Engine**: Ultra-fast computations using Intel MKL for linear algebra
- **Python Bindings**: Seamless PyO3 integration for easy extensibility  
- **Modern UI**: Professional PySide6/Qt desktop interface
- **Statistical Methods**: OLS regression with robust standard errors (HC0-HC3)
- **Export Capabilities**: JSON, CSV, and TeX output formats

## ✅ Current Status

**Phase 1-2 Complete:**
- ✅ Numerically validated OLS regression (matches statsmodels)
- ✅ QR decomposition with SVD fallback for rank-deficient matrices
- ✅ Robust standard errors (HC0, HC1, HC2, HC3)
- ✅ Working Python bindings installed as wheel
- ✅ Desktop UI ready to use

## 🚀 Quick Start

### Prerequisites
- Windows 10/11
- Python 3.11+ (3.13.5 recommended)
- Rust 1.70+ (for development)

### Installation

1. **Activate the virtual environment:**
   ```bash
   .mathemix\Scripts\activate
   ```

2. **The bindings are already installed** as `mathemixx-bindings` wheel

3. **Launch the desktop app:**
   - **Option A**: Double-click `run_mathemixx.bat`
   - **Option B**: Run `python run_desktop.py`

### Usage

1. **Load Data**:
   - Click "Open CSV" in the toolbar
   - Select your CSV file (e.g., `data/sample_regression.csv`)
   - Data appears in the "Data Preview" tab

2. **Run Regression**:
   - Enter dependent variable in the "Dependent (Y)" field
   - Select independent variables from the list (Ctrl+Click for multiple)
   - Click "Run Regression"
   - Results appear in the "Results" tab
   - Scatter plot with fit line shows in "Plots" tab

3. **Use Command Console**:
   - Type Stata-style commands:
     ```
     summarize
     regress y x1 x2
     ```

4. **Export Results**:
   - Export .do script of your session
   - Results can be saved as CSV/JSON/TeX

## 📊 Example Session

```python
import mathemixx_core as mx

# Load data
dataset = mx.DataSet.from_csv("data/sample_regression.csv")

# Get summary statistics
summary = dataset.summarize()
for row in summary:
    print(f"{row.variable}: mean={row.mean:.2f}, sd={row.sd:.2f}")

# Run OLS regression
result = dataset.regress_ols("y", ["x1", "x2"], robust=True)

print(f"R² = {result.r_squared():.4f}")
print(f"Adj R² = {result.adj_r_squared():.4f}")

# View coefficients
for row in result.table():
    print(f"{row.variable}: {row.coefficient:.4f} (SE={row.std_error:.4f})")

# Export results
result.export_csv("results.csv")
result.export_tex("results.tex")
```

## 🛠️ Development

### Building from Source

1. **Build Rust core and bindings:**
   ```bash
   # Build and test Rust core
   cargo test -p mathemixx-core

   # Build Python wheel
   cd bindings
   maturin build --release
   
   # Install the wheel
   pip install ../target/wheels/mathemixx_bindings-*.whl --force-reinstall
   ```

2. **Run tests:**
   ```bash
   python test_bindings.py      # Test Python bindings
   python test_integration.py   # Test full integration
   ```

### Adding New Statistical Methods

The architecture makes it easy to add new methods:

1. **Add to Rust core** (`core/src/lib.rs`):
   ```rust
   pub fn new_method(dataset: &DataSet, params: ...) -> Result<NewResult> {
       // Your implementation
   }
   ```

2. **Add Python binding** (`bindings/src/lib.rs`):
   ```rust
   #[pymethods]
   impl PyDataSet {
       pub fn new_method(&self, params: ...) -> PyResult<PyNewResult> {
           let result = new_method(&self.inner, params)
               .map_err(mathemixx_error_to_pyerr)?;
           Ok(PyNewResult { inner: result })
       }
   }
   ```

3. **Rebuild and install:**
   ```bash
   maturin build --release
   pip install target/wheels/mathemixx_bindings-*.whl --force-reinstall
   ```

## 📁 Project Structure

```
Mathemix/
├── core/                    # Rust statistical engine
│   ├── src/
│   │   ├── lib.rs          # Public API
│   │   ├── ols.rs          # OLS regression implementation
│   │   ├── dataframe.rs    # Data structures
│   │   └── summary.rs      # Descriptive statistics
│   └── tests/              # Rust unit tests
│
├── bindings/                # PyO3 Python bindings
│   └── src/lib.rs          # Python interface
│
├── python/
│   └── mathemixx_desktop/  # PySide6 desktop UI
│       └── app.py          # Main application
│
├── data/                    # Sample datasets
├── target/wheels/          # Built Python wheels
└── run_desktop.py          # App launcher
```

## 🔧 Technical Details

### Performance
- **Rust Core**: Compiled with Intel MKL for optimal BLAS/LAPACK performance
- **Memory Efficient**: Uses Polars DataFrames (Apache Arrow format)
- **Type Safe**: Rust's type system catches bugs at compile time

### Numerical Accuracy
- QR decomposition for stable OLS estimation
- SVD fallback for rank-deficient matrices
- Proper covariance matrix transformation using Jacobian method
- Validated against statsmodels reference implementation

### Supported Features
- ✅ OLS Regression with intercept/centering options
- ✅ Robust standard errors (HC0, HC1, HC2, HC3)
- ✅ Confidence intervals
- ✅ F-statistics and hypothesis tests
- ✅ Residual diagnostics
- ✅ Export to multiple formats

## 🎓 Future Enhancements

Planned additions:
- [ ] More regression models (Logistic, Poisson, Panel data)
- [ ] Hypothesis tests (T-test, F-test, Chi-square, ANOVA)
- [ ] Time series analysis
- [ ] Data manipulation tools (merge, reshape)
- [ ] Enhanced visualizations (residual plots, Q-Q plots)
- [ ] Import from Stata/SPSS formats

## 📝 License

MIT License - See LICENSE file for details

## 👥 Contributors

MatheMixX Development Team

---

**Built with ❤️ using Rust 🦀 and Python 🐍**
