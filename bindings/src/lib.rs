use std::f64;
use std::fs::File;
use std::io::Write;
use std::sync::Arc;

use ::mathemixx_core::{
    // Phase 7: Time Series
    acf,
    adf_test,
    diff,
    ema,
    holt_linear,
    holt_winters,
    kpss_test,
    lag,
    ljung_box_test,
    pacf,
    regress,
    rolling_max,
    rolling_mean,
    rolling_min,
    rolling_std,
    seasonal_decompose,
    simple_exp_smoothing,
    sma,
    summarize_numeric,
    wma,
    ADFResult,
    BoxPlotData,
    ColumnInfo,
    ColumnType,
    CorrelationMatrix,
    CorrelationMethod,
    DataSet,
    DecompType,
    DecompositionResult,
    EnhancedSummary,
    FilterCondition,
    ForecastResult,
    FrequencyRow,
    FrequencyTable,
    HeatmapData,
    HistogramData,
    KPSSResult,
    MatheMixxError,
    OlsOptions,
    OlsResult,
    PairPlotData,
    QQPlotData,
    ResidualFittedData,
    ResidualHistogramData,
    ResidualsLeverageData,
    ScaleLocationData,
    TTestResult,
    TimeSeries,
    Transform,
    ViolinPlotData,
};
use polars::io::SerWriter;
use polars::prelude::CsvWriter;
use polars::prelude::{DataFrame, NamedFrom};
use polars::series::Series;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use serde::Serialize;
use serde_json::json;

#[pyclass(name = "DataSet", module = "mathemixx_core")]
pub struct PyDataSet {
    inner: Arc<DataSet>,
    path: Option<String>,
}

#[pyclass(name = "OlsResult", module = "mathemixx_core")]
pub struct PyOlsResult {
    inner: OlsResult,
}

#[pyclass(name = "SummaryRow", module = "mathemixx_core")]
#[derive(Clone, Serialize)]
pub struct PySummaryRow {
    #[pyo3(get)]
    pub variable: String,
    #[pyo3(get)]
    pub mean: f64,
    #[pyo3(get)]
    pub sd: f64,
    #[pyo3(get)]
    pub min: f64,
    #[pyo3(get)]
    pub max: f64,
}

#[pyclass(name = "CoefficientRow", module = "mathemixx_core")]
#[derive(Clone, Serialize)]
pub struct PyCoefficientRow {
    #[pyo3(get)]
    pub variable: String,
    #[pyo3(get)]
    pub coefficient: f64,
    #[pyo3(get)]
    pub std_error: f64,
    #[pyo3(get)]
    pub t_value: f64,
    #[pyo3(get)]
    pub p_value: f64,
    #[pyo3(get)]
    pub ci_lower: f64,
    #[pyo3(get)]
    pub ci_upper: f64,
}

// ============================================================================
// Phase 4 & 5: New wrapper classes
// ============================================================================

#[pyclass(name = "ColumnInfo", module = "mathemixx_core")]
#[derive(Clone, Serialize)]
pub struct PyColumnInfo {
    #[pyo3(get)]
    pub name: String,
    #[pyo3(get)]
    pub dtype: String, // "numeric", "integer", "string", "boolean", "temporal", "mixed"
    #[pyo3(get)]
    pub null_count: usize,
    #[pyo3(get)]
    pub total_count: usize,
    #[pyo3(get)]
    pub null_percentage: f64,
}

impl From<ColumnInfo> for PyColumnInfo {
    fn from(info: ColumnInfo) -> Self {
        let dtype_str = match info.dtype {
            ColumnType::Numeric => "numeric",
            ColumnType::Integer => "integer",
            ColumnType::String => "string",
            ColumnType::Boolean => "boolean",
            ColumnType::Temporal => "temporal",
            ColumnType::Mixed => "mixed",
        };
        PyColumnInfo {
            name: info.name,
            dtype: dtype_str.to_string(),
            null_count: info.null_count,
            total_count: info.total_count,
            null_percentage: info.null_percentage,
        }
    }
}

#[pyclass(name = "CorrelationMatrix", module = "mathemixx_core")]
#[derive(Clone)]
pub struct PyCorrelationMatrix {
    #[pyo3(get)]
    pub variables: Vec<String>,
    #[pyo3(get)]
    pub matrix: Vec<Vec<f64>>,
    #[pyo3(get)]
    pub method: String,
}

#[pymethods]
impl PyCorrelationMatrix {
    pub fn get(&self, var1: &str, var2: &str) -> Option<f64> {
        let idx1 = self.variables.iter().position(|v| v == var1)?;
        let idx2 = self.variables.iter().position(|v| v == var2)?;
        Some(self.matrix[idx1][idx2])
    }
}

impl From<CorrelationMatrix> for PyCorrelationMatrix {
    fn from(corr: CorrelationMatrix) -> Self {
        PyCorrelationMatrix {
            variables: corr.variables,
            matrix: corr.matrix,
            method: corr.method,
        }
    }
}

#[pyclass(name = "EnhancedSummary", module = "mathemixx_core")]
#[derive(Clone, Serialize)]
pub struct PyEnhancedSummary {
    #[pyo3(get)]
    pub variable: String,
    #[pyo3(get)]
    pub count: usize,
    #[pyo3(get)]
    pub null_count: usize,
    #[pyo3(get)]
    pub mean: f64,
    #[pyo3(get)]
    pub median: f64,
    #[pyo3(get)]
    pub std: f64,
    #[pyo3(get)]
    pub variance: f64,
    #[pyo3(get)]
    pub min: f64,
    #[pyo3(get)]
    pub max: f64,
    #[pyo3(get)]
    pub q25: f64,
    #[pyo3(get)]
    pub q50: f64,
    #[pyo3(get)]
    pub q75: f64,
    #[pyo3(get)]
    pub range: f64,
    #[pyo3(get)]
    pub iqr: f64,
    #[pyo3(get)]
    pub skewness: Option<f64>,
    #[pyo3(get)]
    pub kurtosis: Option<f64>,
}

impl From<EnhancedSummary> for PyEnhancedSummary {
    fn from(summary: EnhancedSummary) -> Self {
        PyEnhancedSummary {
            variable: summary.variable,
            count: summary.count,
            null_count: summary.null_count,
            mean: summary.mean,
            median: summary.median,
            std: summary.std,
            variance: summary.variance,
            min: summary.min,
            max: summary.max,
            q25: summary.q25,
            q50: summary.q50,
            q75: summary.q75,
            range: summary.range,
            iqr: summary.iqr,
            skewness: summary.skewness,
            kurtosis: summary.kurtosis,
        }
    }
}

#[pyclass(name = "FrequencyRow", module = "mathemixx_core")]
#[derive(Clone, Serialize)]
pub struct PyFrequencyRow {
    #[pyo3(get)]
    pub value: String,
    #[pyo3(get)]
    pub count: usize,
    #[pyo3(get)]
    pub percentage: f64,
    #[pyo3(get)]
    pub cumulative_count: usize,
    #[pyo3(get)]
    pub cumulative_percentage: f64,
}

impl From<FrequencyRow> for PyFrequencyRow {
    fn from(row: FrequencyRow) -> Self {
        PyFrequencyRow {
            value: row.value,
            count: row.count,
            percentage: row.percentage,
            cumulative_count: row.cumulative_count,
            cumulative_percentage: row.cumulative_percentage,
        }
    }
}

#[pyclass(name = "FrequencyTable", module = "mathemixx_core")]
#[derive(Clone)]
pub struct PyFrequencyTable {
    #[pyo3(get)]
    pub variable: String,
    #[pyo3(get)]
    pub total_count: usize,
    #[pyo3(get)]
    pub unique_count: usize,
    #[pyo3(get)]
    pub rows: Vec<PyFrequencyRow>,
}

impl From<FrequencyTable> for PyFrequencyTable {
    fn from(table: FrequencyTable) -> Self {
        PyFrequencyTable {
            variable: table.variable,
            total_count: table.total_count,
            unique_count: table.unique_count,
            rows: table.rows.into_iter().map(|r| r.into()).collect(),
        }
    }
}

#[pyclass(name = "TTestResult", module = "mathemixx_core")]
#[derive(Clone, Serialize)]
pub struct PyTTestResult {
    #[pyo3(get)]
    pub test_type: String,
    #[pyo3(get)]
    pub statistic: f64,
    #[pyo3(get)]
    pub degrees_of_freedom: f64,
    #[pyo3(get)]
    pub p_value: f64,
    #[pyo3(get)]
    pub mean1: f64,
    #[pyo3(get)]
    pub mean2: Option<f64>,
    #[pyo3(get)]
    pub ci_lower: f64,
    #[pyo3(get)]
    pub ci_upper: f64,
    #[pyo3(get)]
    pub significant: bool,
}

impl From<TTestResult> for PyTTestResult {
    fn from(result: TTestResult) -> Self {
        PyTTestResult {
            test_type: result.test_type,
            statistic: result.statistic,
            degrees_of_freedom: result.degrees_of_freedom,
            p_value: result.p_value,
            mean1: result.mean1,
            mean2: result.mean2,
            ci_lower: result.confidence_interval.0,
            ci_upper: result.confidence_interval.1,
            significant: result.significant,
        }
    }
}

// ============================================================================
// Phase 6: Visualization wrapper classes
// ============================================================================

#[pyclass(name = "ResidualFittedData", module = "mathemixx_core")]
#[derive(Clone, Serialize)]
pub struct PyResidualFittedData {
    #[pyo3(get)]
    pub fitted_values: Vec<f64>,
    #[pyo3(get)]
    pub residuals: Vec<f64>,
}

impl From<ResidualFittedData> for PyResidualFittedData {
    fn from(data: ResidualFittedData) -> Self {
        PyResidualFittedData {
            fitted_values: data.fitted_values,
            residuals: data.residuals,
        }
    }
}

#[pyclass(name = "QQPlotData", module = "mathemixx_core")]
#[derive(Clone, Serialize)]
pub struct PyQQPlotData {
    #[pyo3(get)]
    pub theoretical_quantiles: Vec<f64>,
    #[pyo3(get)]
    pub sample_quantiles: Vec<f64>,
}

impl From<QQPlotData> for PyQQPlotData {
    fn from(data: QQPlotData) -> Self {
        PyQQPlotData {
            theoretical_quantiles: data.theoretical_quantiles,
            sample_quantiles: data.sample_quantiles,
        }
    }
}

#[pyclass(name = "ScaleLocationData", module = "mathemixx_core")]
#[derive(Clone, Serialize)]
pub struct PyScaleLocationData {
    #[pyo3(get)]
    pub fitted_values: Vec<f64>,
    #[pyo3(get)]
    pub sqrt_abs_residuals: Vec<f64>,
}

impl From<ScaleLocationData> for PyScaleLocationData {
    fn from(data: ScaleLocationData) -> Self {
        PyScaleLocationData {
            fitted_values: data.fitted_values,
            sqrt_abs_residuals: data.sqrt_abs_residuals,
        }
    }
}

#[pyclass(name = "ResidualsLeverageData", module = "mathemixx_core")]
#[derive(Clone, Serialize)]
pub struct PyResidualsLeverageData {
    #[pyo3(get)]
    pub leverage: Vec<f64>,
    #[pyo3(get)]
    pub standardized_residuals: Vec<f64>,
    #[pyo3(get)]
    pub cooks_distance: Vec<f64>,
}

impl From<ResidualsLeverageData> for PyResidualsLeverageData {
    fn from(data: ResidualsLeverageData) -> Self {
        PyResidualsLeverageData {
            leverage: data.leverage,
            standardized_residuals: data.standardized_residuals,
            cooks_distance: data.cooks_distance,
        }
    }
}

#[pyclass(name = "ResidualHistogramData", module = "mathemixx_core")]
#[derive(Clone, Serialize)]
pub struct PyResidualHistogramData {
    #[pyo3(get)]
    pub residuals: Vec<f64>,
    #[pyo3(get)]
    pub bins: usize,
}

impl From<ResidualHistogramData> for PyResidualHistogramData {
    fn from(data: ResidualHistogramData) -> Self {
        PyResidualHistogramData {
            residuals: data.residuals,
            bins: data.bins,
        }
    }
}

#[pyclass(name = "BoxPlotData", module = "mathemixx_core")]
#[derive(Clone, Serialize)]
pub struct PyBoxPlotData {
    #[pyo3(get)]
    pub variable: String,
    #[pyo3(get)]
    pub min: f64,
    #[pyo3(get)]
    pub q1: f64,
    #[pyo3(get)]
    pub median: f64,
    #[pyo3(get)]
    pub q3: f64,
    #[pyo3(get)]
    pub max: f64,
    #[pyo3(get)]
    pub outliers: Vec<f64>,
    #[pyo3(get)]
    pub mean: f64,
}

impl From<BoxPlotData> for PyBoxPlotData {
    fn from(data: BoxPlotData) -> Self {
        PyBoxPlotData {
            variable: data.variable,
            min: data.min,
            q1: data.q1,
            median: data.median,
            q3: data.q3,
            max: data.max,
            outliers: data.outliers,
            mean: data.mean,
        }
    }
}

#[pyclass(name = "HistogramData", module = "mathemixx_core")]
#[derive(Clone, Serialize)]
pub struct PyHistogramData {
    #[pyo3(get)]
    pub variable: String,
    #[pyo3(get)]
    pub values: Vec<f64>,
    #[pyo3(get)]
    pub bins: usize,
}

impl From<HistogramData> for PyHistogramData {
    fn from(data: HistogramData) -> Self {
        PyHistogramData {
            variable: data.variable,
            values: data.values,
            bins: data.bins,
        }
    }
}

#[pyclass(name = "HeatmapData", module = "mathemixx_core")]
#[derive(Clone, Serialize)]
pub struct PyHeatmapData {
    #[pyo3(get)]
    pub variables: Vec<String>,
    #[pyo3(get)]
    pub correlation_matrix: Vec<Vec<f64>>,
    #[pyo3(get)]
    pub method: String,
}

impl From<HeatmapData> for PyHeatmapData {
    fn from(data: HeatmapData) -> Self {
        PyHeatmapData {
            variables: data.variables,
            correlation_matrix: data.correlation_matrix,
            method: data.method,
        }
    }
}

#[pyclass(name = "PairPlotData", module = "mathemixx_core")]
#[derive(Clone, Serialize)]
pub struct PyPairPlotData {
    #[pyo3(get)]
    pub variables: Vec<String>,
}

#[pymethods]
impl PyPairPlotData {
    /// Get data for a specific variable
    pub fn get_data(&self, py: Python, variable: &str) -> PyResult<Vec<f64>> {
        // This will be populated when we add the method
        Ok(vec![])
    }
}

impl From<PairPlotData> for PyPairPlotData {
    fn from(data: PairPlotData) -> Self {
        // Store the HashMap data temporarily
        // We'll convert it to a proper structure
        PyPairPlotData {
            variables: data.variables,
        }
    }
}

#[pyclass(name = "ViolinPlotData", module = "mathemixx_core")]
#[derive(Clone, Serialize)]
pub struct PyViolinPlotData {
    #[pyo3(get)]
    pub variable: String,
    #[pyo3(get)]
    pub values: Vec<f64>,
}

impl From<ViolinPlotData> for PyViolinPlotData {
    fn from(data: ViolinPlotData) -> Self {
        PyViolinPlotData {
            variable: data.variable,
            values: data.values,
        }
    }
}

// ============================================================================

#[pymethods]
impl PyDataSet {
    #[new]
    pub fn new(path: &str) -> PyResult<Self> {
        let ds = DataSet::from_csv(path).map_err(mathemixx_error_to_pyerr)?;
        Ok(Self {
            inner: Arc::new(ds),
            path: Some(path.to_string()),
        })
    }

    #[staticmethod]
    pub fn from_csv(path: &str) -> PyResult<Self> {
        Self::new(path)
    }

    #[getter]
    pub fn source_path(&self) -> Option<String> {
        self.path.clone()
    }

    pub fn column_names(&self) -> Vec<String> {
        self.inner.column_names()
    }

    pub fn n_rows(&self) -> usize {
        self.inner.height()
    }

    pub fn n_cols(&self) -> usize {
        self.inner.width()
    }

    pub fn numeric_columns(&self) -> Vec<String> {
        self.inner.numeric_column_names()
    }

    pub fn summarize(&self) -> PyResult<Vec<PySummaryRow>> {
        let summary = summarize_numeric(&self.inner).map_err(mathemixx_error_to_pyerr)?;
        Ok(dataframe_to_summary(summary))
    }

    pub fn regress_ols(
        &self,
        dependent: &str,
        independents: Vec<String>,
        robust: Option<bool>,
    ) -> PyResult<PyOlsResult> {
        let mut opts = OlsOptions::default();
        if let Some(robust_flag) = robust {
            opts.robust = robust_flag;
        }
        let result = regress(&self.inner, dependent, &independents, opts)
            .map_err(mathemixx_error_to_pyerr)?;
        Ok(PyOlsResult { inner: result })
    }

    // ========================================================================
    // Phase 4: Data Manipulation Methods
    // ========================================================================

    /// Get information about all columns
    pub fn column_info(&self) -> PyResult<Vec<PyColumnInfo>> {
        let info = self.inner.column_info().map_err(mathemixx_error_to_pyerr)?;
        Ok(info.into_iter().map(|i| i.into()).collect())
    }

    /// Check if a column is numeric
    pub fn is_numeric_column(&self, column: &str) -> PyResult<bool> {
        self.inner
            .is_numeric_column(column)
            .map_err(mathemixx_error_to_pyerr)
    }

    /// Select only numeric columns
    pub fn select_numeric(&self) -> PyResult<PyDataSet> {
        let ds = self
            .inner
            .select_numeric()
            .map_err(mathemixx_error_to_pyerr)?;
        Ok(PyDataSet {
            inner: Arc::new(ds),
            path: None,
        })
    }

    /// Rename a column
    pub fn rename_column(&self, old_name: &str, new_name: &str) -> PyResult<PyDataSet> {
        let ds = self
            .inner
            .rename_column(old_name, new_name)
            .map_err(mathemixx_error_to_pyerr)?;
        Ok(PyDataSet {
            inner: Arc::new(ds),
            path: None,
        })
    }

    /// Drop columns
    pub fn drop_columns(&self, columns: Vec<String>) -> PyResult<PyDataSet> {
        let ds = self
            .inner
            .drop_columns(&columns)
            .map_err(mathemixx_error_to_pyerr)?;
        Ok(PyDataSet {
            inner: Arc::new(ds),
            path: None,
        })
    }

    /// Add a transformed column
    /// transform: "log", "log10", "square", "sqrt", "standardize", "center", "inverse"
    pub fn add_column_transform(
        &self,
        source: &str,
        target: &str,
        transform: &str,
    ) -> PyResult<PyDataSet> {
        let trans = match transform.to_lowercase().as_str() {
            "log" => Transform::Log,
            "log10" => Transform::Log10,
            "square" => Transform::Square,
            "sqrt" | "squareroot" => Transform::SquareRoot,
            "standardize" | "std" | "zscore" => Transform::Standardize,
            "center" | "demean" => Transform::Center,
            "inverse" | "reciprocal" => Transform::Inverse,
            _ => {
                return Err(PyValueError::new_err(format!(
                    "Unknown transform: {}",
                    transform
                )))
            }
        };
        let ds = self
            .inner
            .add_column_transform(source, target, trans)
            .map_err(mathemixx_error_to_pyerr)?;
        Ok(PyDataSet {
            inner: Arc::new(ds),
            path: None,
        })
    }

    /// Filter rows by condition
    /// condition: "gt", "lt", "eq", "ge", "le", "ne"
    pub fn filter_rows(&self, column: &str, condition: &str, value: f64) -> PyResult<PyDataSet> {
        let cond = match condition.to_lowercase().as_str() {
            "gt" | ">" => FilterCondition::GreaterThan(value),
            "lt" | "<" => FilterCondition::LessThan(value),
            "eq" | "==" | "=" => FilterCondition::Equal(value),
            "ge" | ">=" => FilterCondition::GreaterOrEqual(value),
            "le" | "<=" => FilterCondition::LessOrEqual(value),
            "ne" | "!=" => FilterCondition::NotEqual(value),
            _ => {
                return Err(PyValueError::new_err(format!(
                    "Unknown condition: {}",
                    condition
                )))
            }
        };
        let ds = self
            .inner
            .filter_rows(column, cond)
            .map_err(mathemixx_error_to_pyerr)?;
        Ok(PyDataSet {
            inner: Arc::new(ds),
            path: None,
        })
    }

    // ========================================================================
    // Phase 5: Statistical Methods
    // ========================================================================

    /// Calculate correlation matrix
    /// method: "pearson" or "spearman"
    pub fn correlation(
        &self,
        columns: Option<Vec<String>>,
        method: Option<&str>,
    ) -> PyResult<PyCorrelationMatrix> {
        let corr_method = match method.unwrap_or("pearson").to_lowercase().as_str() {
            "pearson" => CorrelationMethod::Pearson,
            "spearman" => CorrelationMethod::Spearman,
            other => {
                return Err(PyValueError::new_err(format!(
                    "Unknown correlation method: {}",
                    other
                )))
            }
        };
        let corr = self
            .inner
            .correlation(columns, corr_method)
            .map_err(mathemixx_error_to_pyerr)?;
        Ok(corr.into())
    }

    /// Get enhanced summary statistics
    pub fn enhanced_summary(
        &self,
        columns: Option<Vec<String>>,
    ) -> PyResult<Vec<PyEnhancedSummary>> {
        let summary = self
            .inner
            .enhanced_summary(columns)
            .map_err(mathemixx_error_to_pyerr)?;
        Ok(summary.into_iter().map(|s| s.into()).collect())
    }

    /// Get frequency table for a column
    pub fn frequency_table(
        &self,
        column: &str,
        limit: Option<usize>,
    ) -> PyResult<PyFrequencyTable> {
        let table = self
            .inner
            .frequency_table(column, limit)
            .map_err(mathemixx_error_to_pyerr)?;
        Ok(table.into())
    }

    // ========================================================================
    // Phase 5: Hypothesis Testing
    // ========================================================================

    /// One-sample t-test
    pub fn t_test_one_sample(
        &self,
        column: &str,
        population_mean: f64,
        alpha: Option<f64>,
    ) -> PyResult<PyTTestResult> {
        let a = alpha.unwrap_or(0.05);
        let result = self
            .inner
            .t_test_one_sample(column, population_mean, a)
            .map_err(mathemixx_error_to_pyerr)?;
        Ok(result.into())
    }

    /// Two-sample t-test
    pub fn t_test_two_sample(
        &self,
        col1: &str,
        col2: &str,
        alpha: Option<f64>,
        equal_var: Option<bool>,
    ) -> PyResult<PyTTestResult> {
        let a = alpha.unwrap_or(0.05);
        let eq_var = equal_var.unwrap_or(true);
        let result = self
            .inner
            .t_test_two_sample(col1, col2, a, eq_var)
            .map_err(mathemixx_error_to_pyerr)?;
        Ok(result.into())
    }

    /// Paired t-test
    pub fn t_test_paired(
        &self,
        col1: &str,
        col2: &str,
        alpha: Option<f64>,
    ) -> PyResult<PyTTestResult> {
        let a = alpha.unwrap_or(0.05);
        let result = self
            .inner
            .t_test_paired(col1, col2, a)
            .map_err(mathemixx_error_to_pyerr)?;
        Ok(result.into())
    }

    // ========================================================================
    // Phase 6: General Visualization Data Methods
    // ========================================================================

    /// Get box plot data for a column
    pub fn box_plot_data(&self, column: &str) -> PyResult<PyBoxPlotData> {
        let data = self
            .inner
            .box_plot_data(column)
            .map_err(mathemixx_error_to_pyerr)?;
        Ok(data.into())
    }

    /// Get histogram data for a column
    pub fn histogram_data(&self, column: &str, bins: Option<usize>) -> PyResult<PyHistogramData> {
        let data = self
            .inner
            .histogram_data(column, bins)
            .map_err(mathemixx_error_to_pyerr)?;
        Ok(data.into())
    }

    /// Get correlation heatmap data
    pub fn heatmap_data(
        &self,
        columns: Option<Vec<String>>,
        method: Option<String>,
    ) -> PyResult<PyHeatmapData> {
        let data = self
            .inner
            .heatmap_data(columns, method)
            .map_err(mathemixx_error_to_pyerr)?;
        Ok(data.into())
    }

    /// Get pair plot data
    pub fn pair_plot_data(&self, columns: Option<Vec<String>>) -> PyResult<PyPairPlotData> {
        let data = self
            .inner
            .pair_plot_data(columns)
            .map_err(mathemixx_error_to_pyerr)?;
        Ok(data.into())
    }

    /// Get violin plot data for a column
    pub fn violin_plot_data(&self, column: &str) -> PyResult<PyViolinPlotData> {
        let data = self
            .inner
            .violin_plot_data(column)
            .map_err(mathemixx_error_to_pyerr)?;
        Ok(data.into())
    }
}

#[pymethods]
impl PyOlsResult {
    #[getter]
    pub fn dependent(&self) -> String {
        self.inner.dependent.clone()
    }

    #[getter]
    pub fn regressors(&self) -> Vec<String> {
        self.inner.regressors.clone()
    }

    #[getter]
    pub fn coefficients(&self) -> Vec<f64> {
        self.inner.coefficients.clone()
    }

    #[getter]
    pub fn stderr(&self) -> Vec<f64> {
        self.inner.stderr.clone()
    }

    #[getter]
    pub fn t_values(&self) -> Vec<f64> {
        self.inner.t_values.clone()
    }

    #[getter]
    pub fn p_values(&self) -> Vec<f64> {
        self.inner.p_values.clone()
    }

    #[getter]
    pub fn conf_int(&self) -> Vec<(f64, f64)> {
        self.inner.conf_int.clone()
    }

    pub fn r_squared(&self) -> f64 {
        self.inner.r_squared
    }

    pub fn adj_r_squared(&self) -> f64 {
        self.inner.adj_r_squared
    }

    pub fn sigma2(&self) -> f64 {
        self.inner.sigma2
    }

    pub fn nobs(&self) -> usize {
        self.inner.nobs
    }

    pub fn to_json(&self) -> PyResult<String> {
        let table = self.coefficient_rows();
        let table_json =
            serde_json::to_value(&table).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let json_value = json!({
            "dependent": self.inner.dependent.clone(),
            "regressors": self.inner.regressors.clone(),
            "coefficients": self.inner.coefficients.clone(),
            "stderr": self.inner.stderr.clone(),
            "t_values": self.inner.t_values.clone(),
            "p_values": self.inner.p_values.clone(),
            "conf_int": self.inner.conf_int.clone(),
            "r_squared": self.inner.r_squared,
            "adj_r_squared": self.inner.adj_r_squared,
            "f_stat": self.inner.f_stat,
            "f_p_value": self.inner.f_p_value,
            "df_model": self.inner.df_model,
            "df_resid": self.inner.df_resid,
            "sigma2": self.inner.sigma2,
            "nobs": self.inner.nobs,
            "table": table_json,
        });
        Ok(json_value.to_string())
    }

    pub fn export_csv(&self, path: &str) -> PyResult<()> {
        let mut table = summary_table_to_polars(self.coefficient_rows());
        let file = File::create(path).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let mut writer = CsvWriter::new(file);
        writer
            .finish(&mut table)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(())
    }

    pub fn export_tex(&self, path: &str) -> PyResult<()> {
        let mut file = File::create(path).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let header = [
            "\\begin{tabular}{lrrrrrr}",
            "\\toprule",
            "Variable & Coef & Std. Err. & t & P>|t| & [0.025 & 0.975]\\\\",
            "\\midrule",
        ]
        .join("\n");
        file.write_all(header.as_bytes())
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        for row in self.coefficient_rows() {
            let line = format!(
                "{name} & {coef:.6} & {se:.6} & {t:.3} & {p:.3} & {lo:.3} & {hi:.3}\\\\\n",
                name = row.variable,
                coef = row.coefficient,
                se = row.std_error,
                t = row.t_value,
                p = row.p_value,
                lo = row.ci_lower,
                hi = row.ci_upper
            );
            file.write_all(line.as_bytes())
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        }
        let footer = "\\bottomrule\n\\end{tabular}\n";
        file.write_all(footer.as_bytes())
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(())
    }

    pub fn robust_std_errors(&self, py: Python<'_>) -> PyResult<PyObject> {
        let dict = PyDict::new(py);
        for (kind, values) in &self.inner.robust_stderr {
            dict.set_item(kind.as_str(), PyList::new(py, values.clone()))?;
        }
        Ok(dict.into())
    }

    pub fn table(&self) -> Vec<PyCoefficientRow> {
        self.coefficient_rows()
    }

    // ========================================================================
    // Phase 6: Regression Diagnostic Plot Data Methods
    // ========================================================================

    /// Get data for residual vs fitted plot
    pub fn residual_fitted_data(&self) -> PyResult<PyResidualFittedData> {
        let data = self
            .inner
            .residual_fitted_data()
            .map_err(mathemixx_error_to_pyerr)?;
        Ok(data.into())
    }

    /// Get data for Q-Q plot (normal probability plot)
    pub fn qq_plot_data(&self) -> PyResult<PyQQPlotData> {
        let data = self
            .inner
            .qq_plot_data()
            .map_err(mathemixx_error_to_pyerr)?;
        Ok(data.into())
    }

    /// Get data for scale-location plot
    pub fn scale_location_data(&self) -> PyResult<PyScaleLocationData> {
        let data = self
            .inner
            .scale_location_data()
            .map_err(mathemixx_error_to_pyerr)?;
        Ok(data.into())
    }

    /// Get data for residuals vs leverage plot
    pub fn residuals_leverage_data(&self) -> PyResult<PyResidualsLeverageData> {
        let data = self
            .inner
            .residuals_leverage_data()
            .map_err(mathemixx_error_to_pyerr)?;
        Ok(data.into())
    }

    /// Get data for residual histogram
    pub fn residual_histogram_data(
        &self,
        bins: Option<usize>,
    ) -> PyResult<PyResidualHistogramData> {
        let data = self
            .inner
            .residual_histogram_data(bins)
            .map_err(mathemixx_error_to_pyerr)?;
        Ok(data.into())
    }
}

impl PyOlsResult {
    fn coefficient_rows(&self) -> Vec<PyCoefficientRow> {
        self.inner
            .regressors
            .iter()
            .enumerate()
            .map(|(idx, name)| PyCoefficientRow {
                variable: name.clone(),
                coefficient: self.inner.coefficients[idx],
                std_error: self.inner.stderr[idx],
                t_value: self.inner.t_values[idx],
                p_value: self.inner.p_values[idx],
                ci_lower: self.inner.conf_int[idx].0,
                ci_upper: self.inner.conf_int[idx].1,
            })
            .collect()
    }
}

fn dataframe_to_summary(frame: DataFrame) -> Vec<PySummaryRow> {
    let nrows = frame.height();
    let mut rows = Vec::with_capacity(nrows);

    let variable_col = frame.column("variable").expect("variable column");
    let mean_col = frame.column("mean").expect("mean column");
    let sd_col = frame.column("sd").expect("sd column");
    let min_col = frame.column("min").expect("min column");
    let max_col = frame.column("max").expect("max column");

    for idx in 0..nrows {
        let variable = variable_col
            .str()
            .expect("string column")
            .get(idx)
            .expect("variable value")
            .to_string();
        let mean = mean_col
            .f64()
            .expect("f64 column")
            .get(idx)
            .unwrap_or(f64::NAN);
        let sd = sd_col
            .f64()
            .expect("f64 column")
            .get(idx)
            .unwrap_or(f64::NAN);
        let min = min_col
            .f64()
            .expect("f64 column")
            .get(idx)
            .unwrap_or(f64::NAN);
        let max = max_col
            .f64()
            .expect("f64 column")
            .get(idx)
            .unwrap_or(f64::NAN);

        rows.push(PySummaryRow {
            variable,
            mean,
            sd,
            min,
            max,
        });
    }
    rows
}

fn summary_table_to_polars(rows: Vec<PyCoefficientRow>) -> DataFrame {
    let mut variables = Vec::with_capacity(rows.len());
    let mut coefficients = Vec::with_capacity(rows.len());
    let mut std_errors = Vec::with_capacity(rows.len());
    let mut t_values = Vec::with_capacity(rows.len());
    let mut p_values = Vec::with_capacity(rows.len());
    let mut ci_lower = Vec::with_capacity(rows.len());
    let mut ci_upper = Vec::with_capacity(rows.len());

    for row in rows {
        variables.push(row.variable);
        coefficients.push(row.coefficient);
        std_errors.push(row.std_error);
        t_values.push(row.t_value);
        p_values.push(row.p_value);
        ci_lower.push(row.ci_lower);
        ci_upper.push(row.ci_upper);
    }

    DataFrame::new(vec![
        Series::new("variable".into(), variables).into(),
        Series::new("coefficient".into(), coefficients).into(),
        Series::new("std_error".into(), std_errors).into(),
        Series::new("t_value".into(), t_values).into(),
        Series::new("p_value".into(), p_values).into(),
        Series::new("ci_lower".into(), ci_lower).into(),
        Series::new("ci_upper".into(), ci_upper).into(),
    ])
    .expect("coefficient dataframe")
}

#[pyfunction]
pub fn load_csv(path: &str) -> PyResult<PyDataSet> {
    PyDataSet::new(path)
}

#[pyfunction]
pub fn summarize_dataset(dataset: &PyDataSet) -> PyResult<Vec<PySummaryRow>> {
    dataset.summarize()
}

#[pyfunction]
pub fn regress_dataset(
    dataset: &PyDataSet,
    dependent: &str,
    independents: Vec<String>,
    robust: Option<bool>,
) -> PyResult<PyOlsResult> {
    dataset.regress_ols(dependent, independents, robust)
}

// ============================================================================
// Phase 7: Time Series Bindings
// ============================================================================

/// Python wrapper for time series operations
#[pyfunction]
#[pyo3(signature = (data, periods=1))]
pub fn py_lag(data: Vec<f64>, periods: usize) -> PyResult<Vec<f64>> {
    Ok(lag(&data, periods))
}

#[pyfunction]
#[pyo3(signature = (data, periods=1))]
pub fn py_diff(data: Vec<f64>, periods: usize) -> PyResult<Vec<f64>> {
    Ok(diff(&data, periods))
}

#[pyfunction]
pub fn py_sma(data: Vec<f64>, window: usize) -> PyResult<Vec<f64>> {
    Ok(sma(&data, window))
}

#[pyfunction]
pub fn py_ema(data: Vec<f64>, window: usize) -> PyResult<Vec<f64>> {
    Ok(ema(&data, window))
}

#[pyfunction]
pub fn py_wma(data: Vec<f64>, window: usize) -> PyResult<Vec<f64>> {
    Ok(wma(&data, window))
}

#[pyfunction]
pub fn py_rolling_mean(data: Vec<f64>, window: usize) -> PyResult<Vec<f64>> {
    Ok(rolling_mean(&data, window))
}

#[pyfunction]
pub fn py_rolling_std(data: Vec<f64>, window: usize) -> PyResult<Vec<f64>> {
    Ok(rolling_std(&data, window))
}

#[pyfunction]
pub fn py_rolling_min(data: Vec<f64>, window: usize) -> PyResult<Vec<f64>> {
    Ok(rolling_min(&data, window))
}

#[pyfunction]
pub fn py_rolling_max(data: Vec<f64>, window: usize) -> PyResult<Vec<f64>> {
    Ok(rolling_max(&data, window))
}

#[pyfunction]
pub fn py_acf(data: Vec<f64>, nlags: usize) -> PyResult<Vec<f64>> {
    Ok(acf(&data, nlags))
}

#[pyfunction]
pub fn py_pacf(data: Vec<f64>, nlags: usize) -> PyResult<Vec<f64>> {
    Ok(pacf(&data, nlags))
}

#[pyfunction]
pub fn py_ljung_box_test(data: Vec<f64>, lags: usize) -> PyResult<(f64, f64)> {
    Ok(ljung_box_test(&data, lags))
}

#[pyclass(name = "ADFResult", module = "mathemixx_core")]
#[derive(Clone)]
pub struct PyADFResult {
    #[pyo3(get)]
    pub test_statistic: f64,
    #[pyo3(get)]
    pub p_value: f64,
    #[pyo3(get)]
    pub lags_used: usize,
    #[pyo3(get)]
    pub n_obs: usize,
    #[pyo3(get)]
    pub is_stationary: bool,
}

#[pymethods]
impl PyADFResult {
    fn __repr__(&self) -> String {
        format!(
            "ADFResult(statistic={:.4}, p_value={:.4}, stationary={})",
            self.test_statistic, self.p_value, self.is_stationary
        )
    }

    #[getter]
    fn critical_values(&self) -> (f64, f64, f64) {
        (-3.43, -2.86, -2.57) // 1%, 5%, 10%
    }
}

#[pyclass(name = "KPSSResult", module = "mathemixx_core")]
#[derive(Clone)]
pub struct PyKPSSResult {
    #[pyo3(get)]
    pub test_statistic: f64,
    #[pyo3(get)]
    pub p_value: f64,
    #[pyo3(get)]
    pub lags_used: usize,
    #[pyo3(get)]
    pub is_stationary: bool,
}

#[pymethods]
impl PyKPSSResult {
    fn __repr__(&self) -> String {
        format!(
            "KPSSResult(statistic={:.4}, p_value={:.4}, stationary={})",
            self.test_statistic, self.p_value, self.is_stationary
        )
    }

    #[getter]
    fn critical_values(&self) -> (f64, f64, f64) {
        (0.739, 0.463, 0.347) // 1%, 5%, 10%
    }
}

#[pyfunction]
pub fn py_adf_test(data: Vec<f64>, max_lags: Option<usize>) -> PyResult<PyADFResult> {
    let result = adf_test(&data, max_lags);
    Ok(PyADFResult {
        test_statistic: result.test_statistic,
        p_value: result.p_value,
        lags_used: result.lags_used,
        n_obs: result.n_obs,
        is_stationary: result.is_stationary,
    })
}

#[pyfunction]
pub fn py_kpss_test(data: Vec<f64>, lags: Option<usize>) -> PyResult<PyKPSSResult> {
    let result = kpss_test(&data, lags);
    Ok(PyKPSSResult {
        test_statistic: result.test_statistic,
        p_value: result.p_value,
        lags_used: result.lags_used,
        is_stationary: result.is_stationary,
    })
}

#[pyclass(name = "DecompositionResult", module = "mathemixx_core")]
#[derive(Clone)]
pub struct PyDecompositionResult {
    #[pyo3(get)]
    pub trend: Vec<f64>,
    #[pyo3(get)]
    pub seasonal: Vec<f64>,
    #[pyo3(get)]
    pub residual: Vec<f64>,
    #[pyo3(get)]
    pub observed: Vec<f64>,
}

#[pymethods]
impl PyDecompositionResult {
    fn __repr__(&self) -> String {
        format!(
            "DecompositionResult(n_obs={}, trend_len={}, seasonal_len={}, residual_len={})",
            self.observed.len(),
            self.trend.len(),
            self.seasonal.len(),
            self.residual.len()
        )
    }
}

#[pyfunction]
#[pyo3(signature = (data, period, model="additive"))]
pub fn py_seasonal_decompose(
    data: Vec<f64>,
    period: usize,
    model: &str,
) -> PyResult<PyDecompositionResult> {
    let decomp_type = match model.to_lowercase().as_str() {
        "additive" | "add" => DecompType::Additive,
        "multiplicative" | "mult" => DecompType::Multiplicative,
        _ => {
            return Err(PyValueError::new_err(
                "model must be 'additive' or 'multiplicative'",
            ))
        }
    };

    match seasonal_decompose(&data, period, decomp_type) {
        Ok(result) => Ok(PyDecompositionResult {
            trend: result.trend,
            seasonal: result.seasonal,
            residual: result.residual,
            observed: result.observed,
        }),
        Err(e) => Err(PyValueError::new_err(e)),
    }
}

#[pyclass(name = "ForecastResult", module = "mathemixx_core")]
#[derive(Clone)]
pub struct PyForecastResult {
    #[pyo3(get)]
    pub forecasts: Vec<f64>,
    #[pyo3(get)]
    pub lower_bound: Vec<f64>,
    #[pyo3(get)]
    pub upper_bound: Vec<f64>,
    #[pyo3(get)]
    pub confidence_level: f64,
}

#[pymethods]
impl PyForecastResult {
    fn __repr__(&self) -> String {
        format!(
            "ForecastResult(horizon={}, confidence={:.0}%)",
            self.forecasts.len(),
            self.confidence_level * 100.0
        )
    }
}

#[pyfunction]
#[pyo3(signature = (data, alpha=0.3, horizon=12, confidence=0.95))]
pub fn py_simple_exp_smoothing(
    data: Vec<f64>,
    alpha: f64,
    horizon: usize,
    confidence: f64,
) -> PyResult<PyForecastResult> {
    match simple_exp_smoothing(&data, alpha, horizon, confidence) {
        Ok(result) => Ok(PyForecastResult {
            forecasts: result.forecasts,
            lower_bound: result.lower_bound,
            upper_bound: result.upper_bound,
            confidence_level: result.confidence_level,
        }),
        Err(e) => Err(PyValueError::new_err(e)),
    }
}

#[pyfunction]
#[pyo3(signature = (data, alpha=0.3, beta=0.1, horizon=12, confidence=0.95))]
pub fn py_holt_linear(
    data: Vec<f64>,
    alpha: f64,
    beta: f64,
    horizon: usize,
    confidence: f64,
) -> PyResult<PyForecastResult> {
    match holt_linear(&data, alpha, beta, horizon, confidence) {
        Ok(result) => Ok(PyForecastResult {
            forecasts: result.forecasts,
            lower_bound: result.lower_bound,
            upper_bound: result.upper_bound,
            confidence_level: result.confidence_level,
        }),
        Err(e) => Err(PyValueError::new_err(e)),
    }
}

#[pyfunction]
#[pyo3(signature = (data, alpha=0.3, beta=0.1, gamma=0.2, period=12, horizon=12, seasonal_type="additive", confidence=0.95))]
pub fn py_holt_winters(
    data: Vec<f64>,
    alpha: f64,
    beta: f64,
    gamma: f64,
    period: usize,
    horizon: usize,
    seasonal_type: &str,
    confidence: f64,
) -> PyResult<PyForecastResult> {
    match holt_winters(
        &data,
        alpha,
        beta,
        gamma,
        period,
        horizon,
        seasonal_type,
        confidence,
    ) {
        Ok(result) => Ok(PyForecastResult {
            forecasts: result.forecasts,
            lower_bound: result.lower_bound,
            upper_bound: result.upper_bound,
            confidence_level: result.confidence_level,
        }),
        Err(e) => Err(PyValueError::new_err(e)),
    }
}

#[pymodule]
fn mathemixx_core(_py: Python, module: &PyModule) -> PyResult<()> {
    module.add_class::<PyDataSet>()?;
    module.add_class::<PyOlsResult>()?;
    module.add_class::<PySummaryRow>()?;
    module.add_class::<PyCoefficientRow>()?;
    module.add_class::<PyColumnInfo>()?;
    module.add_class::<PyCorrelationMatrix>()?;
    module.add_class::<PyEnhancedSummary>()?;
    module.add_class::<PyFrequencyRow>()?;
    module.add_class::<PyFrequencyTable>()?;
    module.add_class::<PyTTestResult>()?;
    // Phase 6: Visualization classes
    module.add_class::<PyResidualFittedData>()?;
    module.add_class::<PyQQPlotData>()?;
    module.add_class::<PyScaleLocationData>()?;
    module.add_class::<PyResidualsLeverageData>()?;
    module.add_class::<PyResidualHistogramData>()?;
    module.add_class::<PyBoxPlotData>()?;
    module.add_class::<PyHistogramData>()?;
    module.add_class::<PyHeatmapData>()?;
    module.add_class::<PyPairPlotData>()?;
    module.add_class::<PyViolinPlotData>()?;
    module.add_function(wrap_pyfunction!(load_csv, module)?)?;
    module.add_function(wrap_pyfunction!(summarize_dataset, module)?)?;
    module.add_function(wrap_pyfunction!(regress_dataset, module)?)?;

    // Phase 7: Time Series functions and classes
    module.add_class::<PyADFResult>()?;
    module.add_class::<PyKPSSResult>()?;
    module.add_class::<PyDecompositionResult>()?;
    module.add_class::<PyForecastResult>()?;
    module.add_function(wrap_pyfunction!(py_lag, module)?)?;
    module.add_function(wrap_pyfunction!(py_diff, module)?)?;
    module.add_function(wrap_pyfunction!(py_sma, module)?)?;
    module.add_function(wrap_pyfunction!(py_ema, module)?)?;
    module.add_function(wrap_pyfunction!(py_wma, module)?)?;
    module.add_function(wrap_pyfunction!(py_rolling_mean, module)?)?;
    module.add_function(wrap_pyfunction!(py_rolling_std, module)?)?;
    module.add_function(wrap_pyfunction!(py_rolling_min, module)?)?;
    module.add_function(wrap_pyfunction!(py_rolling_max, module)?)?;
    module.add_function(wrap_pyfunction!(py_acf, module)?)?;
    module.add_function(wrap_pyfunction!(py_pacf, module)?)?;
    module.add_function(wrap_pyfunction!(py_ljung_box_test, module)?)?;
    module.add_function(wrap_pyfunction!(py_adf_test, module)?)?;
    module.add_function(wrap_pyfunction!(py_kpss_test, module)?)?;
    module.add_function(wrap_pyfunction!(py_seasonal_decompose, module)?)?;
    module.add_function(wrap_pyfunction!(py_simple_exp_smoothing, module)?)?;
    module.add_function(wrap_pyfunction!(py_holt_linear, module)?)?;
    module.add_function(wrap_pyfunction!(py_holt_winters, module)?)?;

    Ok(())
}

fn mathemixx_error_to_pyerr(value: MatheMixxError) -> PyErr {
    match value {
        MatheMixxError::ColumnNotFound(name) => {
            PyValueError::new_err(format!("Column '{name}' not found"))
        }
        MatheMixxError::EmptyIndependentSet => {
            PyValueError::new_err("At least one independent variable is required")
        }
        MatheMixxError::RankDeficient => PyRuntimeError::new_err(
            "Design matrix is rank deficient; consider dropping collinear variables",
        ),
        MatheMixxError::Unsupported(msg) => {
            PyRuntimeError::new_err(format!("Unsupported operation: {msg}"))
        }
        MatheMixxError::InvalidInput(msg) => PyValueError::new_err(format!("Invalid input: {msg}")),
        MatheMixxError::UnsupportedOperation(msg) => {
            PyRuntimeError::new_err(format!("Unsupported operation: {msg}"))
        }
        MatheMixxError::Io(err) => PyRuntimeError::new_err(err.to_string()),
        MatheMixxError::Polars(err) => PyRuntimeError::new_err(err.to_string()),
        MatheMixxError::Linalg(err) => PyRuntimeError::new_err(err),
    }
}
