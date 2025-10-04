use std::f64;
use std::fs::File;
use std::io::Write;
use std::sync::Arc;

use ::mathemixx_core::{
    regress, summarize_numeric, DataSet, MatheMixxError, OlsOptions, OlsResult,
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

#[pymodule]
fn mathemixx_core(_py: Python, module: &PyModule) -> PyResult<()> {
    module.add_class::<PyDataSet>()?;
    module.add_class::<PyOlsResult>()?;
    module.add_class::<PySummaryRow>()?;
    module.add_class::<PyCoefficientRow>()?;
    module.add_function(wrap_pyfunction!(load_csv, module)?)?;
    module.add_function(wrap_pyfunction!(summarize_dataset, module)?)?;
    module.add_function(wrap_pyfunction!(regress_dataset, module)?)?;
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
        MatheMixxError::Io(err) => PyRuntimeError::new_err(err.to_string()),
        MatheMixxError::Polars(err) => PyRuntimeError::new_err(err.to_string()),
        MatheMixxError::Linalg(err) => PyRuntimeError::new_err(err),
    }
}
