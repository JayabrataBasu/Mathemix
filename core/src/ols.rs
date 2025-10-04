use std::collections::BTreeMap;

use ndarray::{Array1, Array2, Axis};
use ndarray_linalg::{Diag, InverseInto, QRInto, SVDInto, SolveTriangularInto, UPLO};
use statrs::distribution::{ContinuousCDF, FisherSnedecor, StudentsT};

use crate::{
    dataframe::DataSet,
    errors::{MatheMixxError, Result},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum RobustStdError {
    Hc0,
    Hc1,
    Hc2,
    Hc3,
}

impl RobustStdError {
    pub fn as_str(&self) -> &'static str {
        match self {
            RobustStdError::Hc0 => "HC0",
            RobustStdError::Hc1 => "HC1",
            RobustStdError::Hc2 => "HC2",
            RobustStdError::Hc3 => "HC3",
        }
    }
}

#[derive(Debug, Clone)]
pub struct OlsOptions {
    pub add_intercept: bool,
    pub center: bool,
    pub scale: bool,
    pub robust: bool,
}

impl Default for OlsOptions {
    fn default() -> Self {
        Self {
            add_intercept: true,
            center: true,
            scale: false,
            robust: true,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ColumnTransform {
    pub name: String,
    pub mean: f64,
    pub scale: f64,
}

#[derive(Debug, Clone)]
pub struct OlsResult {
    pub dependent: String,
    pub regressors: Vec<String>,
    pub coefficients: Vec<f64>,
    pub stderr: Vec<f64>,
    pub robust_stderr: BTreeMap<RobustStdError, Vec<f64>>,
    pub t_values: Vec<f64>,
    pub p_values: Vec<f64>,
    pub conf_int: Vec<(f64, f64)>,
    pub r_squared: f64,
    pub adj_r_squared: f64,
    pub f_stat: f64,
    pub f_p_value: f64,
    pub df_model: f64,
    pub df_resid: f64,
    pub sigma2: f64,
    pub nobs: usize,
    pub residuals: Vec<f64>,
    pub fitted_values: Vec<f64>,
    pub hat_diagonal: Vec<f64>,
    pub transforms: Vec<ColumnTransform>,
}

pub fn regress(
    dataset: &DataSet,
    dependent: &str,
    independents: &[String],
    options: OlsOptions,
) -> Result<OlsResult> {
    if independents.is_empty() {
        return Err(MatheMixxError::EmptyIndependentSet);
    }

    let mut column_names: Vec<&str> = Vec::with_capacity(independents.len() + 1);
    column_names.push(dependent);
    for name in independents {
        column_names.push(name.as_str());
    }

    let numeric_frame = dataset.prepare_numeric_frame(&column_names)?;
    let nrows = numeric_frame.height();
    if nrows == 0 {
        return Err(MatheMixxError::Unsupported(
            "No observations remain after dropping missing values",
        ));
    }

    let dep_series = numeric_frame.column(dependent)?;
    let y_chunked = dep_series.f64().expect("dependent cast");
    let mut y = Array1::<f64>::zeros(nrows);
    for (idx, val) in y_chunked.into_no_null_iter().enumerate() {
        y[idx] = val;
    }

    let mut regressors = Vec::new();
    let mut design =
        Array2::<f64>::zeros((nrows, independents.len() + options.add_intercept as usize));
    let mut transforms = Vec::new();

    let mut col_offset = 0;
    if options.add_intercept {
        design.column_mut(0).fill(1.0);
        regressors.push("Intercept".to_string());
        col_offset = 1;
    }

    for (idx, name) in independents.iter().enumerate() {
        let series = numeric_frame.column(name.as_str())?;
        let chunked = series.f64().expect("independent cast");
        let mut column = Array1::<f64>::zeros(nrows);
        for (row, val) in chunked.into_no_null_iter().enumerate() {
            column[row] = val;
        }

        let mean = if options.center {
            column.sum() / nrows as f64
        } else {
            0.0
        };

        if options.center {
            column.mapv_inplace(|v| v - mean);
        }

        let mut scale = 1.0;
        if options.scale {
            let mut var = 0.0;
            for v in column.iter() {
                var += v * v;
            }
            scale = (var / (nrows as f64 - 1.0)).sqrt();
            if scale.is_finite() && scale > f64::EPSILON {
                column.mapv_inplace(|v| v / scale);
            } else {
                scale = 1.0;
            }
        }

        transforms.push(ColumnTransform {
            name: name.clone(),
            mean,
            scale,
        });

        design.column_mut(col_offset + idx).assign(&column.view());
        regressors.push(name.clone());
    }

    let (beta_scaled, xtx_inv) = solve_via_qr(&design, &y)?;
    let fitted = design.dot(&beta_scaled);
    let residuals = &y - &fitted;

    let n = nrows as f64;
    let p = design.ncols() as f64;
    let df_resid = n - p;
    if df_resid <= 0.0 {
        return Err(MatheMixxError::RankDeficient);
    }
    let df_model = if options.add_intercept { p - 1.0 } else { p };
    let df_model = df_model.max(1.0);

    let ssr = residuals.mapv(|v| v * v).sum();
    let sigma2 = ssr / df_resid;

    let cov_beta_scaled = xtx_inv.clone() * sigma2;

    let mut beta = beta_scaled.clone();
    let mut cov_beta = cov_beta_scaled.clone();

    if options.center || options.scale {
        adjust_for_transforms_with_cov(
            options.add_intercept,
            &mut beta,
            &mut cov_beta,
            &transforms,
        );
    }

    let stderr = cov_beta.diag().iter().map(|v| v.sqrt()).collect::<Vec<_>>();

    let (t_values, p_values, conf_int) = compute_inference(&beta, &stderr, df_resid);

    let y_mean = y.sum() / n;
    let sst = y.mapv(|v| (v - y_mean) * (v - y_mean)).sum();
    let r_squared = 1.0 - ssr / sst;
    let adj_r_squared = 1.0 - (1.0 - r_squared) * ((n - 1.0) / df_resid);

    let ss_model = (sst - ssr).max(0.0);
    let ms_model = ss_model / df_model;
    let ms_error = sigma2;
    let f_stat = ms_model / ms_error;
    let f_p_value = FisherSnedecor::new(df_model, df_resid)
        .ok()
        .map(|f| 1.0 - f.cdf(f_stat))
        .unwrap_or(f64::NAN);

    let hat_diag = compute_hat_diagonal(&design, &xtx_inv);

    let mut robust = BTreeMap::new();
    if options.robust {
        let robust_cov =
            compute_robust_covariances(&design, &residuals, &xtx_inv, &hat_diag, df_resid);
        for (kind, cov) in robust_cov {
            let stderr_kind = cov.diag().iter().map(|v| v.sqrt()).collect::<Vec<_>>();
            robust.insert(kind, stderr_kind);
        }
    }

    Ok(OlsResult {
        dependent: dependent.to_string(),
        regressors,
        coefficients: beta.into_raw_vec(),
        stderr,
        robust_stderr: robust,
        t_values,
        p_values,
        conf_int,
        r_squared,
        adj_r_squared,
        f_stat,
        f_p_value,
        df_model,
        df_resid,
        sigma2,
        nobs: nrows,
        residuals: residuals.to_vec(),
        fitted_values: fitted.to_vec(),
        hat_diagonal: hat_diag,
        transforms,
    })
}

fn solve_via_qr(design: &Array2<f64>, y: &Array1<f64>) -> Result<(Array1<f64>, Array2<f64>)> {
    match design.clone().qr_into() {
        Ok((q, r)) => {
            let qty = q.t().dot(y);
            let beta = r.solve_triangular_into(UPLO::Upper, Diag::NonUnit, qty)?;
            let xtx = design.t().dot(design);
            let xtx_inv = xtx.inv_into()?;
            Ok((beta, xtx_inv))
        }
        Err(err) => {
            log::warn!("QR decomposition failed ({err}), attempting SVD fallback");
            let svd = design.clone().svd_into(true, true)?;
            let (u_opt, s, vt_opt) = svd;
            let (u, vt) = match (u_opt, vt_opt) {
                (Some(u), Some(vt)) => (u, vt),
                _ => return Err(MatheMixxError::RankDeficient),
            };

            let max_sigma = s.iter().cloned().fold(0.0, f64::max);
            let tol = f64::EPSILON * s.len() as f64 * max_sigma.max(1.0);
            let mut sigma_inv = Array2::<f64>::zeros((vt.nrows(), u.nrows()));
            for (i, value) in s.iter().enumerate() {
                if *value > tol {
                    sigma_inv[(i, i)] = 1.0 / value;
                }
            }

            let pseudo_inv = vt.t().dot(&sigma_inv.dot(&u.t()));
            let beta = pseudo_inv.dot(y);
            let xtx_inv = pseudo_inv.dot(&pseudo_inv.t());
            Ok((beta, xtx_inv))
        }
    }
}

fn adjust_for_transforms_with_cov(
    has_intercept: bool,
    beta: &mut Array1<f64>,
    cov: &mut Array2<f64>,
    transforms: &[ColumnTransform],
) {
    if !has_intercept || transforms.is_empty() {
        return;
    }

    let p = beta.len();
    // Build Jacobian matrix J where beta_original = J * beta_scaled
    let mut jacobian = Array2::<f64>::eye(p);

    let intercept_idx = 0;

    for (i, transform) in transforms.iter().enumerate() {
        let idx = i + 1; // slope coefficients start at index 1
        let scale = if transform.scale != 0.0 && transform.scale.is_finite() {
            transform.scale
        } else {
            1.0
        };

        // J[idx, idx] = 1/scale (for rescaling slope)
        if scale != 0.0 {
            jacobian[(idx, idx)] = 1.0 / scale;
        }

        // J[0, idx] = -mean / scale (intercept depends on all slopes due to centering)
        if scale != 0.0 {
            jacobian[(intercept_idx, idx)] = -transform.mean / scale;
        }
    }

    // Transform beta: beta_new = J * beta_old
    let beta_transformed = jacobian.dot(beta);
    beta.assign(&beta_transformed);

    // Transform covariance: Cov_new = J * Cov_old * J^T
    let cov_transformed = jacobian.dot(&cov.dot(&jacobian.t()));
    cov.assign(&cov_transformed);
}

fn compute_inference(
    beta: &Array1<f64>,
    stderr: &[f64],
    df_resid: f64,
) -> (Vec<f64>, Vec<f64>, Vec<(f64, f64)>) {
    let dof = df_resid;
    let student = StudentsT::new(0.0, 1.0, dof).expect("Student-t distribution");
    let crit = student.inverse_cdf(0.975);

    let mut t_values = Vec::with_capacity(beta.len());
    let mut p_values = Vec::with_capacity(beta.len());
    let mut ci = Vec::with_capacity(beta.len());

    for (idx, &coef) in beta.iter().enumerate() {
        let se = stderr[idx];
        let t = coef / se;
        let p = 2.0 * (1.0 - student.cdf(t.abs()));
        t_values.push(t);
        p_values.push(p);
        ci.push((coef - crit * se, coef + crit * se));
    }

    (t_values, p_values, ci)
}

fn compute_hat_diagonal(design: &Array2<f64>, xtx_inv: &Array2<f64>) -> Vec<f64> {
    let mut hat = Vec::with_capacity(design.nrows());
    for row in design.outer_iter() {
        let v = row.to_owned();
        let tmp = xtx_inv.dot(&v);
        let leverage = v.dot(&tmp);
        hat.push(leverage);
    }
    hat
}

fn compute_robust_covariances(
    design: &Array2<f64>,
    residuals: &Array1<f64>,
    xtx_inv: &Array2<f64>,
    hat_diag: &[f64],
    df_resid: f64,
) -> BTreeMap<RobustStdError, Array2<f64>> {
    let mut results = BTreeMap::new();
    let xtx_inv_ref = xtx_inv.to_owned();

    let mut meat_hc0 = Array2::<f64>::zeros((design.ncols(), design.ncols()));
    let mut meat_hc2 = meat_hc0.clone();
    let mut meat_hc3 = meat_hc0.clone();

    for (i, row) in design.outer_iter().enumerate() {
        let x_vec = row.to_owned();
        let x_i = x_vec.clone().insert_axis(Axis(1));
        let r_i = residuals[i];
        let outer = x_i.dot(&x_i.t());
        let contrib = outer * (r_i * r_i);
        meat_hc0 += &contrib;

        let leverage = hat_diag[i];
        let adj = 1.0 - leverage;
        let adj_sq = if adj.abs() < f64::EPSILON {
            f64::EPSILON
        } else {
            adj * adj
        };

        meat_hc2 += &(&contrib / adj);
        meat_hc3 += &(&contrib / adj_sq);
    }

    let scale_hc1 = design.nrows() as f64 / df_resid;

    let cov_hc0 = xtx_inv_ref.dot(&meat_hc0).dot(xtx_inv);
    let cov_hc1 = &cov_hc0 * scale_hc1;
    let cov_hc2 = xtx_inv_ref.dot(&meat_hc2).dot(xtx_inv);
    let cov_hc3 = xtx_inv_ref.dot(&meat_hc3).dot(xtx_inv);

    results.insert(RobustStdError::Hc0, cov_hc0);
    results.insert(RobustStdError::Hc1, cov_hc1);
    results.insert(RobustStdError::Hc2, cov_hc2);
    results.insert(RobustStdError::Hc3, cov_hc3);

    results
}
