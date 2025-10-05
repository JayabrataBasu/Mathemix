// Time series analysis module
// Implements core time series operations, tests, and forecasting methods

pub mod autocorr;
pub mod decomposition;
pub mod forecasting;
pub mod operations;
pub mod stationarity;
pub mod structures;

pub use autocorr::{acf, ljung_box_test, pacf};
pub use decomposition::{seasonal_decompose, DecompType, DecompositionResult};
pub use forecasting::{holt_linear, holt_winters, simple_exp_smoothing, ForecastResult};
pub use operations::{
    diff, ema, lag, rolling_max, rolling_mean, rolling_min, rolling_std, sma, wma,
};
pub use stationarity::{adf_test, kpss_test, ADFResult, KPSSResult};
pub use structures::TimeSeries;
