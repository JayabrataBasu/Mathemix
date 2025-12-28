// Time series analysis module
// Implements core time series operations, tests, and forecasting methods

pub mod arima;
pub mod autocorr;
pub mod cointegration;
pub mod decomposition;
pub mod forecasting;
pub mod garch;
pub mod granger;
pub mod operations;
pub mod stationarity;
pub mod structures;
pub mod var;

pub use arima::ArimaModel;
pub use autocorr::{acf, ljung_box_test, pacf, LjungBoxResult};
pub use cointegration::{engle_granger_test, johansen_test, EngleGrangerResult, JohansenResult};
pub use decomposition::{seasonal_decompose, DecompType, DecompositionResult};
pub use forecasting::{holt_linear, holt_winters, simple_exp_smoothing, ForecastResult};
pub use garch::GarchModel;
pub use granger::{granger_causality_test, GrangerResult};
pub use operations::{
    diff, ema, lag, rolling_max, rolling_mean, rolling_min, rolling_std, sma, wma,
};
pub use stationarity::{adf_test, kpss_test, ADFResult, KPSSResult};
pub use structures::TimeSeries;
pub use var::VarModel;
