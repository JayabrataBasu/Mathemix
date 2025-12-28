pub mod dataframe;
pub mod errors;
pub mod hypothesis;
pub mod manipulation;
pub mod ols;
pub mod statistics;
pub mod summary;
pub mod timeseries;
pub mod visualization;

pub use dataframe::DataSet;
pub use errors::{MatheMixxError, Result};
pub use hypothesis::{AnovaResult, ChiSquareResult, TTestResult};
pub use manipulation::{ColumnInfo, ColumnType, FilterCondition, Transform};
pub use ols::{regress, ColumnTransform, OlsOptions, OlsResult, RobustStdError};
pub use statistics::{
    CorrelationMatrix, CorrelationMethod, EnhancedSummary, FrequencyRow, FrequencyTable,
};
pub use summary::summarize_numeric;
pub use timeseries::{
    acf, adf_test, diff, ema, holt_linear, holt_winters, kpss_test, lag, ljung_box_test, pacf,
    rolling_max, rolling_mean, rolling_min, rolling_std, seasonal_decompose, simple_exp_smoothing,
    sma, wma, ADFResult, DecompType, DecompositionResult, ForecastResult, KPSSResult, LjungBoxResult, TimeSeries,
};
pub use visualization::{
    BoxPlotData, HeatmapData, HistogramData, PairPlotData, QQPlotData, ResidualFittedData,
    ResidualHistogramData, ResidualsLeverageData, ScaleLocationData, ViolinPlotData,
};
