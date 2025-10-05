pub mod dataframe;
pub mod errors;
pub mod hypothesis;
pub mod manipulation;
pub mod ols;
pub mod statistics;
pub mod summary;

pub use dataframe::DataSet;
pub use errors::{MatheMixxError, Result};
pub use hypothesis::{AnovaResult, ChiSquareResult, TTestResult};
pub use manipulation::{ColumnInfo, ColumnType, FilterCondition, Transform};
pub use ols::{regress, ColumnTransform, OlsOptions, OlsResult, RobustStdError};
pub use statistics::{
    CorrelationMatrix, CorrelationMethod, EnhancedSummary, FrequencyRow, FrequencyTable,
};
pub use summary::summarize_numeric;
