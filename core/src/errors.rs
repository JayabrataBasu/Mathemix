use std::io;

use thiserror::Error;

pub type Result<T> = std::result::Result<T, MatheMixxError>;

#[derive(Debug, Error)]
pub enum MatheMixxError {
    #[error("I/O error: {0}")]
    Io(#[from] io::Error),

    #[error("Polars error: {0}")]
    Polars(#[from] polars::error::PolarsError),

    #[error("Column '{0}' not found in DataFrame")]
    ColumnNotFound(String),

    #[error("No independent variables were provided")]
    EmptyIndependentSet,

    #[error("Design matrix is rank deficient")]
    RankDeficient,

    #[error("Linear algebra error: {0}")]
    Linalg(String),

    #[error("Operation unsupported: {0}")]
    Unsupported(&'static str),

    #[error("Invalid input: {0}")]
    InvalidInput(String),

    #[error("Unsupported operation: {0}")]
    UnsupportedOperation(String),
}

impl From<ndarray_linalg::error::LinalgError> for MatheMixxError {
    fn from(value: ndarray_linalg::error::LinalgError) -> Self {
        Self::Linalg(value.to_string())
    }
}
