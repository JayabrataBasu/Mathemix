use polars::prelude::*;

use crate::{dataframe::DataSet, errors::Result};

pub fn summarize_numeric(dataset: &DataSet) -> Result<DataFrame> {
    let numeric_cols = dataset.numeric_column_names();
    let mut names = Vec::with_capacity(numeric_cols.len());
    let mut means = Vec::with_capacity(numeric_cols.len());
    let mut sds = Vec::with_capacity(numeric_cols.len());
    let mut mins = Vec::with_capacity(numeric_cols.len());
    let mut maxs = Vec::with_capacity(numeric_cols.len());

    for col_name in numeric_cols {
        let series = dataset.column(&col_name)?;
        let float_series = series.cast(&DataType::Float64)?;
        let chunked = float_series.f64().expect("cast to f64");
        let mean = chunked.mean();
        let std = chunked.std(1);
        let min = chunked.min();
        let max = chunked.max();

        names.push(col_name);
        means.push(mean.unwrap_or(f64::NAN));
        sds.push(std.unwrap_or(f64::NAN));
        mins.push(min.unwrap_or(f64::NAN));
        maxs.push(max.unwrap_or(f64::NAN));
    }

    Ok(DataFrame::new(vec![
        Series::new("variable".into(), names).into(),
        Series::new("mean".into(), means).into(),
        Series::new("sd".into(), sds).into(),
        Series::new("min".into(), mins).into(),
        Series::new("max".into(), maxs).into(),
    ])?)
}
