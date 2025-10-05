// Time series data structures

use std::collections::HashMap;

/// Represents a time series with optional time index
#[derive(Debug, Clone)]
pub struct TimeSeries {
    /// The actual data values
    pub values: Vec<f64>,
    /// Optional time index (timestamps, dates, etc.)
    pub index: Option<Vec<String>>,
    /// Optional metadata
    pub metadata: HashMap<String, String>,
}

impl TimeSeries {
    /// Create a new time series from values
    pub fn new(values: Vec<f64>) -> Self {
        TimeSeries {
            values,
            index: None,
            metadata: HashMap::new(),
        }
    }

    /// Create a new time series with time index
    pub fn with_index(values: Vec<f64>, index: Vec<String>) -> Self {
        assert_eq!(
            values.len(),
            index.len(),
            "Values and index must have the same length"
        );
        TimeSeries {
            values,
            index: Some(index),
            metadata: HashMap::new(),
        }
    }

    /// Get the length of the time series
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Check if the time series is empty
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Add metadata
    pub fn with_metadata(mut self, key: String, value: String) -> Self {
        self.metadata.insert(key, value);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_timeseries_new() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let ts = TimeSeries::new(data.clone());
        assert_eq!(ts.len(), 5);
        assert_eq!(ts.values, data);
        assert!(ts.index.is_none());
    }

    #[test]
    fn test_timeseries_with_index() {
        let data = vec![1.0, 2.0, 3.0];
        let index = vec![
            "2024-01".to_string(),
            "2024-02".to_string(),
            "2024-03".to_string(),
        ];
        let ts = TimeSeries::with_index(data.clone(), index.clone());
        assert_eq!(ts.len(), 3);
        assert_eq!(ts.values, data);
        assert_eq!(ts.index.unwrap(), index);
    }

    #[test]
    #[should_panic(expected = "Values and index must have the same length")]
    fn test_timeseries_length_mismatch() {
        let data = vec![1.0, 2.0, 3.0];
        let index = vec!["2024-01".to_string(), "2024-02".to_string()];
        TimeSeries::with_index(data, index);
    }
}
