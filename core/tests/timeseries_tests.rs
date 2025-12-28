#[cfg(test)]
mod tests {
    use mathemixx_core::timeseries::arima::ArimaModel;
    use mathemixx_core::timeseries::garch::GarchModel;
    use mathemixx_core::timeseries::var::VarModel;
    use mathemixx_core::timeseries::{engle_granger_test, johansen_test, granger_causality_test};

    #[test]
    fn test_arima_fit() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0];
        let model = ArimaModel::fit(&data, 1, 0, 1).unwrap();
        assert_eq!(model.p, 1);
        assert_eq!(model.d, 0);
        assert_eq!(model.q, 1);
        assert!(!model.ar_coeffs.is_empty());
    }

    #[test]
    fn test_arima_forecast() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0];
        let model = ArimaModel::fit(&data, 1, 0, 1).unwrap();
        let forecast = model.forecast(3, 0.95).unwrap();
        assert_eq!(forecast.forecasts.len(), 3);
        assert_eq!(forecast.lower_bound.len(), 3);
        assert_eq!(forecast.upper_bound.len(), 3);
    }

    #[test]
    fn test_garch_fit() {
        let returns = vec![0.01, -0.02, 0.03, -0.01, 0.02, 0.01, -0.03, 0.02, -0.01, 0.01, 0.02, -0.02, 0.03, -0.01, 0.02];
        let model = GarchModel::fit(&returns, 1, 1).unwrap();
        assert_eq!(model.p, 1);
        assert_eq!(model.q, 1);
        assert!(!model.alpha.is_empty());
        assert!(!model.beta.is_empty());
    }

    #[test]
    fn test_garch_forecast() {
        let returns = vec![0.01, -0.02, 0.03, -0.01, 0.02, 0.01, -0.03, 0.02, -0.01, 0.01, 0.02, -0.02, 0.03, -0.01, 0.02];
        let model = GarchModel::fit(&returns, 1, 1).unwrap();
        let forecast = model.forecast_volatility(3, 0.95).unwrap();
        assert_eq!(forecast.variances.len(), 3);
        assert_eq!(forecast.lower_bound.len(), 3);
        assert_eq!(forecast.upper_bound.len(), 3);
    }

    #[test]
    fn test_var_fit() {
        let data = vec![
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
            vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0],
        ];
        let model = VarModel::fit(&data, 1).unwrap();
        assert_eq!(model.lag_order, 1);
        assert!(!model.coefficients.is_empty());
    }

    #[test]
    fn test_var_forecast() {
        let data = vec![
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
            vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0],
        ];
        let model = VarModel::fit(&data, 1).unwrap();
        let forecast = model.forecast(3, 0.95).unwrap();
        assert_eq!(forecast.forecasts.len(), 3);
        assert_eq!(forecast.lower_bound.len(), 3);
        assert_eq!(forecast.upper_bound.len(), 3);
    }

    #[test]
    fn test_engle_granger() {
        let series1 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0];
        let series2 = vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0];
        let result = engle_granger_test(&series1, &series2).unwrap();
        assert!(result.t_statistic <= 0.0); // Should be negative for cointegrated
    }

    #[test]
    fn test_johansen() {
        let data = vec![
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0],
            vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0],
        ];
        let results = johansen_test(&data).unwrap();
        assert!(!results.is_empty());
    }

    #[test]
    fn test_granger_causality() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0];
        let y = vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0];
        let result = granger_causality_test(&x, &y, 2).unwrap();
        // Just check that it returns a result
        assert!(result.p_value >= 0.0 && result.p_value <= 1.0);
    }
}