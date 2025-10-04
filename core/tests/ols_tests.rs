use approx::assert_abs_diff_eq;
use mathemixx_core::{regress, DataSet, OlsOptions};

fn coeffs() -> Vec<f64> {
    vec![2.182641921397378, 0.47314410480349334, 1.490174672489083]
}

fn stderrs() -> Vec<f64> {
    vec![0.2851315510955933, 0.051567873995546104, 0.0735137631596075]
}

#[test]
fn ols_matches_reference() {
    let data_path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("data")
        .join("sample_regression.csv");
    let dataset = DataSet::from_csv(&data_path).expect("load sample data");
    let independents = vec!["x1".to_string(), "x2".to_string()];
    let result = regress(&dataset, "y", &independents, OlsOptions::default()).expect("ols");

    for (idx, expected) in coeffs().iter().enumerate() {
        assert_abs_diff_eq!(result.coefficients[idx], expected, epsilon = 1e-8);
    }

    for (idx, expected) in stderrs().iter().enumerate() {
        assert_abs_diff_eq!(result.stderr[idx], expected, epsilon = 1e-8);
    }

    assert_abs_diff_eq!(result.r_squared, 0.9920821826668708_f64, epsilon = 1e-10);
    assert_abs_diff_eq!(result.adj_r_squared, 0.988915055733619_f64, epsilon = 1e-10);
    assert_abs_diff_eq!(result.f_stat, 313.24358119372994_f64, epsilon = 1e-6);
}
