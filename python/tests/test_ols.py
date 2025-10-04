from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

try:
    from mathemixx_core import DataSet
except ImportError:  # pragma: no cover - compiled extension required
    pytest.skip("mathemixx_core extension not built", allow_module_level=True)


DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "sample_regression.csv"


def test_ols_matches_statsmodels(tmp_path: Path) -> None:
    dataset = DataSet.from_csv(str(DATA_PATH))
    result = dataset.regress_ols("y", ["x1", "x2"], True)

    df = pd.read_csv(DATA_PATH)
    X = sm.add_constant(df[["x1", "x2"]])
    model = sm.OLS(df["y"], X).fit()

    np.testing.assert_allclose(result.coefficients(), model.params.values, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(result.stderr(), model.bse.values, rtol=1e-9, atol=1e-9)
    assert pytest.approx(result.r_squared(), rel=1e-12) == model.rsquared
    assert pytest.approx(result.adj_r_squared(), rel=1e-12) == model.rsquared_adj

    output_file = tmp_path / "ols.csv"
    result.export_csv(output_file.as_posix())
    assert output_file.exists()

    tex_file = tmp_path / "ols.tex"
    result.export_tex(tex_file.as_posix())
    assert tex_file.exists()

    json_blob = json.loads(result.to_json())
    assert "coefficients" in json_blob
    assert len(json_blob["coefficients"]) == 3

    table = pd.DataFrame(
        [
            {
                "variable": row.variable,
                "coefficient": row.coefficient,
                "std_error": row.std_error,
            }
            for row in result.table()
        ]
    )
    assert list(table["variable"]) == ["Intercept", "x1", "x2"]
