"""Utility functions for Mathemix analytics workflow.

Functions are intentionally lightweight to keep the runtime lean.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import pandas as pd
import statsmodels.api as sm  # type: ignore


def load_csv(path_or_buffer: str | Path) -> pd.DataFrame:
    """Load a CSV file into a DataFrame with minimal overhead."""
    return pd.read_csv(path_or_buffer, low_memory=False)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Return descriptive statistics for numeric columns."""
    numeric_df = df.select_dtypes(include="number")
    if numeric_df.empty:
        return pd.DataFrame(columns=["mean", "std", "min", "max"])
    summary = numeric_df.agg(["mean", "std", "min", "max"]).T
    summary = summary.rename(columns={"std": "sd"})
    return summary[["mean", "sd", "min", "max"]]


def regress(df: pd.DataFrame, y: str, X: Iterable[str]) -> sm.regression.linear_model.RegressionResultsWrapper:
    """Run an OLS regression using statsmodels."""
    predictors: List[str] = [col for col in X if col != y]
    if not predictors:
        raise ValueError("At least one independent variable must be selected.")

    design = df[predictors].select_dtypes(include="number").dropna()
    target = df.loc[design.index, y]

    design_with_const = sm.add_constant(design, has_constant="add")
    model = sm.OLS(target, design_with_const)
    return model.fit()
