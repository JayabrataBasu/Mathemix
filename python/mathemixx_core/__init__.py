"""Python bindings for the MatheMixX core engine."""
from __future__ import annotations

from .mathemixx_core import (  # type: ignore[attr-defined]
    CoefficientRow,
    DataSet,
    OlsResult,
    SummaryRow,
    load_csv,
    regress_dataset,
    summarize_dataset,
)

__all__ = [
    "CoefficientRow",
    "DataSet",
    "OlsResult",
    "SummaryRow",
    "load_csv",
    "summarize_dataset",
    "regress_dataset",
]
