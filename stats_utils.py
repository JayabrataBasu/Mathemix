"""Legacy helpers retained for backward compatibility.

The MatheMixX project now exposes all analytics functionality through the
Rust-powered `mathemixx_core` package. These placeholders remain to prevent
import errors in existing notebooks and scripts while signaling the migration
path.
"""

from __future__ import annotations


class LegacyModuleRemovedError(RuntimeError):
    """Raised when a deprecated helper is used."""


def _raise() -> None:
    raise LegacyModuleRemovedError(
        "The lightweight Streamlit-era helpers have been superseded. "
        "Use the `mathemixx_core` bindings exposed via PyO3 instead."
    )


def load_csv(*args, **kwargs):  # type: ignore[unused-argument]
    _raise()


def summarize(*args, **kwargs):  # type: ignore[unused-argument]
    _raise()


def regress(*args, **kwargs):  # type: ignore[unused-argument]
    _raise()
