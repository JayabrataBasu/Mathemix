"""Mathemix Streamlit application."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

from stats_utils import load_csv, regress, summarize

BASE_DIR = Path(__file__).parent
SAMPLE_DATA_PATH = BASE_DIR / "data" / "example.csv"

st.set_page_config(page_title="Mathemix", layout="wide")


def get_dataframe(uploaded_file: st.UploadedFile | None) -> pd.DataFrame | None:
    if uploaded_file is not None:
        return load_csv(uploaded_file)
    if SAMPLE_DATA_PATH.exists():
        return load_csv(SAMPLE_DATA_PATH)
    return None


def get_numeric_columns(df: pd.DataFrame) -> list[str]:
    return df.select_dtypes(include="number").columns.tolist()


def render_summary(df: pd.DataFrame) -> None:
    summary_df = summarize(df)
    if summary_df.empty:
        st.info("No numeric columns available for summary.")
        return
    st.dataframe(summary_df, use_container_width=True)


def render_regression(df: pd.DataFrame, dependent: str, independents: Iterable[str]) -> None:
    try:
        results = regress(df, dependent, independents)
    except ValueError as exc:
        st.warning(str(exc))
        return

    st.subheader("Regression Results")
    st.text(results.summary())


def render_plot(df: pd.DataFrame, dependent: str, independents: Iterable[str], plot_type: str) -> None:
    if plot_type == "None":
        return

    plt.figure(figsize=(6, 4))

    if plot_type == "Histogram" and dependent:
        sns.histplot(df[dependent].dropna(), kde=False)
        plt.xlabel(dependent)
        plt.title(f"Histogram of {dependent}")
    elif plot_type == "Scatter" and dependent and independents:
        x_var = next((col for col in independents if df[col].dtype.kind in "if"), None)
        if x_var is None:
            st.info("Select at least one numeric independent variable for scatter plot.")
            plt.close()
            return
        sns.scatterplot(x=df[x_var], y=df[dependent])
        plt.xlabel(x_var)
        plt.ylabel(dependent)
        plt.title(f"{dependent} vs {x_var}")
    else:
        plt.close()
        return

    st.pyplot(plt.gcf(), clear_figure=True)


st.title("Mathemix: Lightweight Regression Explorer")

with st.sidebar:
    st.header("Workflow")
    uploaded_file = st.file_uploader("Load CSV", type="csv")
    data_source = "Uploaded" if uploaded_file else "Sample dataset"
    st.caption(f"Using: {data_source}")

    df = get_dataframe(uploaded_file)

    dependent = ""
    independents: list[str] = []
    plot_choice = "None"

    if df is not None and not df.empty:
        numeric_cols = get_numeric_columns(df)
        all_cols = df.columns.tolist()

        dependent = st.selectbox("Dependent variable", options=all_cols, index=all_cols.index(numeric_cols[0]) if numeric_cols else 0)
        independents = st.multiselect(
            "Independent variables",
            options=[col for col in all_cols if col != dependent],
            default=[col for col in numeric_cols if col != dependent][:2],
        )
        plot_choice = st.selectbox("Plot", options=["None", "Histogram", "Scatter"], index=0)

summarize_triggered = st.button("Summarize Data", disabled=df is None or df.empty)
regression_triggered = st.button("Run Regression", disabled=df is None or df.empty)

if df is None:
    st.info("Upload a CSV file to get started. A sample dataset will load automatically when available.")
else:
    st.subheader("Data Preview")
    st.dataframe(df.head(), use_container_width=True)

    if summarize_triggered:
        st.subheader("Descriptive Statistics")
        render_summary(df)

    if regression_triggered and dependent:
        render_regression(df.dropna(subset=[dependent]), dependent, independents)

    render_plot(df.dropna(subset=[dependent]) if dependent else df, dependent, independents, plot_choice)
