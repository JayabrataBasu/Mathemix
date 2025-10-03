# Mathemix

A razor-sharp, lightweight Streamlit app that mimics a simplified Stata workflow for quick exploratory regression analysis.

## Features

- Upload a CSV or use the bundled sample dataset.
- Preview the data instantly.
- Summarize numeric columns (mean, sd, min, max).
- Run Ordinary Least Squares (OLS) regressions via statsmodels.
- Visualize results with a histogram or scatter plot.

## Quick start

```bash
pip install -r requirements.txt
streamlit run app.py
```

On first launch the app loads the sample dataset from `data/example.csv`. Use the sidebar to switch to your own CSV, choose variables, run summaries, regressions, and optional plots.

## Design principles

- Minimal dependencies beyond the core analytics stack.
- Lean data processing—only numeric columns are materialized for statistics and models.
- Efficient plotting with seaborn/matplotlib kept to essential views.
