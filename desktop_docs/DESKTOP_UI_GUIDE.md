# MatheMixX Desktop UI - User Guide

## Overview

The MatheMixX Desktop application provides a graphical user interface for statistical analysis and data visualization. With Phase 6 integration, the UI now includes comprehensive regression diagnostics and exploratory data analysis plots.

## Features

### 📊 Data Management
- **Load CSV files** via toolbar or File menu
- **Preview data** in tabular format (first 100 rows)
- **Variable selection** for regression analysis
- **Session logging** automatically tracks all operations

### 📈 Statistical Analysis

#### Summarize
- Descriptive statistics for all numeric variables
- Mean, standard deviation, min, max
- Results displayed in Results tab

#### Regression Analysis
1. Select dependent variable (Y)
2. Select one or more independent variables (X)
3. Click "Run Regression"
4. View coefficient table with:
   - Coefficients and standard errors
   - t-values and p-values
   - 95% confidence intervals
   - R², Adjusted R², F-statistic

### 🎨 Visualization Features (Phase 6)

#### Regression Diagnostic Plots

After running a regression, click **"📊 Diagnostic Plots"** to view:

1. **Residual vs Fitted Plot**
   - Detects non-linearity
   - Identifies heteroscedasticity
   - Highlights outliers
   - Reference line at zero

2. **Q-Q Plot (Normal Probability)**
   - Assesses normality of residuals
   - Points should fall on diagonal line
   - Deviations indicate non-normality

3. **Scale-Location Plot**
   - Tests homoscedasticity assumption
   - √|Standardized Residuals| vs Fitted
   - Horizontal spread indicates constant variance

4. **Residuals vs Leverage**
   - Identifies influential observations
   - Color-coded by Cook's distance
   - High Cook's D (>0.5) = influential points

5. **Residual Histogram**
   - Distribution of residuals
   - Normal curve overlay
   - Should be bell-shaped and centered at zero

6. **Summary Statistics Panel**
   - R² and Adjusted R²
   - Number of observations
   - Dependent variable name

**All 6 plots displayed in a 2×3 grid for comprehensive model assessment.**

#### Exploratory Data Analysis Plots

Select a variable from the list, then click:

1. **Box Plot**
   - Shows distribution via quartiles
   - Displays median (line) and mean (triangle)
   - Marks outliers beyond whiskers
   - IQR = Q3 - Q1

2. **Histogram + KDE**
   - Distribution visualization
   - Kernel Density Estimate overlay (smooth curve)
   - Helps assess normality and identify skewness

3. **Correlation Heatmap**
   - Pearson correlation matrix
   - Color-coded: red (positive), blue (negative)
   - Values range from -1 to +1
   - No variable selection needed (uses all numeric columns)

4. **Violin Plot**
   - Combines box plot + density plot
   - Shows full distribution shape
   - Wider = more data at that value

### 💾 Plot Export

Click **"💾 Save Plot"** in toolbar to export current plot:
- **PNG** - High-quality raster (default 300 DPI)
- **PDF** - Vector format for publications
- **SVG** - Vector format for web/editing
- **JPEG** - Compressed raster format

### 🎨 Plot Styling

Click **"🎨 Plot Style"** to change visual theme:
- **whitegrid** - White background with grid lines (default)
- **darkgrid** - Gray background with white grid
- **white** - Clean white background, no grid
- **dark** - Dark background for presentations
- **ticks** - Minimal style with tick marks only

## User Interface Layout

```
┌─────────────────────────────────────────────────────────────┐
│  Toolbar: [Open CSV] [Export .do] [Open log] │ [Save Plot] [Style] │
├───────────────┬─────────────────────────────────────────────┤
│               │  ╔═══════════════════════════════════════╗  │
│  Control      │  ║ Tabs: [Data Preview] [Results] [Plots]║  │
│  Panel        │  ╚═══════════════════════════════════════╝  │
│               │                                             │
│  Dataset:     │  Data Preview: Shows first 100 rows        │
│  filename.csv │                                             │
│               │  Results: Regression output & statistics   │
│  Dependent Y: │                                             │
│  [variable]   │  Plots: Interactive matplotlib canvas      │
│               │         (Displays all visualizations)      │
│  Independent: │                                             │
│  ☑ var1       │                                             │
│  ☑ var2       │                                             │
│  ☐ var3       │                                             │
│               │                                             │
│ [Summarize]   │                                             │
│ [Regression]  │                                             │
│ [📊 Diagnost] │                                             │
│               │                                             │
│ ───────────── │                                             │
│ Exploratory:  │                                             │
│ [Box Plot]    │                                             │
│ [Histogram]   │                                             │
│ [Heatmap]     │                                             │
│ [Violin]      │                                             │
│               │                                             │
│ Command:      │                                             │
│ [console...]  │                                             │
│ > command     │                                             │
└───────────────┴─────────────────────────────────────────────┘
```

## Workflow Examples

### Example 1: Basic Regression with Diagnostics

1. Click **"Open CSV"** → Select `data/iris.csv`
2. Set Dependent Y: `petal_length`
3. Select Independent: Check `sepal_length`, `sepal_width`
4. Click **"Run Regression"**
   - View results in Results tab
   - See scatter plot with fit line in Plots tab
5. Click **"📊 Diagnostic Plots"**
   - Comprehensive 6-plot diagnostic suite appears
   - Assess model quality visually
6. Click **"💾 Save Plot"** → `regression_diagnostics.pdf`

### Example 2: Exploratory Data Analysis

1. Load dataset: `data/example.csv`
2. Select variable: `sepal_width`
3. Click **"Histogram"**
   - View distribution with KDE overlay
   - Check for normality/skewness
4. Click **"Box Plot"**
   - Identify outliers and quartiles
5. Click **"Correlation Heatmap"**
   - See relationships between all variables
   - No variable selection needed
6. Export each plot for your report

### Example 3: Command Console Usage

Type commands in the console input box:

```
> summarize
> regress petal_length sepal_length sepal_width
> diagnostic_plots
```

Commands are logged to session log file automatically.

## Keyboard Shortcuts

- **Enter** in command box: Execute command
- Variable list: **Click** to select, **Ctrl+Click** for multiple

## File Outputs

### Session Log
- Location: `logs/session_YYYYMMDD_HHMMSS.log`
- Contains: All commands and outputs
- View location: Click **"Open log"** in toolbar

### Stata .do Script Export
- Click **"Export .do"** in toolbar
- Saves reproducible command script
- Compatible with Stata syntax

### Plot Exports
- Choose format when saving
- Default 300 DPI for high quality
- Tight bounding box (minimal whitespace)

## Technical Details

### Data Loading
- Supports CSV files with headers
- Numeric columns automatically detected
- Missing values handled gracefully

### Regression Engine
- OLS (Ordinary Least Squares) estimation
- Robust standard errors available
- White's heteroscedasticity-consistent errors

### Visualization Engine
- **Backend**: Matplotlib + Seaborn
- **Integration**: Qt Agg backend for PySide6
- **Quality**: Publication-ready plots
- **Customization**: All Phase 6 features available

### Performance
- Data preview limited to 100 rows for speed
- Full dataset used for analysis
- Plots render in <1 second for typical datasets
- Large datasets (>10,000 rows) may take longer

## Troubleshooting

### "Failed to load CSV"
- Check file format (must be valid CSV)
- Ensure headers in first row
- Verify numeric data in analysis columns

### "Regression failed"
- Check for missing values (NaN)
- Ensure sufficient observations
- Verify variable names are correct

### "Failed to create plots"
- Ensure regression was run successfully
- Check that matplotlib/seaborn are installed
- Try changing plot style if rendering issues

### Plots not displaying
- Switch to "Plots" tab manually
- Check console for error messages
- Restart application if canvas frozen

## Advanced Features

### Session Reproducibility
All operations are logged with timestamps. You can:
1. Export session log for documentation
2. Export .do script for reproduction
3. Re-run commands from command console

### Plot Customization
While the UI provides default styling, you can:
- Change themes via **"🎨 Plot Style"**
- Export and edit SVG files externally
- Use command console for advanced options

### Multiple Regressions
Run different models in sequence:
1. Each regression updates diagnostic plots
2. Results accumulate in Results tab
3. All operations logged

## Dependencies

Required Python packages:
- `PySide6` - Qt GUI framework
- `matplotlib >= 3.10.0` - Plotting
- `seaborn >= 0.13.0` - Statistical visualization
- `pandas` - Data manipulation
- `numpy` - Numerical operations
- `mathemixx_core` - Rust backend (via maturin wheel)

## Tips for Best Results

### Regression Diagnostics
- Always check diagnostic plots after regression
- Look for patterns in residual plots (indicates problems)
- Check Q-Q plot for normality
- Investigate points with high Cook's distance

### Exploratory Analysis
- Start with correlation heatmap for overview
- Use histograms to check distributions
- Box plots help identify outliers
- Compare violin plots across groups

### Plot Quality
- Use PDF/SVG for publications (vector formats)
- Use PNG for presentations (specify high DPI)
- Export before changing plot to preserve
- Apply consistent styling for professional look

### Workflow Efficiency
- Use keyboard shortcuts where available
- Select multiple variables at once
- Save plots incrementally during analysis
- Review session log before closing

## Future Enhancements

Planned features:
- Time series plots (ACF, PACF)
- Interactive plots with zoom/pan
- Categorical variable support
- Model comparison tools
- Batch plot export

## Getting Help

- Check session log: `logs/session_*.log`
- Console shows error messages
- All operations logged with timestamps
- Export .do script to share workflow

---

**MatheMixX Desktop UI** - Statistical analysis made visual and interactive!

For more information, see:
- `PHASE6_COMPLETE.md` - Visualization implementation details
- `README.md` - Project overview
- `ROADMAP.md` - Development roadmap
