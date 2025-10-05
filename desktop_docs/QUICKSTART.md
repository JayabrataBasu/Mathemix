# MatheMixX Desktop - Quick Reference

## 🚀 Getting Started

```bash
# Launch desktop UI
python run_desktop.py

# Or from project root
cd Mathemix
python run_desktop.py
```

## 📂 Loading Data

**Toolbar** → **Open CSV** → Select file

Supported: CSV files with headers, numeric data

## 📊 Basic Analysis

### Summarize Statistics
1. Load data
2. Click **"Summarize"**
3. View in Results tab

### Run Regression
1. Set **Dependent Y** (top field)
2. Select **Independent X** variables (multi-select list)
3. Click **"Run Regression"**
4. View results + scatter plot

## 🎨 Visualization (Phase 6)

### Diagnostic Plots (After Regression)
Click **"📊 Diagnostic Plots"** → See 6-plot suite:
- Residual vs Fitted
- Q-Q Plot
- Scale-Location
- Residuals vs Leverage
- Histogram
- Summary Stats

### Exploratory Plots (Anytime)
1. Select a variable from list
2. Click plot button:
   - **Box Plot** - Quartiles + outliers
   - **Histogram** - Distribution + KDE
   - **Heatmap** - Correlation matrix (all variables)
   - **Violin** - Distribution shape

## 💾 Export Options

### Save Current Plot
**Toolbar** → **💾 Save Plot** → Choose format:
- PNG (300 DPI)
- PDF (vector)
- SVG (vector)
- JPEG

### Save Session
- **Export .do** - Stata script (reproducible)
- **Open log** - View session log path

## 🎨 Customization

**Toolbar** → **🎨 Plot Style** → Choose theme:
- whitegrid (default)
- darkgrid
- white
- dark
- ticks

## ⌨️ Command Console

Type at bottom:
```
> summarize
> regress petal_length sepal_length sepal_width
```

## 🔍 Tabs

- **Data Preview** - First 100 rows
- **Results** - Analysis output
- **Plots** - All visualizations

## 💡 Tips

✅ **DO:**
- Check diagnostic plots after every regression
- Export plots before changing visualization
- Use PDF/SVG for publications
- Review session log for reproducibility

❌ **DON'T:**
- Run regression without selecting variables
- Forget to switch to Plots tab
- Close app without exporting important results

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| Plots not showing | Switch to "Plots" tab |
| Regression fails | Check for missing values (NaN) |
| Can't save plot | Ensure a plot is displayed |
| Style change not visible | Create new plot to see effect |

## 📋 Keyboard Shortcuts

- **Enter** in console → Execute command
- **Ctrl+Click** in list → Multi-select variables

## 📍 File Locations

```
logs/session_YYYYMMDD_HHMMSS.log  # Session log
exported_plots/                    # Your saved plots
data/                              # Example datasets
```

## 🎯 Common Workflows

### Workflow 1: Quick Analysis
1. Open CSV
2. Summarize
3. Run regression
4. View diagnostics
5. Export plot

### Workflow 2: Exploratory
1. Open CSV
2. Heatmap (overview)
3. Histogram (distributions)
4. Box plots (outliers)
5. Export all plots

### Workflow 3: Model Building
1. Load data
2. Exploratory plots
3. Run regression
4. Check diagnostics
5. Refine model
6. Export results + plots

## 📚 Documentation

- `DESKTOP_UI_GUIDE.md` - Full user guide
- `PHASE6_COMPLETE.md` - Visualization details
- `README.md` - Project overview

---

**Version**: 1.0.0 (Phase 6 Integration)  
**Updated**: October 5, 2024
