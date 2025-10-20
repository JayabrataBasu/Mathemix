"""
Phase 6: Data Visualization Module

Provides high-level plotting functions using matplotlib and seaborn.
Uses data structures from the Rust core mathemixx_core module.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Optional, List, Tuple
import mathemixx_core as mx


# ============================================================================
# Utility Functions
# ============================================================================

def set_plot_style(style: str = 'whitegrid', font_scale: float = 1.0):
    """
    Set the plotting style for all subsequent plots.
    
    Args:
        style: Seaborn style name ('whitegrid', 'darkgrid', 'white', 'dark', 'ticks')
        font_scale: Font scaling factor (default=1.0)
    """
    sns.set_style(style)
    sns.set_context("notebook", font_scale=font_scale)


def save_plot(path: str, fig: Optional[plt.Figure] = None, dpi: int = 300, bbox_inches: str = 'tight'):
    """
    Save a plot to file.
    
    Args:
        path: Output file path (supports .png, .pdf, .svg, .jpg)
        fig: Matplotlib figure object (if None, uses current figure)
        dpi: Dots per inch for raster formats
        bbox_inches: Bounding box ('tight' removes whitespace)
    """
    if fig is None:
        fig = plt.gcf()  # Get current figure
    
    fig.savefig(path, dpi=dpi, bbox_inches=bbox_inches)
    print(f"Plot saved to: {path}")


# ============================================================================
# Regression Diagnostic Plots
# ============================================================================

def plot_residual_fitted(ols_result: mx.OlsResult, ax: Optional[plt.Axes] = None, **kwargs) -> plt.Axes:
    """
    Create a residual vs fitted values plot.
    
    This plot helps detect:
    - Non-linearity (patterns in residuals)
    - Heteroscedasticity (changing variance)
    - Outliers
    
    Args:
        ols_result: OLS regression result object
        ax: Optional matplotlib axes object
        **kwargs: Additional arguments passed to scatter()
    
    Returns:
        Matplotlib axes object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    data = ols_result.residual_fitted_data()
    
    # Scatter plot
    ax.scatter(data.fitted_values, data.residuals, alpha=0.6, **kwargs)
    ax.axhline(y=0, color='red', linestyle='--', linewidth=2, label='Zero residuals')
    ax.set_xlabel('Fitted Values', fontsize=12)
    ax.set_ylabel('Residuals', fontsize=12)
    ax.set_title('Residual vs Fitted Plot', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return ax


def plot_qq(ols_result: mx.OlsResult, ax: Optional[plt.Axes] = None, **kwargs) -> plt.Axes:
    """
    Create a Q-Q (quantile-quantile) plot for normality assessment.
    
    Points should fall on the diagonal line if residuals are normally distributed.
    
    Args:
        ols_result: OLS regression result object
        ax: Optional matplotlib axes object
        **kwargs: Additional arguments passed to scatter()
    
    Returns:
        Matplotlib axes object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    data = ols_result.qq_plot_data()
    
    # Q-Q plot
    ax.scatter(data.theoretical_quantiles, data.sample_quantiles, alpha=0.6, **kwargs)
    
    # Add diagonal reference line
    min_val = min(min(data.theoretical_quantiles), min(data.sample_quantiles))
    max_val = max(max(data.theoretical_quantiles), max(data.sample_quantiles))
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Normal line')
    
    ax.set_xlabel('Theoretical Quantiles', fontsize=12)
    ax.set_ylabel('Sample Quantiles', fontsize=12)
    ax.set_title('Normal Q-Q Plot', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return ax


def plot_scale_location(ols_result: mx.OlsResult, ax: Optional[plt.Axes] = None, **kwargs) -> plt.Axes:
    """
    Create a scale-location plot (sqrt of standardized residuals vs fitted values).
    
    This plot helps detect heteroscedasticity. Points should be randomly distributed
    around a horizontal line.
    
    Args:
        ols_result: OLS regression result object
        ax: Optional matplotlib axes object
        **kwargs: Additional arguments passed to scatter()
    
    Returns:
        Matplotlib axes object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    data = ols_result.scale_location_data()
    
    # Scatter plot
    ax.scatter(data.fitted_values, data.sqrt_abs_residuals, alpha=0.6, **kwargs)
    
    # Add trend line
    try:
        from scipy.stats import loess
        # Simple moving average as trend line
        window = max(3, len(data.fitted_values) // 10)
        trend = np.convolve(data.sqrt_abs_residuals, np.ones(window)/window, mode='same')
        sorted_indices = np.argsort(data.fitted_values)
        ax.plot(np.array(data.fitted_values)[sorted_indices], trend[sorted_indices], 
               'r-', linewidth=2, label='Trend')
    except ImportError:
        pass
    
    ax.set_xlabel('Fitted Values', fontsize=12)
    ax.set_ylabel('√|Standardized Residuals|', fontsize=12)
    ax.set_title('Scale-Location Plot', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    return ax


def plot_residuals_leverage(ols_result: mx.OlsResult, ax: Optional[plt.Axes] = None, **kwargs) -> plt.Axes:
    """
    Create a residuals vs leverage plot with Cook's distance.
    
    Helps identify influential observations. Points with high Cook's distance
    (typically > 0.5 or > 1.0) have strong influence on the regression.
    
    Args:
        ols_result: OLS regression result object
        ax: Optional matplotlib axes object
        **kwargs: Additional arguments passed to scatter()
    
    Returns:
        Matplotlib axes object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    data = ols_result.residuals_leverage_data()
    
    # Color by Cook's distance
    cooks_d = np.array(data.cooks_distance)
    scatter = ax.scatter(data.leverage, data.standardized_residuals, 
                        c=cooks_d, cmap='YlOrRd', alpha=0.6, **kwargs)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Cook's Distance", fontsize=10)
    
    # Add reference lines
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    
    ax.set_xlabel('Leverage', fontsize=12)
    ax.set_ylabel('Standardized Residuals', fontsize=12)
    ax.set_title("Residuals vs Leverage (Cook's Distance)", fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    return ax


def plot_residual_histogram(ols_result: mx.OlsResult, bins: Optional[int] = None,
                            ax: Optional[plt.Axes] = None, **kwargs) -> plt.Axes:
    """
    Create a histogram of residuals with normal curve overlay.
    
    Helps assess normality assumption. Residuals should follow a roughly
    normal (bell-shaped) distribution.
    
    Args:
        ols_result: OLS regression result object
        bins: Number of bins for histogram (None = auto)
        ax: Optional matplotlib axes object
        **kwargs: Additional arguments passed to hist()
    
    Returns:
        Matplotlib axes object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    data = ols_result.residual_histogram_data(bins=bins)
    
    # Histogram
    ax.hist(data.residuals, bins=data.bins, density=True, alpha=0.7, 
           edgecolor='black', **kwargs)
    
    # Overlay normal curve
    mu, std = np.mean(data.residuals), np.std(data.residuals)
    x = np.linspace(min(data.residuals), max(data.residuals), 100)
    normal_curve = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / std) ** 2)
    ax.plot(x, normal_curve, 'r-', linewidth=2, label='Normal curve')
    
    ax.set_xlabel('Residuals', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Residual Histogram', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    return ax


def plot_diagnostic_suite(ols_result: mx.OlsResult, figsize: Tuple[int, int] = (18, 12)) -> plt.Figure:
    """
    Create a comprehensive diagnostic plot suite (2x3 grid).
    
    Includes:
    1. Residual vs Fitted
    2. Q-Q Plot
    3. Scale-Location
    4. Residuals vs Leverage
    5. Residual Histogram
    6. Summary Statistics
    
    Args:
        ols_result: OLS regression result object
        figsize: Figure size (width, height)
    
    Returns:
        Matplotlib figure object
    """
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    axes = axes.flatten()
    
    # Create individual plots
    plot_residual_fitted(ols_result, ax=axes[0])
    plot_qq(ols_result, ax=axes[1])
    plot_scale_location(ols_result, ax=axes[2])
    plot_residuals_leverage(ols_result, ax=axes[3])
    plot_residual_histogram(ols_result, ax=axes[4])
    
    # Use last subplot for summary statistics
    axes[5].axis('off')
    summary_text = f"""
    Regression Summary
    ─────────────────────
    Dependent: {ols_result.dependent}
    R²: {ols_result.r_squared():.4f}
    Adj. R²: {ols_result.adj_r_squared():.4f}
    Observations: {ols_result.nobs()}
    """
    axes[5].text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
                   verticalalignment='center')
    
    plt.tight_layout()
    return fig


# ============================================================================
# General Statistical Plots
# ============================================================================

def plot_boxplot(dataset: mx.DataSet, column: str, ax: Optional[plt.Axes] = None, **kwargs) -> plt.Axes:
    """
    Create a box plot for a single variable.
    
    Shows distribution via quartiles, median, and outliers.
    
    Args:
        dataset: DataSet object
        column: Name of the column to plot
        ax: Optional matplotlib axes object
        **kwargs: Additional arguments passed to bxp()
    
    Returns:
        Matplotlib axes object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    data = dataset.box_plot_data(column)
    
    # Create box plot
    box_data = {
        'whislo': data.min,
        'q1': data.q1,
        'med': data.median,
        'q3': data.q3,
        'whishi': data.max,
        'mean': data.mean,
        'fliers': data.outliers
    }
    
    bp = ax.bxp([box_data], showmeans=True, meanline=True, **kwargs)
    ax.set_xticklabels([data.variable])
    ax.set_ylabel('Value', fontsize=12)
    ax.set_title(f'Box Plot: {data.variable}', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    return ax


def plot_histogram(dataset: mx.DataSet, column: str, bins: Optional[int] = None,
                  kde: bool = True, ax: Optional[plt.Axes] = None, **kwargs) -> plt.Axes:
    """
    Create a histogram with optional kernel density estimate.
    
    Args:
        dataset: DataSet object
        column: Name of the column to plot
        bins: Number of bins (None = auto)
        kde: Whether to overlay KDE curve
        ax: Optional matplotlib axes object
        **kwargs: Additional arguments passed to hist()
    
    Returns:
        Matplotlib axes object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    data = dataset.histogram_data(column, bins=bins)
    
    # Histogram
    ax.hist(data.values, bins=data.bins, density=kde, alpha=0.7,
           edgecolor='black', **kwargs)
    
    # Optional KDE overlay
    if kde:
        try:
            from scipy.stats import gaussian_kde
            kde_func = gaussian_kde(data.values)
            x_range = np.linspace(min(data.values), max(data.values), 200)
            ax.plot(x_range, kde_func(x_range), 'r-', linewidth=2, label='KDE')
            ax.legend()
        except ImportError:
            pass
    
    ax.set_xlabel(data.variable, fontsize=12)
    ax.set_ylabel('Density' if kde else 'Frequency', fontsize=12)
    ax.set_title(f'Histogram: {data.variable}', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    return ax


def plot_correlation_heatmap(dataset: mx.DataSet, columns: Optional[List[str]] = None,
                            method: str = 'pearson', ax: Optional[plt.Axes] = None,
                            **kwargs) -> plt.Axes:
    """
    Create a correlation heatmap.
    
    Args:
        dataset: DataSet object
        columns: List of columns to include (None = all numeric)
        method: Correlation method ('pearson' or 'spearman')
        ax: Optional matplotlib axes object
        **kwargs: Additional arguments passed to heatmap()
    
    Returns:
        Matplotlib axes object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    
    # Get correlation data
    data = dataset.heatmap_data(columns=columns, method=method)
    
    # Create heatmap
    sns.heatmap(data.correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
               xticklabels=data.variables, yticklabels=data.variables,
               vmin=-1, vmax=1, ax=ax, **kwargs)
    
    ax.set_title(f'{method.capitalize()} Correlation Heatmap', 
                fontsize=14, fontweight='bold')
    
    return ax


def plot_violin(dataset: mx.DataSet, column: str, ax: Optional[plt.Axes] = None, **kwargs) -> plt.Axes:
    """
    Create a violin plot showing distribution shape.
    
    Combines box plot and kernel density estimate.
    
    Args:
        dataset: DataSet object
        column: Name of the column to plot
        ax: Optional matplotlib axes object
        **kwargs: Additional arguments passed to violinplot()
    
    Returns:
        Matplotlib axes object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    data = dataset.violin_plot_data(column)
    
    # Create violin plot
    parts = ax.violinplot([data.values], positions=[0], showmeans=True, 
                          showmedians=True, **kwargs)
    
    ax.set_xticks([0])
    ax.set_xticklabels([data.variable])
    ax.set_ylabel('Value', fontsize=12)
    ax.set_title(f'Violin Plot: {data.variable}', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    return ax


# ============================================================================
# Phase 7: Time Series Plots
# ============================================================================

def plot_acf(
    acf_values: List[float],
    lags: Optional[List[int]] = None,
    ax: Optional[plt.Axes] = None,
    title: str = "Autocorrelation Function (ACF)",
    **kwargs
) -> plt.Axes:
    """
    Plot Autocorrelation Function (ACF).
    
    Args:
        acf_values: ACF values from timeseries.acf()
        lags: Lag values (if None, uses 0 to len(acf_values)-1)
        ax: Matplotlib axes (if None, creates new figure)
        title: Plot title
        **kwargs: Additional arguments for matplotlib
        
    Returns:
        Matplotlib axes object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
    
    if lags is None:
        lags = list(range(len(acf_values)))
    
    # Plot stems
    ax.stem(lags, acf_values, linefmt='C0-', markerfmt='C0o', basefmt='k-', **kwargs)
    
    # Add confidence interval bands (95%)
    n = len(acf_values)
    conf_int = 1.96 / np.sqrt(n)  # Approximate 95% CI
    ax.axhline(y=conf_int, color='r', linestyle='--', alpha=0.5, label='95% CI')
    ax.axhline(y=-conf_int, color='r', linestyle='--', alpha=0.5)
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    
    ax.set_xlabel('Lag', fontsize=12)
    ax.set_ylabel('ACF', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return ax


def plot_pacf(
    pacf_values: List[float],
    lags: Optional[List[int]] = None,
    ax: Optional[plt.Axes] = None,
    title: str = "Partial Autocorrelation Function (PACF)",
    **kwargs
) -> plt.Axes:
    """
    Plot Partial Autocorrelation Function (PACF).
    
    Args:
        pacf_values: PACF values from timeseries.pacf()
        lags: Lag values (if None, uses 0 to len(pacf_values)-1)
        ax: Matplotlib axes (if None, creates new figure)
        title: Plot title
        **kwargs: Additional arguments for matplotlib
        
    Returns:
        Matplotlib axes object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
    
    if lags is None:
        lags = list(range(len(pacf_values)))
    
    # Plot stems
    ax.stem(lags, pacf_values, linefmt='C1-', markerfmt='C1o', basefmt='k-', **kwargs)
    
    # Add confidence interval bands (95%)
    n = len(pacf_values)
    conf_int = 1.96 / np.sqrt(n)
    ax.axhline(y=conf_int, color='r', linestyle='--', alpha=0.5, label='95% CI')
    ax.axhline(y=-conf_int, color='r', linestyle='--', alpha=0.5)
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    
    ax.set_xlabel('Lag', fontsize=12)
    ax.set_ylabel('PACF', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return ax


def plot_decomposition(
    decomp_result,
    figsize: Tuple[int, int] = (12, 10),
    **kwargs
) -> plt.Figure:
    """
    Plot seasonal decomposition results in a 4-panel layout.
    
    Args:
        decomp_result: DecompositionResult from timeseries.seasonal_decompose()
        figsize: Figure size
        **kwargs: Additional arguments for matplotlib
        
    Returns:
        Matplotlib figure object
    """
    fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True)
    
    time = np.arange(len(decomp_result.observed))
    
    # Observed
    axes[0].plot(time, decomp_result.observed, label='Observed', color='C0', **kwargs)
    axes[0].set_ylabel('Observed', fontsize=11)
    axes[0].set_title('Time Series Decomposition', fontsize=14, fontweight='bold')
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)
    
    # Trend
    axes[1].plot(time, decomp_result.trend, label='Trend', color='C1', **kwargs)
    axes[1].set_ylabel('Trend', fontsize=11)
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)
    
    # Seasonal
    axes[2].plot(time, decomp_result.seasonal, label='Seasonal', color='C2', **kwargs)
    axes[2].set_ylabel('Seasonal', fontsize=11)
    axes[2].legend(loc='upper right')
    axes[2].grid(True, alpha=0.3)
    
    # Residual
    axes[3].plot(time, decomp_result.residual, label='Residual', color='C3', **kwargs)
    axes[3].set_ylabel('Residual', fontsize=11)
    axes[3].set_xlabel('Time', fontsize=12)
    axes[3].legend(loc='upper right')
    axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_forecast(
    observed: List[float],
    forecast_result,
    n_history: Optional[int] = None,
    figsize: Tuple[int, int] = (12, 6),
    **kwargs
) -> plt.Axes:
    """
    Plot time series forecast with confidence intervals.
    
    Args:
        observed: Historical observed data
        forecast_result: ForecastResult from forecasting methods
        n_history: Number of historical points to show (if None, shows all)
        figsize: Figure size
        **kwargs: Additional arguments for matplotlib
        
    Returns:
        Matplotlib axes object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Historical data
    if n_history is not None:
        observed = observed[-n_history:]
    
    n_obs = len(observed)
    n_forecast = len(forecast_result.forecasts)
    
    time_obs = np.arange(n_obs)
    time_forecast = np.arange(n_obs, n_obs + n_forecast)
    
    # Plot observed data
    ax.plot(time_obs, observed, label='Observed', color='C0', linewidth=2, **kwargs)
    
    # Plot forecast
    ax.plot(time_forecast, forecast_result.forecasts, 
            label='Forecast', color='C1', linewidth=2, linestyle='--')
    
    # Plot confidence interval
    ax.fill_between(
        time_forecast,
        forecast_result.lower_bound,
        forecast_result.upper_bound,
        alpha=0.3,
        color='C1',
        label=f'{int(forecast_result.confidence_level*100)}% CI'
    )
    
    # Add vertical line at forecast start
    ax.axvline(x=n_obs-0.5, color='k', linestyle=':', alpha=0.5)
    
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Value', fontsize=12)
    ax.set_title('Time Series Forecast', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    return ax


def plot_acf_pacf(
    data: List[float],
    nlags: int = 20,
    figsize: Tuple[int, int] = (12, 5),
    **kwargs
) -> plt.Figure:
    """
    Plot ACF and PACF side by side.
    
    Args:
        data: Time series data
        nlags: Number of lags to plot
        figsize: Figure size
        **kwargs: Additional arguments for matplotlib
        
    Returns:
        Matplotlib figure object
    """
    from python.timeseries import acf, pacf
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Calculate ACF and PACF
    acf_vals = acf(data, nlags)
    pacf_vals = pacf(data, nlags)
    
    # Plot ACF
    plot_acf(acf_vals, ax=axes[0], **kwargs)
    
    # Plot PACF
    plot_pacf(pacf_vals, ax=axes[1], **kwargs)
    
    plt.tight_layout()
    return fig


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    # This section demonstrates basic usage
    print("Mathemix Plotting Module")
    print("Import this module and use the plotting functions with mathemixx_core data structures.")

