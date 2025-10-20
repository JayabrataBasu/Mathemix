"""Time Series Analysis widget for MatheMixX Desktop."""
from __future__ import annotations

import numpy as np
import pandas as pd
from PySide6.QtCore import Slot
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import plots
from timeseries import TimeSeriesAnalyzer


class TimeSeriesWidget(QWidget):
    """Widget for time series analysis and forecasting."""
    
    def __init__(self, plot_canvas, log_callback):
        super().__init__()
        self.plot_canvas = plot_canvas
        self.log_callback = log_callback
        self.dataset = None
        self.dataframe = None
        self.current_series = None
        self.ts_analyzer = None
        
        self._build_ui()
    
    def _build_ui(self):
        """Build the time series UI."""
        layout = QVBoxLayout(self)
        
        # Column selection
        select_group = QGroupBox("1. Select Time Series Column")
        select_layout = QVBoxLayout()
        
        self.column_list = QListWidget()
        self.column_list.itemClicked.connect(self._on_column_selected)
        select_layout.addWidget(QLabel("Available numeric columns:"))
        select_layout.addWidget(self.column_list)
        
        self.series_info = QLabel("No series selected")
        select_layout.addWidget(self.series_info)
        
        select_group.setLayout(select_layout)
        layout.addWidget(select_group)
        
        # Operations section
        ops_group = QGroupBox("2. Time Series Operations")
        ops_layout = QVBoxLayout()
        
        # Basic transformations
        transform_layout = QHBoxLayout()
        
        self.lag_spin = QSpinBox()
        self.lag_spin.setMinimum(1)
        self.lag_spin.setMaximum(50)
        self.lag_spin.setValue(1)
        self.lag_button = QPushButton("Lag")
        self.lag_button.clicked.connect(lambda: self._apply_operation("lag"))
        transform_layout.addWidget(QLabel("Periods:"))
        transform_layout.addWidget(self.lag_spin)
        transform_layout.addWidget(self.lag_button)
        
        self.diff_spin = QSpinBox()
        self.diff_spin.setMinimum(1)
        self.diff_spin.setMaximum(5)
        self.diff_spin.setValue(1)
        self.diff_button = QPushButton("Difference")
        self.diff_button.clicked.connect(lambda: self._apply_operation("diff"))
        transform_layout.addWidget(QLabel("Order:"))
        transform_layout.addWidget(self.diff_spin)
        transform_layout.addWidget(self.diff_button)
        
        ops_layout.addLayout(transform_layout)
        
        # Moving averages
        ma_layout = QHBoxLayout()
        
        self.ma_window = QSpinBox()
        self.ma_window.setMinimum(2)
        self.ma_window.setMaximum(100)
        self.ma_window.setValue(7)
        
        self.sma_button = QPushButton("SMA")
        self.sma_button.clicked.connect(lambda: self._apply_operation("sma"))
        self.ema_button = QPushButton("EMA")
        self.ema_button.clicked.connect(lambda: self._apply_operation("ema"))
        self.wma_button = QPushButton("WMA")
        self.wma_button.clicked.connect(lambda: self._apply_operation("wma"))
        
        ma_layout.addWidget(QLabel("Window:"))
        ma_layout.addWidget(self.ma_window)
        ma_layout.addWidget(self.sma_button)
        ma_layout.addWidget(self.ema_button)
        ma_layout.addWidget(self.wma_button)
        
        ops_layout.addLayout(ma_layout)
        
        ops_group.setLayout(ops_layout)
        layout.addWidget(ops_group)
        
        # ACF/PACF section
        acf_group = QGroupBox("3. Autocorrelation Analysis")
        acf_layout = QVBoxLayout()
        
        acf_control_layout = QHBoxLayout()
        self.nlags_spin = QSpinBox()
        self.nlags_spin.setMinimum(1)
        self.nlags_spin.setMaximum(100)
        self.nlags_spin.setValue(20)
        
        self.acf_button = QPushButton("Plot ACF")
        self.acf_button.clicked.connect(lambda: self._plot_acf_pacf("acf"))
        self.pacf_button = QPushButton("Plot PACF")
        self.pacf_button.clicked.connect(lambda: self._plot_acf_pacf("pacf"))
        self.acf_pacf_button = QPushButton("Plot ACF + PACF")
        self.acf_pacf_button.clicked.connect(lambda: self._plot_acf_pacf("both"))
        
        acf_control_layout.addWidget(QLabel("Lags:"))
        acf_control_layout.addWidget(self.nlags_spin)
        acf_control_layout.addWidget(self.acf_button)
        acf_control_layout.addWidget(self.pacf_button)
        acf_control_layout.addWidget(self.acf_pacf_button)
        
        acf_layout.addLayout(acf_control_layout)
        
        # Ljung-Box test
        lb_layout = QHBoxLayout()
        self.ljung_box_button = QPushButton("Ljung-Box Test")
        self.ljung_box_button.clicked.connect(self._ljung_box_test)
        lb_layout.addWidget(self.ljung_box_button)
        lb_layout.addStretch()
        acf_layout.addLayout(lb_layout)
        
        acf_group.setLayout(acf_layout)
        layout.addWidget(acf_group)
        
        # Stationarity tests
        stat_group = QGroupBox("4. Stationarity Tests")
        stat_layout = QHBoxLayout()
        
        self.adf_button = QPushButton("ADF Test")
        self.adf_button.clicked.connect(self._adf_test)
        self.kpss_button = QPushButton("KPSS Test")
        self.kpss_button.clicked.connect(self._kpss_test)
        self.both_tests_button = QPushButton("Run Both Tests")
        self.both_tests_button.clicked.connect(self._both_tests)
        
        stat_layout.addWidget(self.adf_button)
        stat_layout.addWidget(self.kpss_button)
        stat_layout.addWidget(self.both_tests_button)
        stat_layout.addStretch()
        
        stat_group.setLayout(stat_layout)
        layout.addWidget(stat_group)
        
        # Decomposition section
        decomp_group = QGroupBox("5. Seasonal Decomposition")
        decomp_layout = QFormLayout()
        
        self.period_spin = QSpinBox()
        self.period_spin.setMinimum(2)
        self.period_spin.setMaximum(365)
        self.period_spin.setValue(12)
        
        self.decomp_model = QComboBox()
        self.decomp_model.addItems(["additive", "multiplicative"])
        
        self.decompose_button = QPushButton("Decompose Series")
        self.decompose_button.clicked.connect(self._decompose)
        
        decomp_layout.addRow("Period:", self.period_spin)
        decomp_layout.addRow("Model:", self.decomp_model)
        decomp_layout.addRow("", self.decompose_button)
        
        decomp_group.setLayout(decomp_layout)
        layout.addWidget(decomp_group)
        
        # Forecasting section
        forecast_group = QGroupBox("6. Forecasting")
        forecast_layout = QFormLayout()
        
        # Method selection
        self.forecast_method = QComboBox()
        self.forecast_method.addItems([
            "Simple Exp Smoothing",
            "Holt Linear",
            "Holt-Winters"
        ])
        self.forecast_method.currentIndexChanged.connect(self._on_forecast_method_changed)
        
        # Common parameters
        self.alpha_spin = QDoubleSpinBox()
        self.alpha_spin.setMinimum(0.01)
        self.alpha_spin.setMaximum(0.99)
        self.alpha_spin.setValue(0.3)
        self.alpha_spin.setSingleStep(0.05)
        
        self.beta_spin = QDoubleSpinBox()
        self.beta_spin.setMinimum(0.01)
        self.beta_spin.setMaximum(0.99)
        self.beta_spin.setValue(0.1)
        self.beta_spin.setSingleStep(0.05)
        
        self.gamma_spin = QDoubleSpinBox()
        self.gamma_spin.setMinimum(0.01)
        self.gamma_spin.setMaximum(0.99)
        self.gamma_spin.setValue(0.2)
        self.gamma_spin.setSingleStep(0.05)
        
        self.forecast_period = QSpinBox()
        self.forecast_period.setMinimum(2)
        self.forecast_period.setMaximum(365)
        self.forecast_period.setValue(12)
        
        self.horizon_spin = QSpinBox()
        self.horizon_spin.setMinimum(1)
        self.horizon_spin.setMaximum(100)
        self.horizon_spin.setValue(10)
        
        self.confidence_spin = QDoubleSpinBox()
        self.confidence_spin.setMinimum(0.50)
        self.confidence_spin.setMaximum(0.99)
        self.confidence_spin.setValue(0.95)
        self.confidence_spin.setSingleStep(0.05)
        
        self.forecast_button = QPushButton("Generate Forecast")
        self.forecast_button.clicked.connect(self._forecast)
        
        forecast_layout.addRow("Method:", self.forecast_method)
        forecast_layout.addRow("Alpha (α):", self.alpha_spin)
        self.beta_row = QLabel("Beta (β):")
        forecast_layout.addRow(self.beta_row, self.beta_spin)
        self.gamma_row = QLabel("Gamma (γ):")
        forecast_layout.addRow(self.gamma_row, self.gamma_spin)
        self.period_row = QLabel("Period:")
        forecast_layout.addRow(self.period_row, self.forecast_period)
        forecast_layout.addRow("Horizon:", self.horizon_spin)
        forecast_layout.addRow("Confidence:", self.confidence_spin)
        forecast_layout.addRow("", self.forecast_button)
        
        forecast_group.setLayout(forecast_layout)
        layout.addWidget(forecast_group)
        
        # Results display
        results_group = QGroupBox("Results")
        results_layout = QVBoxLayout()
        
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setMaximumHeight(150)
        results_layout.addWidget(self.results_text)
        
        results_group.setLayout(results_layout)
        layout.addWidget(results_group)
        
        layout.addStretch()
        
        # Initial state
        self._set_controls_enabled(False)
        self._on_forecast_method_changed(0)
    
    def set_dataset(self, dataset, dataframe):
        """Set the current dataset."""
        self.dataset = dataset
        self.dataframe = dataframe
        self._populate_columns()
    
    def _populate_columns(self):
        """Populate column list with numeric columns."""
        self.column_list.clear()
        if self.dataframe is not None:
            numeric_cols = self.dataframe.select_dtypes(include=[np.number]).columns
            self.column_list.addItems(numeric_cols.tolist())
    
    def _set_controls_enabled(self, enabled: bool):
        """Enable/disable all controls."""
        for widget in [
            self.lag_button, self.diff_button,
            self.sma_button, self.ema_button, self.wma_button,
            self.acf_button, self.pacf_button, self.acf_pacf_button,
            self.ljung_box_button, self.adf_button, self.kpss_button,
            self.both_tests_button, self.decompose_button, self.forecast_button
        ]:
            widget.setEnabled(enabled)
    
    @Slot()
    def _on_column_selected(self):
        """Handle column selection."""
        selected_items = self.column_list.selectedItems()
        if not selected_items:
            return
        
        col_name = selected_items[0].text()
        self.current_series = self.dataframe[col_name].dropna().tolist()
        self.ts_analyzer = TimeSeriesAnalyzer(self.current_series)
        
        self.series_info.setText(
            f"Series: {col_name} | Length: {len(self.current_series)} | "
            f"Mean: {np.mean(self.current_series):.2f} | Std: {np.std(self.current_series):.2f}"
        )
        
        self._set_controls_enabled(True)
        self.results_text.clear()
        self.log_callback(f"Selected time series: {col_name}")
    
    def _on_forecast_method_changed(self, index):
        """Handle forecast method change."""
        method = self.forecast_method.currentText()
        
        # Show/hide parameters based on method
        if method == "Simple Exp Smoothing":
            self.beta_row.setVisible(False)
            self.beta_spin.setVisible(False)
            self.gamma_row.setVisible(False)
            self.gamma_spin.setVisible(False)
            self.period_row.setVisible(False)
            self.forecast_period.setVisible(False)
        elif method == "Holt Linear":
            self.beta_row.setVisible(True)
            self.beta_spin.setVisible(True)
            self.gamma_row.setVisible(False)
            self.gamma_spin.setVisible(False)
            self.period_row.setVisible(False)
            self.forecast_period.setVisible(False)
        else:  # Holt-Winters
            self.beta_row.setVisible(True)
            self.beta_spin.setVisible(True)
            self.gamma_row.setVisible(True)
            self.gamma_spin.setVisible(True)
            self.period_row.setVisible(True)
            self.forecast_period.setVisible(True)
    
    @Slot()
    def _apply_operation(self, op_type):
        """Apply time series operation."""
        if not self.ts_analyzer:
            return
        
        try:
            if op_type == "lag":
                result = self.ts_analyzer.lag(self.lag_spin.value())
                self.results_text.setPlainText(
                    f"Lag {self.lag_spin.value()}: {len(result)} values\n"
                    f"Mean: {np.mean(result):.4f}"
                )
                self.log_callback(f"lag({self.lag_spin.value()})")
            
            elif op_type == "diff":
                result = self.ts_analyzer.diff(self.diff_spin.value())
                self.results_text.setPlainText(
                    f"Difference order {self.diff_spin.value()}: {len(result)} values\n"
                    f"Mean: {np.mean(result):.4f} | Std: {np.std(result):.4f}"
                )
                self.log_callback(f"diff({self.diff_spin.value()})")
            
            elif op_type in ["sma", "ema", "wma"]:
                window = self.ma_window.value()
                if op_type == "sma":
                    result = self.ts_analyzer.sma(window)
                elif op_type == "ema":
                    result = self.ts_analyzer.ema(window)
                else:
                    result = self.ts_analyzer.wma(window)
                
                self.results_text.setPlainText(
                    f"{op_type.upper()}({window}): {len(result)} values\n"
                    f"Mean: {np.mean(result):.4f} | Std: {np.std(result):.4f}"
                )
                self.log_callback(f"{op_type}({window})")
        
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Operation failed: {e}")
    
    @Slot()
    def _plot_acf_pacf(self, plot_type):
        """Plot ACF and/or PACF."""
        if not self.ts_analyzer:
            return
        
        try:
            nlags = self.nlags_spin.value()
            
            if plot_type == "acf":
                acf_vals = self.ts_analyzer.acf(nlags)
                self.plot_canvas.clear()
                plots.plot_acf(acf_vals, ax=self.plot_canvas.ax)
                self.plot_canvas.draw()
                self.log_callback(f"plot_acf(nlags={nlags})")
            
            elif plot_type == "pacf":
                pacf_vals = self.ts_analyzer.pacf(nlags)
                self.plot_canvas.clear()
                plots.plot_pacf(pacf_vals, ax=self.plot_canvas.ax)
                self.plot_canvas.draw()
                self.log_callback(f"plot_pacf(nlags={nlags})")
            
            else:  # both
                self.plot_canvas.clear_figure()
                fig = plots.plot_acf_pacf(self.current_series, nlags=nlags)
                # Copy figure to canvas
                self.plot_canvas.fig = fig
                self.plot_canvas.draw()
                self.log_callback(f"plot_acf_pacf(nlags={nlags})")
        
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Plot failed: {e}")
    
    @Slot()
    def _ljung_box_test(self):
        """Run Ljung-Box test."""
        if not self.ts_analyzer:
            return
        
        try:
            nlags = self.nlags_spin.value()
            result = self.ts_analyzer.ljung_box_test(nlags)
            
            self.results_text.setPlainText(
                f"Ljung-Box Test (lags={nlags})\n"
                f"{'='*40}\n"
                f"Statistic: {result['statistic']:.4f}\n"
                f"P-value: {result['p_value']:.4f}\n"
                f"Degrees of Freedom: {result['degrees_of_freedom']}\n\n"
                f"Interpretation: {'No significant autocorrelation' if result['p_value'] > 0.05 else 'Significant autocorrelation detected'}"
            )
            self.log_callback(f"ljung_box_test(nlags={nlags})")
        
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Test failed: {e}")
    
    @Slot()
    def _adf_test(self):
        """Run ADF test."""
        if not self.ts_analyzer:
            return
        
        try:
            result = self.ts_analyzer.adf_test(max_lags=10)
            
            self.results_text.setPlainText(
                f"Augmented Dickey-Fuller Test\n"
                f"{'='*40}\n"
                f"Test Statistic: {result.test_statistic:.4f}\n"
                f"P-value: {result.p_value:.4f}\n"
                f"Lags Used: {result.lags_used}\n"
                f"Observations: {result.n_obs}\n"
                f"Critical Values:\n"
                f"  1%: {result.critical_values[0]:.4f}\n"
                f"  5%: {result.critical_values[1]:.4f}\n"
                f"  10%: {result.critical_values[2]:.4f}\n\n"
                f"Result: {'STATIONARY' if result.is_stationary else 'NON-STATIONARY'} (α=0.05)"
            )
            self.log_callback("adf_test()")
        
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Test failed: {e}")
    
    @Slot()
    def _kpss_test(self):
        """Run KPSS test."""
        if not self.ts_analyzer:
            return
        
        try:
            result = self.ts_analyzer.kpss_test(nlags=10)
            
            self.results_text.setPlainText(
                f"KPSS Test\n"
                f"{'='*40}\n"
                f"Test Statistic: {result.test_statistic:.4f}\n"
                f"P-value: {result.p_value:.4f}\n"
                f"Lags Used: {result.lags_used}\n"
                f"Observations: {result.n_obs}\n"
                f"Critical Values:\n"
                f"  1%: {result.critical_values[0]:.4f}\n"
                f"  5%: {result.critical_values[1]:.4f}\n"
                f"  10%: {result.critical_values[2]:.4f}\n\n"
                f"Result: {'STATIONARY' if result.is_stationary else 'NON-STATIONARY'} (α=0.05)"
            )
            self.log_callback("kpss_test()")
        
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Test failed: {e}")
    
    @Slot()
    def _both_tests(self):
        """Run both ADF and KPSS tests."""
        if not self.ts_analyzer:
            return
        
        try:
            adf_result = self.ts_analyzer.adf_test(max_lags=10)
            kpss_result = self.ts_analyzer.kpss_test(nlags=10)
            
            # Determine conclusion
            if adf_result.is_stationary and kpss_result.is_stationary:
                conclusion = "STATIONARY (both tests agree)"
            elif not adf_result.is_stationary and not kpss_result.is_stationary:
                conclusion = "NON-STATIONARY (both tests agree)"
            elif adf_result.is_stationary and not kpss_result.is_stationary:
                conclusion = "TREND STATIONARY (consider detrending)"
            else:
                conclusion = "DIFFERENCE STATIONARY (consider differencing)"
            
            self.results_text.setPlainText(
                f"Stationarity Tests\n"
                f"{'='*50}\n\n"
                f"ADF Test: {'STATIONARY' if adf_result.is_stationary else 'NON-STATIONARY'}\n"
                f"  Statistic: {adf_result.test_statistic:.4f}, P-value: {adf_result.p_value:.4f}\n\n"
                f"KPSS Test: {'STATIONARY' if kpss_result.is_stationary else 'NON-STATIONARY'}\n"
                f"  Statistic: {kpss_result.test_statistic:.4f}, P-value: {kpss_result.p_value:.4f}\n\n"
                f"Conclusion: {conclusion}"
            )
            self.log_callback("adf_test() + kpss_test()")
        
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Tests failed: {e}")
    
    @Slot()
    def _decompose(self):
        """Perform seasonal decomposition."""
        if not self.ts_analyzer:
            return
        
        try:
            period = self.period_spin.value()
            model = self.decomp_model.currentText()
            
            result = self.ts_analyzer.decompose(period=period, model=model)
            
            # Plot decomposition
            self.plot_canvas.clear_figure()
            fig = plots.plot_decomposition(result)
            self.plot_canvas.fig = fig
            self.plot_canvas.draw()
            
            # Show stats
            self.results_text.setPlainText(
                f"Seasonal Decomposition ({model.upper()})\n"
                f"{'='*40}\n"
                f"Period: {period}\n"
                f"Observations: {len(result.observed)}\n\n"
                f"Trend: Mean={np.nanmean(result.trend):.4f}\n"
                f"Seasonal: Range=[{np.nanmin(result.seasonal):.4f}, {np.nanmax(result.seasonal):.4f}]\n"
                f"Residual: Std={np.nanstd(result.residual):.4f}"
            )
            self.log_callback(f"seasonal_decompose(period={period}, model={model})")
        
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Decomposition failed: {e}")
    
    @Slot()
    def _forecast(self):
        """Generate forecast."""
        if not self.ts_analyzer:
            return
        
        try:
            method = self.forecast_method.currentText()
            alpha = self.alpha_spin.value()
            beta = self.beta_spin.value()
            gamma = self.gamma_spin.value()
            period = self.forecast_period.value()
            horizon = self.horizon_spin.value()
            confidence = self.confidence_spin.value()
            
            if method == "Simple Exp Smoothing":
                result = self.ts_analyzer.forecast_ses(
                    alpha=alpha,
                    horizon=horizon,
                    confidence_level=confidence
                )
            elif method == "Holt Linear":
                result = self.ts_analyzer.forecast_holt(
                    alpha=alpha,
                    beta=beta,
                    horizon=horizon,
                    confidence_level=confidence
                )
            else:  # Holt-Winters
                result = self.ts_analyzer.forecast_holt_winters(
                    alpha=alpha,
                    beta=beta,
                    gamma=gamma,
                    period=period,
                    horizon=horizon,
                    confidence_level=confidence
                )
            
            # Plot forecast
            self.plot_canvas.clear()
            plots.plot_forecast(
                self.current_series,
                result,
                n_history=min(50, len(self.current_series)),
                ax=self.plot_canvas.ax
            )
            self.plot_canvas.draw()
            
            # Show forecast summary
            self.results_text.setPlainText(
                f"{method} Forecast\n"
                f"{'='*40}\n"
                f"Horizon: {horizon} periods\n"
                f"Confidence Level: {int(confidence*100)}%\n\n"
                f"Forecast Values:\n" +
                "\n".join([f"  t+{i+1}: {val:.4f} [{result.lower_bound[i]:.4f}, {result.upper_bound[i]:.4f}]" 
                          for i, val in enumerate(result.forecasts[:min(10, horizon)])])
            )
            self.log_callback(f"{method.lower().replace(' ', '_')}(horizon={horizon})")
        
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Forecasting failed: {e}")


__all__ = ["TimeSeriesWidget"]
