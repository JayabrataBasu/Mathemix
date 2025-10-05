"""MatheMixX PySide6 desktop application."""
from __future__ import annotations

import datetime as _dt
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from PySide6.QtCore import QAbstractTableModel, QModelIndex, Qt, Slot
from PySide6.QtGui import QAction, QCloseEvent
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTabWidget,
    QTableView,
    QTextEdit,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

# Import directly from the installed Rust bindings wheel
import mathemixx_core as mx

# Import Phase 6 visualization module
import sys
from pathlib import Path as PathLib
sys.path.insert(0, str(PathLib(__file__).parent.parent))
import plots

matplotlib.use("QtAgg")


class PandasModel(QAbstractTableModel):
    def __init__(self, frame: pd.DataFrame):
        super().__init__()
        self._frame = frame

    def rowCount(self, parent: QModelIndex | None = None) -> int:
        return 0 if parent and parent.isValid() else len(self._frame.index)

    def columnCount(self, parent: QModelIndex | None = None) -> int:
        return 0 if parent and parent.isValid() else len(self._frame.columns)

    def data(self, index: QModelIndex, role: int = Qt.DisplayRole):
        if not index.isValid() or role not in (Qt.DisplayRole, Qt.EditRole):
            return None
        value = self._frame.iat[index.row(), index.column()]
        if isinstance(value, float):
            return f"{value:.6g}"
        return str(value)

    def headerData(self, section: int, orientation: Qt.Orientation, role: int = Qt.DisplayRole):
        if role != Qt.DisplayRole:
            return None
        if orientation == Qt.Horizontal:
            return str(self._frame.columns[section])
        return str(self._frame.index[section])

    def set_frame(self, frame: pd.DataFrame) -> None:
        self.beginResetModel()
        self._frame = frame
        self.endResetModel()


@dataclass
class SessionLog:
    path: Path
    commands: List[str] = field(default_factory=list)

    def record(self, command: str, output: str | None = None) -> None:
        timestamp = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        entry = f"[{timestamp}] {command}"
        self.commands.append(command)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(entry + "\n")
            if output:
                handle.write(output + "\n")

    def export_do(self, path: Path) -> None:
        with path.open("w", encoding="utf-8") as handle:
            handle.write("* MatheMixX .do script generated from session\n")
            for cmd in self.commands:
                handle.write(cmd + "\n")


class PlotCanvas(FigureCanvasQTAgg):
    def __init__(self) -> None:
        self.fig, self.ax = plt.subplots(figsize=(6, 4))
        super().__init__(self.fig)
        self.current_result: mx.OlsResult | None = None

    def clear(self) -> None:
        self.ax.clear()
        self.draw()

    def clear_figure(self) -> None:
        """Clear the entire figure (for multi-subplot plots)."""
        self.fig.clear()
        self.ax = self.fig.add_subplot(111)
        self.draw()

    def plot_histogram(self, data: pd.Series, title: str) -> None:
        self.ax.clear()
        self.ax.hist(data.dropna(), bins=20, color="#1f77b4", alpha=0.8)
        self.ax.set_title(title)
        self.ax.set_xlabel(data.name)
        self.ax.set_ylabel("Frequency")
        self.fig.tight_layout()
        self.draw()

    def plot_scatter_with_fit(self, x: pd.Series, y: pd.Series, coefs: Iterable[float]) -> None:
        self.ax.clear()
        self.ax.scatter(x, y, color="#ff7f0e", alpha=0.8)
        xs = np.linspace(x.min(), x.max(), 100)
        beta0 = coefs[0]
        beta1 = coefs[1] if len(coefs) > 1 else 0.0
        y_hat = beta0 + beta1 * xs
        self.ax.plot(xs, y_hat, color="#1f77b4")
        self.ax.set_title(f"{y.name} vs {x.name}")
        self.ax.set_xlabel(x.name)
        self.ax.set_ylabel(y.name)
        self.fig.tight_layout()
        self.draw()
    
    def plot_diagnostic_suite(self, result: mx.OlsResult) -> None:
        """Plot regression diagnostic suite using Phase 6 functionality."""
        self.current_result = result
        self.fig.clear()
        
        # Create 2x3 subplot grid
        axes = self.fig.subplots(2, 3)
        axes = axes.flatten()
        
        # Plot each diagnostic
        plots.plot_residual_fitted(result, ax=axes[0])
        plots.plot_qq(result, ax=axes[1])
        plots.plot_scale_location(result, ax=axes[2])
        plots.plot_residuals_leverage(result, ax=axes[3])
        plots.plot_residual_histogram(result, ax=axes[4])
        
        # Summary statistics in last panel
        axes[5].axis('off')
        summary_text = f"""
    Regression Summary
    ─────────────────────
    Dependent: {result.dependent}
    R²: {result.r_squared():.4f}
    Adj. R²: {result.adj_r_squared():.4f}
    Observations: {result.nobs()}
    """
        axes[5].text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
                    verticalalignment='center')
        
        self.fig.tight_layout()
        self.draw()
    
    def plot_individual_diagnostic(self, result: mx.OlsResult, plot_type: str) -> None:
        """Plot a single diagnostic plot."""
        self.current_result = result
        self.fig.clear()
        self.ax = self.fig.add_subplot(111)
        
        if plot_type == "residual_fitted":
            plots.plot_residual_fitted(result, ax=self.ax)
        elif plot_type == "qq":
            plots.plot_qq(result, ax=self.ax)
        elif plot_type == "scale_location":
            plots.plot_scale_location(result, ax=self.ax)
        elif plot_type == "residuals_leverage":
            plots.plot_residuals_leverage(result, ax=self.ax)
        elif plot_type == "histogram":
            plots.plot_residual_histogram(result, ax=self.ax)
        
        self.fig.tight_layout()
        self.draw()
    
    def plot_exploratory(self, dataset: mx.DataSet, plot_type: str, column: str = None) -> None:
        """Plot exploratory data visualization."""
        self.fig.clear()
        self.ax = self.fig.add_subplot(111)
        
        if plot_type == "boxplot" and column:
            plots.plot_boxplot(dataset, column, ax=self.ax)
        elif plot_type == "histogram" and column:
            plots.plot_histogram(dataset, column, kde=True, ax=self.ax)
        elif plot_type == "violin" and column:
            plots.plot_violin(dataset, column, ax=self.ax)
        elif plot_type == "heatmap":
            plots.plot_correlation_heatmap(dataset, method='pearson', ax=self.ax)
        
        self.fig.tight_layout()
        self.draw()


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("MatheMixX Desktop")
        self.resize(1200, 800)
        self.dataset: mx.DataSet | None = None
        self.dataset_path: Path | None = None
        self.dataframe: pd.DataFrame | None = None
        self.current_regression_result: mx.OlsResult | None = None
        self.log_path = Path("logs") / f"session_{_dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.session_log = SessionLog(self.log_path)

        self._build_ui()
        self._build_actions()

    def _build_ui(self) -> None:
        container = QWidget(self)
        layout = QHBoxLayout(container)

        splitter = QSplitter()
        layout.addWidget(splitter)

        control_panel = QWidget()
        control_layout = QVBoxLayout(control_panel)

        self.file_label = QLabel("No dataset loaded")
        control_layout.addWidget(self.file_label)

        control_layout.addWidget(QLabel("Dependent (Y)"))
        self.dependent_input = QLineEdit()
        self.dependent_input.setReadOnly(True)
        control_layout.addWidget(self.dependent_input)

        control_layout.addWidget(QLabel("Independent variables (X)"))
        self.independent_list = QListWidget()
        self.independent_list.setSelectionMode(QListWidget.MultiSelection)
        control_layout.addWidget(self.independent_list)

        self.summarize_button = QPushButton("Summarize")
        self.summarize_button.clicked.connect(self.handle_summarize)
        self.summarize_button.setEnabled(False)
        control_layout.addWidget(self.summarize_button)

        self.regress_button = QPushButton("Run Regression")
        self.regress_button.clicked.connect(self.handle_regression)
        self.regress_button.setEnabled(False)
        control_layout.addWidget(self.regress_button)

        self.diagnostic_button = QPushButton("📊 Diagnostic Plots")
        self.diagnostic_button.clicked.connect(self.show_diagnostic_plots)
        self.diagnostic_button.setEnabled(False)
        control_layout.addWidget(self.diagnostic_button)

        control_layout.addWidget(QLabel("─" * 30))
        control_layout.addWidget(QLabel("Exploratory Plots"))
        
        exploratory_layout = QVBoxLayout()
        
        self.boxplot_button = QPushButton("Box Plot")
        self.boxplot_button.clicked.connect(lambda: self.show_exploratory_plot("boxplot"))
        self.boxplot_button.setEnabled(False)
        exploratory_layout.addWidget(self.boxplot_button)
        
        self.histogram_button = QPushButton("Histogram + KDE")
        self.histogram_button.clicked.connect(lambda: self.show_exploratory_plot("histogram"))
        self.histogram_button.setEnabled(False)
        exploratory_layout.addWidget(self.histogram_button)
        
        self.heatmap_button = QPushButton("Correlation Heatmap")
        self.heatmap_button.clicked.connect(lambda: self.show_exploratory_plot("heatmap"))
        self.heatmap_button.setEnabled(False)
        exploratory_layout.addWidget(self.heatmap_button)
        
        self.violin_button = QPushButton("Violin Plot")
        self.violin_button.clicked.connect(lambda: self.show_exploratory_plot("violin"))
        self.violin_button.setEnabled(False)
        exploratory_layout.addWidget(self.violin_button)
        
        control_layout.addLayout(exploratory_layout)

        control_layout.addStretch(1)

        control_layout.addWidget(QLabel("Command Console"))
        self.command_view = QTextEdit()
        self.command_view.setReadOnly(True)
        control_layout.addWidget(self.command_view)

        input_layout = QHBoxLayout()
        self.command_input = QLineEdit()
        self.command_input.returnPressed.connect(self.handle_command)
        input_layout.addWidget(self.command_input)
        run_button = QPushButton("Run")
        run_button.clicked.connect(self.handle_command)
        input_layout.addWidget(run_button)
        control_layout.addLayout(input_layout)

        splitter.addWidget(control_panel)

        right_panel = QTabWidget()

        self.data_table = QTableView()
        self.data_model = PandasModel(pd.DataFrame())
        self.data_table.setModel(self.data_model)
        right_panel.addTab(self.data_table, "Data Preview")

        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        right_panel.addTab(self.results_text, "Results")

        self.plot_canvas = PlotCanvas()
        plot_container = QWidget()
        plot_layout = QVBoxLayout(plot_container)
        plot_layout.addWidget(self.plot_canvas)
        right_panel.addTab(plot_container, "Plots")

        splitter.addWidget(right_panel)
        splitter.setStretchFactor(1, 3)

        self.setCentralWidget(container)

    def _build_actions(self) -> None:
        toolbar = QToolBar("Main")
        self.addToolBar(toolbar)

        open_action = QAction("Open CSV", self)
        open_action.triggered.connect(self.open_csv)
        toolbar.addAction(open_action)

        export_do = QAction("Export .do", self)
        export_do.triggered.connect(self.export_do_script)
        toolbar.addAction(export_do)

        export_log = QAction("Open log", self)
        export_log.triggered.connect(self.show_log_location)
        toolbar.addAction(export_log)

        toolbar.addSeparator()

        export_plot = QAction("💾 Save Plot", self)
        export_plot.triggered.connect(self.export_current_plot)
        toolbar.addAction(export_plot)

        set_style = QAction("🎨 Plot Style", self)
        set_style.triggered.connect(self.change_plot_style)
        toolbar.addAction(set_style)

    @Slot()
    def open_csv(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(self, "Open CSV", "", "CSV Files (*.csv)")
        if not file_path:
            return
        try:
            dataset = mx.DataSet.from_csv(file_path)
            self.dataset = dataset
            self.dataset_path = Path(file_path)
            self.file_label.setText(self.dataset_path.name)
            self.dataframe = pd.read_csv(self.dataset_path)
            self.update_data_preview(self.dataframe)
            self.populate_variables(self.dataframe.columns.tolist())
            self.log_command(f'use "{file_path}"')
            self.command_view.append(f"Loaded dataset: {file_path}")
        except Exception as exc:  # pylint: disable=broad-except
            logging.exception("Failed to load CSV", exc_info=exc)
            QMessageBox.critical(self, "Error", f"Failed to load CSV: {exc}")

    def update_data_preview(self, frame: pd.DataFrame) -> None:
        self.data_model.set_frame(frame.head(100))
        self.summarize_button.setEnabled(True)
        self.regress_button.setEnabled(True)
        
        # Enable exploratory plot buttons
        self.boxplot_button.setEnabled(True)
        self.histogram_button.setEnabled(True)
        self.heatmap_button.setEnabled(True)
        self.violin_button.setEnabled(True)
        
        if frame.columns.size:
            self.dependent_input.setText(frame.columns[0])

    def populate_variables(self, columns: List[str]) -> None:
        self.independent_list.clear()
        for name in columns:
            item = QListWidgetItem(name)
            item.setSelected(False)
            self.independent_list.addItem(item)

    def selected_independents(self) -> List[str]:
        return [item.text() for item in self.independent_list.selectedItems()]

    @Slot()
    def handle_summarize(self) -> None:
        if not self.dataset:
            return
        rows = self.dataset.summarize()
        summary_df = pd.DataFrame(
            [
                {
                    "variable": row.variable,
                    "mean": row.mean,
                    "sd": row.sd,
                    "min": row.min,
                    "max": row.max,
                }
                for row in rows
            ]
        )
        self.results_text.setPlainText(summary_df.to_string(index=False))
        self.log_command("summarize", summary_df.to_string(index=False))

    @Slot()
    def handle_regression(self) -> None:
        if not self.dataset:
            return
        dependent = self.dependent_input.text().strip()
        if not dependent:
            QMessageBox.warning(self, "Regression", "Select a dependent variable")
            return
        independents = self.selected_independents()
        if dependent in independents:
            independents.remove(dependent)
        if not independents:
            QMessageBox.warning(self, "Regression", "Select at least one independent variable")
            return
        try:
            result = self.dataset.regress_ols(dependent, independents, True)
            self.current_regression_result = result
            self.diagnostic_button.setEnabled(True)
        except Exception as exc:  # pylint: disable=broad-except
            logging.exception("Regression failed", exc_info=exc)
            QMessageBox.critical(self, "Regression", f"Regression failed: {exc}")
            return

        coeff_rows = result.table()
        table = pd.DataFrame(
            [
                {
                    "variable": row.variable,
                    "coefficient": row.coefficient,
                    "std_error": row.std_error,
                    "t_value": row.t_value,
                    "p_value": row.p_value,
                    "ci_lower": row.ci_lower,
                    "ci_upper": row.ci_upper,
                }
                for row in coeff_rows
            ]
        )
        meta = {
            "R-squared": result.r_squared(),
            "Adj. R-squared": result.adj_r_squared(),
            "nobs": result.nobs(),
        }
        output = table.to_string(index=False) + "\n" + json.dumps(meta, indent=2)
        self.results_text.setPlainText(output)
        self.log_command(
            f"regress {dependent} {' '.join(independents)}",
            output,
        )
        if self.dataframe is not None and independents:
            x = self.dataframe[independents[0]]
            y = self.dataframe[dependent]
            self.plot_canvas.plot_scatter_with_fit(x, y, result.coefficients)

    @Slot()
    def handle_command(self) -> None:
        command = self.command_input.text().strip()
        if not command:
            return
        self.command_input.clear()
        self.command_view.append(f". {command}")
        if command.startswith("summarize"):
            self.handle_summarize()
        elif command.startswith("regress"):
            parts = command.split()
            if len(parts) >= 3:
                dependent = parts[1]
                independents = parts[2:]
                self.dependent_input.setText(dependent)
                for i in range(self.independent_list.count()):
                    item = self.independent_list.item(i)
                    item.setSelected(item.text() in independents)
                self.handle_regression()
            else:
                QMessageBox.warning(self, "Command", "Usage: regress y x1 x2 ...")
        else:
            QMessageBox.information(self, "Command", f"Unsupported command: {command}")
            self.log_command(command, "Unsupported command")

    def log_command(self, command: str, output: str | None = None) -> None:
        self.session_log.record(command, output)

    @Slot()
    def export_do_script(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Export .do", "session.do", "Stata Do Files (*.do)")
        if not path:
            return
        self.session_log.export_do(Path(path))
        QMessageBox.information(self, "Export", f"Saved .do script to {path}")

    @Slot()
    def show_log_location(self) -> None:
        QMessageBox.information(self, "Log", f"Session log: {self.log_path}")

    @Slot()
    def show_diagnostic_plots(self) -> None:
        """Show comprehensive regression diagnostic plots."""
        if not self.current_regression_result:
            QMessageBox.warning(self, "Diagnostics", "Run a regression first")
            return
        
        try:
            # Switch to Plots tab
            right_tabs = self.findChild(QTabWidget)
            if right_tabs:
                right_tabs.setCurrentIndex(2)  # Plots tab
            
            # Plot diagnostic suite
            self.plot_canvas.plot_diagnostic_suite(self.current_regression_result)
            self.log_command("diagnostic_plots", "Displayed regression diagnostic suite")
            
        except Exception as exc:  # pylint: disable=broad-except
            logging.exception("Failed to create diagnostic plots", exc_info=exc)
            QMessageBox.critical(self, "Error", f"Failed to create plots: {exc}")
    
    @Slot()
    def show_exploratory_plot(self, plot_type: str) -> None:
        """Show exploratory data visualization."""
        if not self.dataset:
            QMessageBox.warning(self, "Plots", "Load a dataset first")
            return
        
        # Get selected variable from list or dependent input
        selected = self.selected_independents()
        if selected:
            column = selected[0]
        else:
            column = self.dependent_input.text().strip()
        
        if not column and plot_type != "heatmap":
            QMessageBox.warning(self, "Plots", "Select a variable first")
            return
        
        try:
            # Switch to Plots tab
            right_tabs = self.findChild(QTabWidget)
            if right_tabs:
                right_tabs.setCurrentIndex(2)  # Plots tab
            
            # Create plot
            self.plot_canvas.plot_exploratory(self.dataset, plot_type, column)
            
            plot_names = {
                "boxplot": "Box Plot",
                "histogram": "Histogram with KDE",
                "heatmap": "Correlation Heatmap",
                "violin": "Violin Plot"
            }
            self.log_command(f"{plot_type} {column if column else ''}", 
                           f"Displayed {plot_names.get(plot_type, plot_type)}")
            
        except Exception as exc:  # pylint: disable=broad-except
            logging.exception(f"Failed to create {plot_type} plot", exc_info=exc)
            QMessageBox.critical(self, "Error", f"Failed to create plot: {exc}")
    
    @Slot()
    def export_current_plot(self) -> None:
        """Export the current plot to file."""
        file_path, selected_filter = QFileDialog.getSaveFileName(
            self, 
            "Save Plot", 
            "plot.png", 
            "PNG Image (*.png);;PDF Document (*.pdf);;SVG Vector (*.svg);;JPEG Image (*.jpg)"
        )
        if not file_path:
            return
        
        try:
            plots.save_plot(file_path, self.plot_canvas.fig, dpi=300)
            QMessageBox.information(self, "Export", f"Plot saved to {file_path}")
            self.log_command(f"export_plot {file_path}", f"Saved plot to {file_path}")
        except Exception as exc:  # pylint: disable=broad-except
            logging.exception("Failed to export plot", exc_info=exc)
            QMessageBox.critical(self, "Error", f"Failed to export plot: {exc}")
    
    @Slot()
    def change_plot_style(self) -> None:
        """Change the plotting style."""
        from PySide6.QtWidgets import QInputDialog
        
        styles = ['whitegrid', 'darkgrid', 'white', 'dark', 'ticks']
        style, ok = QInputDialog.getItem(
            self, 
            "Plot Style", 
            "Select a style:", 
            styles, 
            0, 
            False
        )
        
        if ok and style:
            try:
                plots.set_plot_style(style)
                QMessageBox.information(self, "Style", f"Plot style set to: {style}")
                self.log_command(f"set_plot_style {style}", f"Changed style to {style}")
            except Exception as exc:  # pylint: disable=broad-except
                logging.exception("Failed to set style", exc_info=exc)
                QMessageBox.critical(self, "Error", f"Failed to set style: {exc}")

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802
        self.plot_canvas.clear()
        super().closeEvent(event)


def launch() -> None:
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.show()
    app.exec()


__all__ = ["launch", "MainWindow"]
