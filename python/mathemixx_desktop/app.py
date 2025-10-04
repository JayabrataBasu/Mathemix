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
import polars as pl
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from PySide6.QtCore import QAbstractTableModel, QModelIndex, Qt, Slot
from PySide6.QtGui import QAction, QCloseEvent
from PySide6.QtWidgets import (
    QApplication,
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

from mathemixx_core import DataSet

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

    def clear(self) -> None:
        self.ax.clear()
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


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("MatheMixX Desktop")
        self.resize(1200, 800)
        self.dataset: DataSet | None = None
        self.dataset_path: Path | None = None
        self.dataframe: pd.DataFrame | None = None
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

    @Slot()
    def open_csv(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(self, "Open CSV", "", "CSV Files (*.csv)")
        if not file_path:
            return
        try:
            dataset = DataSet.from_csv(file_path)
            self.dataset = dataset
            self.dataset_path = Path(file_path)
            self.file_label.setText(self.dataset_path.name)
            self.dataframe = pd.read_csv(self.dataset_path)
            self.update_data_preview(self.dataframe)
            self.populate_variables(self.dataframe.columns.tolist())
            self.log_command(f'use "{file_path}"')
            self.command_view.appendPlainText(f"Loaded dataset: {file_path}")
        except Exception as exc:  # pylint: disable=broad-except
            logging.exception("Failed to load CSV", exc_info=exc)
            QMessageBox.critical(self, "Error", f"Failed to load CSV: {exc}")

    def update_data_preview(self, frame: pd.DataFrame) -> None:
        self.data_model.set_frame(frame.head(100))
        self.summarize_button.setEnabled(True)
        self.regress_button.setEnabled(True)
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
            self.plot_canvas.plot_scatter_with_fit(x, y, result.coefficients())

    @Slot()
    def handle_command(self) -> None:
        command = self.command_input.text().strip()
        if not command:
            return
        self.command_input.clear()
        self.command_view.appendPlainText(f". {command}")
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

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802
        self.plot_canvas.clear()
        super().closeEvent(event)


def launch() -> None:
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.show()
    app.exec()


__all__ = ["launch", "MainWindow"]
