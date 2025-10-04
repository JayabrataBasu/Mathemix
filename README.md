# MatheMixX

MatheMixX is a Windows-first statistical desktop environment that mirrors Stata 19
workflows while prioritising numerical accuracy and raw performance.

- **Core engine (Rust)** — columnar storage via Polars, dense linear algebra via
  `ndarray` + LAPACK/OpenBLAS, robust QR/SVD solvers, and high-precision
  inference utilities.
- **Python bindings (PyO3 + maturin)** — a stable ABI3 wheel exposing datasets,
  descriptive statistics, and OLS regression to Python.
- **Desktop UI (PySide6)** — native Windows application featuring a Stata-like
  workflow: data browser, variable chooser, interactive console, results, plots,
  session logging, and `.do` export.

The repository is organised as a Cargo workspace hosting both the Rust core and
PyO3 bindings, plus a Python package containing the PySide6 front-end.

```text
├── Cargo.toml            # Workspace definition (core + bindings)
├── core/                 # Rust statistical engine (Polars + ndarray)
├── bindings/             # PyO3 layer producing the mathemixx_core wheel
├── python/
│   ├── mathemixx_core/   # Python package shim that re-exports the extension
│   ├── mathemixx_desktop/# PySide6 desktop application
│   └── tests/            # Python test suite (parity vs statsmodels)
├── data/                 # Sample datasets used in tests and demos
└── docs/                 # Additional notes and design docs (WIP)
```

## Prerequisites (Windows)

1. **Rust toolchain** (1.79 or newer) with the `stable-x86_64-pc-windows-msvc`
   target. Install via [rustup](https://www.rust-lang.org/tools/install).
2. **Visual Studio Build Tools 2022** with the "Desktop development with C++"
   workload (provides MSVC, Windows SDK, and CMake).
3. **Python 3.11+** (64-bit) and `pip`. The repository already ships with a
   dedicated virtual environment under `.mathemix/`; activate it or create your
   own.
4. **OpenBLAS** (recommended) — install a prebuilt distribution and set:

   ```powershell
   setx OPENBLAS_DIR "C:\openblas"
   setx OPENBLAS_LIB_DIR "C:\openblas\lib"
   setx OPENBLAS_INCLUDE_DIR "C:\openblas\include"
   ```

   MatheMixX defaults to OpenBLAS. To use Intel MKL instead, set the environment
   variable `MATHEMIXX_USE_MKL=1` and ensure MKL libraries are on your PATH; the
   `ndarray-linalg` feature set can then be switched accordingly in
   `core/Cargo.toml`.
5. **Python tooling**: `pip install maturin==1.6.* pyinstaller`. The desktop UI
   depends on `PySide6`, `matplotlib`, `pandas`, `polars`, `statsmodels`, and
   `seaborn`; these are declared in `pyproject.toml`.

## Building the Rust core + Python wheel

```powershell
cd C:\Users\jayab\Mathemix
python -m venv .venv         # optional (use existing environment if preferred)
.venv\Scripts\activate
pip install -r requirements-dev.txt  # TODO: consolidate developer deps
pip install maturin==1.6.*
maturin develop --release
```

`maturin develop` compiles the Rust core, links against OpenBLAS, and installs
the resulting `mathemixx_core` extension into the active environment.

## Running tests

```powershell
# Rust unit tests
cargo test -p mathemixx-core

# Python parity checks (requires the extension to be built)
pytest python/tests
```

The Python tests compare MatheMixX OLS output to statsmodels within tight
numerical tolerances (1e-9) and exercise CSV/TeX export paths.

## Launching the desktop application

```powershell
python -c "from mathemixx_desktop import launch; launch()"
```

Features available in the initial MVP:

- CSV import with Polars-backed in-memory storage.
- Variable selection (dependent + multiple independent variables).
- Descriptive statistics (`summarize`) for numeric columns.
- OLS regression with QR + SVD fallback, full inference metrics, and robust
	(HC0–HC3) standard errors.
- Regression tables exportable to CSV/TeX, JSON summary for downstream use.
- Embedded Matplotlib plotting (histogram, scatter with fitted line).
- Command console accepting basic Stata-like commands (`summarize`, `regress`).
- Session logging to timestamped `.log` files and action replay via generated
	`.do` scripts.

## Packaging into a Windows executable

After building the wheel (`maturin build --release --strip`), install the wheel
into a clean environment and run:

```powershell
pyinstaller --noconfirm --name MatheMixX \
	--add-data "data;data" \
	--hidden-import mathemixx_core \
	python\mathemixx_desktop\cli.py
```

Create `cli.py` that imports `mathemixx_desktop.launch()` (see docs/packaging
for a template). Ensure the resulting `dist/MatheMixX/` folder includes
OpenBLAS DLLs. Compress the directory or wrap with an installer of your choice
(e.g., Inno Setup) for final distribution.

## Roadmap

- Generalised linear models (logistic, Poisson) via IRLS with convergence
	diagnostics.
- High-dimensional fixed-effects (HDFE) absorption using within-transformation
	and sparse methods.
- Command interpreter parity with Stata `.do` scripts (macros, loops).
- Scenario benchmarks over large panel datasets.
- Comprehensive developer documentation in `docs/`.

Contributions are welcome—see `CONTRIBUTING.md` (TBD) for coding guidelines and
style conventions.
