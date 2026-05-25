# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

**Install (editable dev mode):**
```bash
pip install -e .
pip install -e ".[dev]"   # adds flake8, pytest, coverage
```

**Run tests:**
```bash
pytest
pytest tests/test_SMISWAXSLoader_chunked.py   # single test file
pytest tests/test_SMISWAXSLoader_chunked.py::TestClassName::test_method  # single test
```

**Lint:**
```bash
# Fatal errors only (breaks CI):
flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
# Style warnings (non-blocking):
flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics
```

**Test data setup** (required for most tests):
```bash
wget https://github.com/usnistgov/PyHyperScattering/releases/download/0.0.0-example-data/cyrsoxs-example.zip && unzip cyrsoxs-example.zip && rm cyrsoxs-example.zip
wget https://github.com/usnistgov/PyHyperScattering/releases/download/0.0.0-example-data/Example.zip && unzip Example.zip && rm Example.zip
wget https://github.com/usnistgov/PyHyperScattering/releases/download/0.0.0-example-data/mask-test-pack.zip && unzip mask-test-pack.zip && rm mask-test-pack.zip
```

## Architecture

PyHyperScattering is a scientific Python library for loading, reducing, and analyzing hyperspectral x-ray/neutron scattering data. The top-level namespace exposes three sub-modules:

```python
from PyHyperScattering import load     # loaders
from PyHyperScattering import integrate # integrators
from PyHyperScattering import util
```

### Data model

All loaders produce `xr.DataArray` objects with dims `['pix_x', 'pix_y']` plus instrument-specific metadata in `.attrs`. Multi-scan series are concatenated into a `'system'` dimension with a `pd.MultiIndex`. After integration, arrays gain `'q'` and `'chi'` dimensions.

Downstream analysis (fitting, slicing) is exposed as xarray accessors:
- `da.rsoxs.*` — chi slicing, anisotropy ratio, etc. (`RSoXS.py`)
- `da.fit.apply(fit_func)` — curve fitting via scipy (`Fitting.py`)

### Loaders (`src/PyHyperScattering/load.py`)

Each loader subclasses `FileLoader` (abstract base in `FileLoader.py`), which provides `loadFileSeries()`. Subclasses implement `loadSingleImage()`.

| Class | Instrument | Data source |
|---|---|---|
| `SST1RSoXSLoader` | NSLS-II SST-1 RSoXS | TIFF files |
| `SST1RSoXSDB` | NSLS-II SST-1 RSoXS | Bluesky/Tiled (live/archive) |
| `TiledSMISWAXSLoader` | NSLS-II SMI WAXS/SAXS | Tiled (`https://tiled.nsls2.bnl.gov`) |
| `SMIRSoXSLoader` | NSLS-II SMI RSoXS | Files |
| `ALS11012RSoXSLoader` | ALS 11.0.1.2 RSoXS | Files |
| `CMSGIWAXSLoader` | NSLS-II CMS GIWAXS | Files |
| `cyrsoxsLoader` | CyRSoXS simulation | Files |

`TiledSMISWAXSLoader` (in `SMISWAXSLoader.py`) is the most actively developed loader. It handles SMI's two detectors (Pilatus 2M SAXS; 900KW 3-panel arc WAXS) separately, caches per-run baseline data in module-level dicts, and implements a chunked-read fallback for HTTP 500 errors from Tiled. SMI-specific calibration constants and mask paths live in `smi_defaults.py`; bundled mask polygon JSON files are in `src/PyHyperScattering/data/smi/masks/`.

### Integrators (`src/PyHyperScattering/integrate.py`)

| Class | Backend | Use case |
|---|---|---|
| `PFGeneralIntegrator` | pyFAI | General azimuthal integration; main workhorse |
| `PFEnergySeriesIntegrator` | pyFAI | Energy series with per-energy calibration |
| `WPIntegrator` | skimage / CuPy (GPU) | qx/qy warp-polar integration |
| `PGGeneralIntegrator` | pygix | Grazing-incidence geometry |
| `NRSSIntegrator` | Custom | Neutron RSoXS |
| `SMISWAXSIntegrator` | pyFAI | SMI WAXS-specific wrapper |

`PFGeneralIntegrator` accepts masks via `maskmethod` (options: `nika`, `polygon`, `image`, `pyhyper`, `edf`, `numpy`, `none`) and geometry via `geomethod` (options: `template_xr`, `ponifile`, `NI`, `none`).

### Optional dependency groups

- `[bluesky]` — Tiled client + bluesky-tiled-plugins (needed for `SST1RSoXSDB`, `TiledSMISWAXSLoader`)
- `[grazing]` — silx + pygix (needed for `PGGeneralIntegrator`)
- `[performance]` — pyopencl, dask, cupy (GPU/parallel acceleration)
- `[ui]` — holoviews, hvplot, matplotlib (interactive plotting, `IntegrationUtils.py`)

### Versioning

Uses versioneer (PEP 440 from git tags). Dev builds show as `0.x+N.gHASH[.dirty]`.
