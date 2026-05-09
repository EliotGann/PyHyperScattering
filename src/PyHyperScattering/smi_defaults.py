"""SMI defaults bundled with PyHyperScattering.

Provides on-disk locations and helpers for default masks and other small
calibration assets that previously lived in the smi-browser repository.

The masks are shipped as package data under ``PyHyperScattering/data/smi/``.
Downstream code (notebooks, browsers, UIs) should treat these as canonical
defaults: simply pass ``None`` (or omit the argument) to integrator entry
points, and the bundled mask will be used. Callers can still pass an
explicit absolute path to override.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any, Literal, Optional, Union

import numpy as np

__all__ = [
    "DEFAULT_TILED_URI",
    "DEFAULT_CATALOG",
    "DEFAULT_SAXS_MASK_NAME",
    "DEFAULT_WAXS_MASK_NAME",
    "default_saxs_mask_path",
    "default_waxs_mask_path",
    "resolve_mask_path",
    # Detector classification
    "SAXS_DETECTOR_NAMES",
    "WAXS_DETECTOR_NAMES",
    "DetectorKind",
    "classify_detector_field",
    # Beamline / loader calibration
    "BSX_PER_ARC_DEG",
    "LoaderCalibration",
    "LOADER_DEFAULTS",
    # Display orientation
    "orient_frame_for_display",
    "orient_polygon_xy",
    "orient_polygon_xy_inverse",
    # Mask I/O
    "load_mask_polygons",
    "save_mask_polygons",
    # SMI-specific Tiled defaults
    "SAXS_IMAGE_FIELD",
    "WAXS_IMAGE_FIELD",
    "WAXS_ARC_FIELD",
    "WAXS_BSX_FIELD",
    "DEFAULT_DETECTOR_FIELDS",
]

# ---------------------------------------------------------------------------
# Tiled connection defaults (also re-exported by SMISWAXSLoader)
# ---------------------------------------------------------------------------

DEFAULT_TILED_URI: str = "https://tiled.nsls2.bnl.gov"
DEFAULT_CATALOG: str = "smi/migration"

# ---------------------------------------------------------------------------
# Bundled mask filenames (kept stable for downstream lookup-by-name)
# ---------------------------------------------------------------------------

DEFAULT_SAXS_MASK_NAME: str = "pil2M_mask_polygons.json"
DEFAULT_WAXS_MASK_NAME: str = "900KW_mask_polygons.json"

_MASKS_PKG = "PyHyperScattering.data.smi.masks"


def _bundled_mask_path(name: str) -> Path:
    """Return the absolute filesystem path to a bundled mask JSON."""
    res = files(_MASKS_PKG).joinpath(name)
    # `files()` returns a Traversable; for a package shipped as a regular
    # directory this is a real filesystem path. For a wheel installed
    # zipfile, the caller would need to read bytes — we keep things simple
    # and require the package to be installed normally (PyHyper already is).
    return Path(str(res))


def default_saxs_mask_path() -> Path:
    """Path to the bundled SAXS (Pilatus 2M) polygon mask."""
    return _bundled_mask_path(DEFAULT_SAXS_MASK_NAME)


def default_waxs_mask_path() -> Path:
    """Path to the bundled WAXS (900KW) polygon mask."""
    return _bundled_mask_path(DEFAULT_WAXS_MASK_NAME)


def resolve_mask_path(
    user_value: Union[str, Path, None],
    detector: str,
) -> Path | None:
    """Resolve a mask-path argument, falling back to bundled defaults.

    Parameters
    ----------
    user_value : str | Path | None
        What the caller passed in. Semantics:

        * ``None``  → return the bundled default for ``detector``.
        * absolute path or existing file → returned as-is.
        * bare filename matching a bundled mask → returns the bundled path
          (this lets the smi-browser keep passing legacy filenames like
          ``"pil2M_mask_polygons.json"`` without shipping its own copy).
        * any other string → returned as ``Path(value)`` unchanged so that
          callers still get a clear FileNotFoundError downstream.
    detector : {'saxs', 'waxs'}
        Which bundled default to use when ``user_value`` is None.

    Returns
    -------
    Path | None
        Resolved path, or ``None`` only if both ``user_value`` is None and
        no default is bundled for the requested detector (shouldn't happen).
    """
    detector = detector.lower()
    if detector not in ("saxs", "waxs"):
        raise ValueError(f"detector must be 'saxs' or 'waxs', got {detector!r}")

    default_fn = (
        default_saxs_mask_path if detector == "saxs" else default_waxs_mask_path
    )

    if user_value is None:
        return default_fn()

    user_path = Path(user_value)
    if user_path.is_absolute() and user_path.exists():
        return user_path
    if user_path.exists():
        return user_path.resolve()

    # Bare filename matching the bundled basename → use the bundle.
    if user_path.name == DEFAULT_SAXS_MASK_NAME and detector == "saxs":
        return default_saxs_mask_path()
    if user_path.name == DEFAULT_WAXS_MASK_NAME and detector == "waxs":
        return default_waxs_mask_path()

    # Hand back the original Path; downstream open() will raise a clear error.
    return user_path


# ---------------------------------------------------------------------------
# Detector field classification (SMI naming conventions)
# ---------------------------------------------------------------------------

#: Substrings (case-insensitive) that identify a SAXS detector field.
SAXS_DETECTOR_NAMES: frozenset = frozenset({"pil2m", "pilatus2m", "saxs"})
#: Substrings (case-insensitive) that identify a WAXS detector field.
WAXS_DETECTOR_NAMES: frozenset = frozenset({"900kw", "waxs"})

DetectorKind = Literal["saxs", "waxs"]

#: SMI primary-stream field names for the two detectors.
SAXS_IMAGE_FIELD: str = "pil2M_image"
WAXS_IMAGE_FIELD: str = "pil900KW_image"
WAXS_ARC_FIELD: str = "waxs_arc"
WAXS_BSX_FIELD: str = "waxs_bsx"

DEFAULT_DETECTOR_FIELDS: dict = {
    "saxs": SAXS_IMAGE_FIELD,
    "waxs": WAXS_IMAGE_FIELD,
}


def classify_detector_field(name: str) -> Optional[str]:
    """Classify a detector/field name as ``'saxs'``, ``'waxs'``, or ``None``.

    Matching is case-insensitive substring against
    :data:`SAXS_DETECTOR_NAMES` / :data:`WAXS_DETECTOR_NAMES`. WAXS is
    checked first to avoid the rare case where a name contains both
    ``"waxs"`` and ``"saxs"`` substrings (e.g. ``"waxs_saxs_combined"``).
    """
    if name is None:
        return None
    lowered = str(name).lower()
    for tag in WAXS_DETECTOR_NAMES:
        if tag in lowered:
            return "waxs"
    for tag in SAXS_DETECTOR_NAMES:
        if tag in lowered:
            return "saxs"
    return None


# ---------------------------------------------------------------------------
# Beamline / loader calibration constants
# ---------------------------------------------------------------------------

#: SMI mechanical linkage: change in waxs_bsx (mm) per degree of waxs_arc.
#: Identical to the ``_BSX_PER_ARC_DEG`` constant inside
#: :mod:`PyHyperScattering.SMISWAXSIntegrator`.
BSX_PER_ARC_DEG: float = -4.39


@dataclass(frozen=True)
class LoaderCalibration:
    """Calibrated default deltas applied by :class:`SMISWAXSLoader`.

    These mirror the ``_{SAXS,WAXS}_DEFAULT_*_PX`` / ``_MM`` module-level
    constants in :mod:`PyHyperScattering.SMISWAXSLoader`.  External
    callers can introspect "what would the loader use if I pass None?"
    without instantiating the loader.
    """
    saxs_row_delta_px: float = 2.0
    saxs_col_delta_px: float = 3.0
    waxs_row_delta_px: float = 0.0
    waxs_col_delta_px: float = 0.0
    saxs_distance_delta_mm: float = -20.0


LOADER_DEFAULTS: LoaderCalibration = LoaderCalibration()


# ---------------------------------------------------------------------------
# Display orientation (pure numpy)
# ---------------------------------------------------------------------------

def _validate_detector(detector: str) -> str:
    d = str(detector).lower()
    if d not in ("saxs", "waxs"):
        raise ValueError(f"detector must be 'saxs' or 'waxs', got {detector!r}")
    return d


def orient_frame_for_display(arr: np.ndarray, detector: str) -> np.ndarray:
    """Apply the canonical display orientation for an SMI detector frame.

    * WAXS: ``np.fliplr(np.rot90(arr, k=3))`` (equivalent to ``arr.T``)
    * SAXS: ``np.flipud(arr)``
    """
    d = _validate_detector(detector)
    a = np.asarray(arr)
    if d == "waxs":
        return np.fliplr(np.rot90(a, k=3))
    return np.flipud(a)


def orient_polygon_xy(
    col_raw: float,
    row_raw: float,
    detector: str,
    raw_shape: tuple,
) -> tuple:
    """Map a raw-detector ``(col, row)`` vertex to display ``(x, y)``.

    Bokeh's image glyph draws ``array[0, :]`` at the bottom of the figure,
    so the math accounts for that placement.

    SAXS:  ``(col, raw_h - row)``
    WAXS:  ``(row, col)``
    """
    d = _validate_detector(detector)
    raw_h, raw_w = int(raw_shape[0]), int(raw_shape[1])
    if d == "waxs":
        return (float(row_raw), float(col_raw))
    return (float(col_raw), float(raw_h) - float(row_raw))


def orient_polygon_xy_inverse(
    x: float,
    y: float,
    detector: str,
    raw_shape: tuple,
) -> tuple:
    """Inverse of :func:`orient_polygon_xy`. Use when saving edits."""
    d = _validate_detector(detector)
    raw_h, _raw_w = int(raw_shape[0]), int(raw_shape[1])
    if d == "waxs":
        return (float(y), float(x))
    return (float(x), float(raw_h) - float(y))


# ---------------------------------------------------------------------------
# Mask file I/O (schema-normalized)
# ---------------------------------------------------------------------------

NormalizedMask = dict


def _coerce_polygon(verts: Any) -> list:
    """Return ``[[col, row], ...]`` of floats for an iterable of 2-vertices."""
    out = []
    for v in verts:
        # Accept (col, row) tuples / lists / numpy rows.
        c, r = v[0], v[1]
        out.append([float(c), float(r)])
    return out


def load_mask_polygons(path) -> NormalizedMask:
    """Load a polygon-mask JSON file and return a normalized dict.

    Two on-disk schemas are supported:

    * **Nested** (SAXS / pil2M)::

        {
            "image_shape": [rows, cols],
            "static_regions": {name: [[col, row], ...], ...},
            "beamstops": {name: <polygon or wrapped polygon>, ...},
        }

      A beamstop entry may either be a bare polygon or a dict
      ``{"polygon": [...], ...}`` (e.g. carrying ``"x_motor_key"``,
      ``"reference_mm"``, etc.).  The wrapper is unwrapped on load.

    * **Flat** (WAXS / 900KW)::

        {name: [[col, row], ...], ...}

      Names containing ``"beamstop"`` (case-insensitive) are routed to
      the ``beamstops`` bucket, everything else to ``static_regions``.
      A top-level ``"image_shape"`` entry is also honored.

    Returns
    -------
    dict
        Always shaped as::

            {
                "image_shape": [rows, cols] | None,
                "static_regions": {name: [[col, row], ...], ...},
                "beamstops":      {name: [[col, row], ...], ...},
            }

        Coordinates are in **raw detector indexing** (col, row), unchanged
        from the source file.
    """
    p = Path(path)
    with open(p) as f:
        data = json.load(f)

    image_shape = data.get("image_shape")
    if image_shape is not None:
        image_shape = list(image_shape)

    static_regions: dict = {}
    beamstops: dict = {}

    if "static_regions" in data or "beamstops" in data:
        # Nested schema (SAXS-style)
        for name, verts in (data.get("static_regions") or {}).items():
            static_regions[str(name)] = _coerce_polygon(verts)
        for name, entry in (data.get("beamstops") or {}).items():
            if isinstance(entry, dict):
                poly = entry.get("polygon")
                if poly is None:
                    # legacy variant: nested {"polygons": [[...]]}
                    polys = entry.get("polygons")
                    poly = polys[0] if polys else []
            else:
                poly = entry
            beamstops[str(name)] = _coerce_polygon(poly) if poly else []
    else:
        # Flat schema (WAXS-style)
        for name, verts in data.items():
            if name == "image_shape":
                continue
            key = str(name)
            target = beamstops if "beamstop" in key.lower() else static_regions
            target[key] = _coerce_polygon(verts) if verts else []

    return {
        "image_shape": image_shape,
        "static_regions": static_regions,
        "beamstops": beamstops,
    }


def save_mask_polygons(mask: NormalizedMask, path) -> None:
    """Write a normalized mask dict back out using the nested schema.

    Empty ``static_regions`` / ``beamstops`` buckets are written as ``{}``
    (not omitted), so a round-trip ``load_mask_polygons → save_mask_polygons``
    is structurally stable for nested-schema inputs.
    """
    payload: dict = {}
    if mask.get("image_shape") is not None:
        payload["image_shape"] = list(mask["image_shape"])
    payload["static_regions"] = {
        str(k): _coerce_polygon(v) for k, v in (mask.get("static_regions") or {}).items()
    }
    payload["beamstops"] = {
        str(k): _coerce_polygon(v) for k, v in (mask.get("beamstops") or {}).items()
    }
    with open(Path(path), "w") as f:
        json.dump(payload, f, indent=2)
