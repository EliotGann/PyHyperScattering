"""
TiledSMISWAXSLoader
====================
A PyHyperScattering-compatible tiled data loader for the SMI WAXS+SAXS
instrument at NSLS-II.

Design principles
-----------------
- Follows the FileLoader attribute contract used by SST1RSoXSDB /
  PFGeneralIntegrator so that an existing PFGeneralIntegrator(geomethod=
  'template_xr') call works without modification.
- Detects SAXS (Pilatus 2M, fixed flat panel) and WAXS (900KW, 3-panel folded
  arc detector) separately and returns correctly-typed xr.DataArray objects.
- All geometry arrives as xr.DataArray.attrs, mirroring the PyHyperScattering
  convention: dist, poni1, poni2, rot1, rot2, rot3, pixel1, pixel2, energy,
  wavelength.  SMI-specific extras (panel geometry, arc angle, beamstop …) live
  under the ``smi_`` prefix.
- Metadata fallback order: user overrides > baseline stream > primary
  configuration > start > defaults.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import time

import numpy as np
import xarray as xr


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PILATUS_PIXEL_SIZE_M: float = 0.172e-3           # 172 µm in metres
_HBAR_C_EV_M: float = 1.239841984e-6             # eV·m  →  λ = hbar_c / E(eV)

SAXS_IMAGE_FIELD = "pil2M_image"
WAXS_IMAGE_FIELD = "pil900KW_image"
WAXS_ARC_FIELD   = "waxs_arc"
WAXS_BSX_FIELD   = "waxs_bsx"

DEFAULT_TILED_URI = "https://tiled.nsls2.bnl.gov"
DEFAULT_CATALOG   = "smi/migration"
DEFAULT_ENERGY_KEV = 16.1

# SAXS defaults (Pilatus 2M at SMI long-distance position)
_SAXS_DEFAULT_DISTANCE_MM  = 2000.0
_SAXS_DEFAULT_BEAM_ROW_PX  = 1165.0          # fallback if metadata absent
_SAXS_DEFAULT_BEAM_COL_PX  =  746.0          # fallback if metadata absent
_SAXS_DEFAULT_DISTANCE_DELTA_MM = -20.0       # additive correction to motor z
_SAXS_DEFAULT_BEAM_DELTA_ROW_PX =  2.0        # additive correction to metadata row
_SAXS_DEFAULT_BEAM_DELTA_COL_PX =  3.0        # additive correction to metadata col

# WAXS defaults (900KW arc detector at ~274 mm)
_WAXS_DEFAULT_DISTANCE_MM  = 270.0
_WAXS_DEFAULT_BEAM_ROW_PX  = 217.0            # fallback if metadata absent
_WAXS_DEFAULT_BEAM_COL_PX  = 319.0            # fallback if metadata absent
_WAXS_DEFAULT_BEAM_DELTA_ROW_PX =  0.0        # additive correction to metadata row
_WAXS_DEFAULT_BEAM_DELTA_COL_PX = -2.0        # additive correction to metadata col
_WAXS_DEFAULT_PANEL_OFFSETS_DEG = (-7.0, 0.0, 7.0)
_WAXS_DEFAULT_PANEL_COL_RANGES  = ((0, 206), (206, 413), (413, 619))
_WAXS_ROTATION_K = 3                             # np.rot90 k-value


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _as_scalar(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "values"):
        value = value.values
    arr = np.asarray(value)
    if arr.size == 0:
        return None
    item = arr.reshape(-1)[0]
    return item.item() if hasattr(item, "item") else item


def _read_baseline(run: Any) -> xr.Dataset | None:
    """Read the baseline stream as an xr.Dataset (legacy path).

    Falls back to the new bluesky-tiled-plugins layout where baseline is
    accessed via ``run["baseline"]["internal"]`` (a DataFrameClient) rather
    than ``run["baseline"].read()`` (which raises KeyError('data')).
    """
    try:
        return run["baseline"].read()
    except (KeyError, Exception):
        pass
    # New layout: baseline["internal"] is a DataFrameClient (pandas-like).
    # Convert to xr.Dataset so existing _dataset_scalar() calls still work.
    try:
        internal = run["baseline"]["internal"]
        df = internal.read() if hasattr(internal, "read") else None
        if df is not None:
            import pandas as pd
            if isinstance(df, pd.DataFrame):
                return xr.Dataset.from_dataframe(df)
            # May already be an xr.Dataset
            if isinstance(df, xr.Dataset):
                return df
    except Exception:
        pass
    return None


def _baseline_scalar(run: Any, key: str) -> Any:
    """Read a single scalar from the baseline stream (first value).

    Tries the efficient per-column access path on the new tiled layout
    first (avoids pulling all 564 columns), then falls back to reading the
    full baseline as xr.Dataset.
    """
    # Fast path: baseline/internal DataFrameClient with per-column access
    try:
        internal = run["baseline"]["internal"]
        columns = list(internal)
        if key in columns:
            vals = internal[key].read() if hasattr(internal[key], "read") else internal[key][...]
            return _as_scalar(np.asarray(vals))
    except Exception:
        pass
    # Fallback: full baseline read (old layout or xr.Dataset path)
    baseline = _read_baseline(run)
    return _dataset_scalar(baseline, key)


def _dataset_scalar(ds: xr.Dataset | None, key: str) -> Any:
    if ds is None or key not in ds:
        return None
    return _as_scalar(ds[key].values)


def _conf_scalar(conf: dict, key: str) -> Any:
    return _as_scalar(conf.get(key))


def _primary_conf(run: Any, det_key: str) -> dict:
    try:
        return (
            run["primary"]
            .metadata.get("configuration", {})
            .get(det_key, {})
            .get("data", {})
        )
    except Exception:
        return {}


def _primary_scalar(run: Any, field: str) -> Any:
    """Read a scalar motor/signal value from the primary stream.

    Only returns a value if the field exists in primary AND has a single
    unique value (i.e., it's a "read" companion, not the varying scan axis).
    For varying scan axes, use _read_scan_axis() instead.
    """
    if not _has_primary_field(run, field):
        return None
    try:
        node = _get_primary_field_node(run, field)
        values = node.read() if hasattr(node, "read") else node[...]
        arr = np.asarray(values, dtype=float)
        # If all values are the same, treat as a scalar
        if arr.size > 0 and np.all(arr == arr[0]):
            return float(arr[0])
        # If values vary, return the first (start-of-scan position)
        if arr.size > 0:
            return float(arr[0])
    except Exception:
        pass
    return None


def _energy_to_wavelength_m(energy_ev: float) -> float:
    return _HBAR_C_EV_M / float(energy_ev)


# ---------------------------------------------------------------------------
# Sample name parsing
# ---------------------------------------------------------------------------

def parse_sample_name_geometry(sample_name: str) -> dict[str, float]:
    """Extract geometry parameters encoded in the sample_name string.

    Common SMI naming conventions:
      _wa{X}_   → WAXS arc angle (degrees)
      _sdd{X}m  → sample-detector distance (metres)
      _{X}keV   → photon energy (keV)
      _ai{X}_   → incident angle (degrees)
      _th{X}_   → sample theta (degrees)

    Returns a dict with only the keys that were successfully parsed.
    """
    result: dict[str, float] = {}

    # WAXS arc angle: _wa20.0_ or _wa20.0 (at end)
    m = re.search(r"_wa([\d.]+)", sample_name)
    if m:
        result["waxs_arc_deg"] = float(m.group(1))

    # Sample-detector distance: _sdd2.0m or _sdd2.0m_ (value in metres)
    m = re.search(r"_sdd([\d.]+)m?", sample_name)
    if m:
        result["sdd_m"] = float(m.group(1))

    # Photon energy: _16.10keV_
    m = re.search(r"_([\d.]+)keV", sample_name)
    if m:
        result["energy_kev"] = float(m.group(1))

    # Incident angle: _ai0.12_
    m = re.search(r"_ai([\d.]+)", sample_name)
    if m:
        result["incident_angle_deg"] = float(m.group(1))

    # Sample theta: _th0.5_
    m = re.search(r"_th([\d.]+)", sample_name)
    if m:
        result["theta_deg"] = float(m.group(1))

    return result


# ---------------------------------------------------------------------------
# Geometry resolution dataclasses
# ---------------------------------------------------------------------------

@dataclass
class SAXSGeometry:
    """Resolved geometry parameters for the SAXS (Pilatus 2M) detector."""
    dist_m: float
    poni1_m: float
    poni2_m: float
    pixel1_m: float = PILATUS_PIXEL_SIZE_M
    pixel2_m: float = PILATUS_PIXEL_SIZE_M
    rot1: float = 0.0
    rot2: float = 0.0
    rot3: float = 0.0
    energy_ev: float = DEFAULT_ENERGY_KEV * 1000.0
    wavelength_m: float = _energy_to_wavelength_m(DEFAULT_ENERGY_KEV * 1000.0)
    beam_center_row_px: float = _SAXS_DEFAULT_BEAM_ROW_PX
    beam_center_col_px: float = _SAXS_DEFAULT_BEAM_COL_PX
    active_beamstop: str = "rod"
    beamstop_pos_mm: dict | None = None


@dataclass
class WAXSPanelGeometry:
    """Geometry for a single WAXS panel."""
    col_start: int
    col_end: int
    offset_deg: float
    row_shift_px: float = 0.0
    col_shift_px: float = 0.0


@dataclass
class WAXSGeometry:
    """Resolved geometry for the WAXS (900KW 3-panel arc) detector."""
    dist_m: float
    beam_center_row_px: float
    beam_center_col_px: float
    pixel_m: float = PILATUS_PIXEL_SIZE_M
    energy_ev: float = DEFAULT_ENERGY_KEV * 1000.0
    wavelength_m: float = _energy_to_wavelength_m(DEFAULT_ENERGY_KEV * 1000.0)
    theta_zero_deg: float = 0.0
    sample_offset_x_mm: float = 0.0
    sample_offset_z_mm: float = 0.0
    rotation_k: int = _WAXS_ROTATION_K
    panels: tuple[WAXSPanelGeometry, ...] = ()


# ---------------------------------------------------------------------------
# Geometry resolvers
# ---------------------------------------------------------------------------

def resolve_saxs_geometry(
    run: Any,
    energy_kev: float | None = None,
    **overrides: Any,
) -> SAXSGeometry:
    """Resolve full SAXS geometry from a tiled run.

    Fallback order for each parameter:
      1. User override (``overrides`` dict)
      2. Primary stream (per-frame value, if field present)
      3. Baseline stream (start-of-scan snapshot — always present)
      4. Primary configuration metadata
      5. Start metadata / sample_name encoding
      6. Hardcoded instrument defaults
    """
    baseline = _read_baseline(run)
    conf = _primary_conf(run, "pil2M")
    start = run.metadata.get("start", {})
    sample_name = start.get("sample_name", "")
    name_geo = parse_sample_name_geometry(sample_name)

    # Energy resolution: override > start metadata > baseline > sample_name > default
    if energy_kev is None:
        _baseline_energy_ev = _baseline_scalar(run, "energy_energy")
        _baseline_energy_kev = (
            _baseline_energy_ev / 1000.0 if _baseline_energy_ev is not None else None
        )
        energy_kev = (
            start.get("energy")
            or _baseline_energy_kev
            or name_geo.get("energy_kev")
            or DEFAULT_ENERGY_KEV
        )
    energy_ev = float(energy_kev) * 1000.0

    # Beam center: override > baseline > primary conf > default
    beam_row = (
        overrides.get("beam_center_row_px")
        or _baseline_scalar(run, "pil2M_beam_center_y_px")
        or _dataset_scalar(baseline, "pil2M_beam_center_y_px")
        or _conf_scalar(conf, "pil2M_beam_center_y_px")
        or _SAXS_DEFAULT_BEAM_ROW_PX
    )
    beam_col = (
        overrides.get("beam_center_col_px")
        or _baseline_scalar(run, "pil2M_beam_center_x_px")
        or _dataset_scalar(baseline, "pil2M_beam_center_x_px")
        or _conf_scalar(conf, "pil2M_beam_center_x_px")
        or _SAXS_DEFAULT_BEAM_COL_PX
    )

    # Distance: override > primary > baseline > sample_name > conf > default
    _sdd_from_name = name_geo.get("sdd_m")
    _sdd_from_name_mm = _sdd_from_name * 1000.0 if _sdd_from_name is not None else None
    dist_mm = (
        overrides.get("sample_distance_mm")
        or _primary_scalar(run, "pil2M_motor_z")
        or _baseline_scalar(run, "pil2M_motor_z_user_setpoint")
        or _baseline_scalar(run, "pil2M_motor_z")
        or _dataset_scalar(baseline, "pil2M_motor_z_user_setpoint")
        or _dataset_scalar(baseline, "pil2M_motor_z")
        or _conf_scalar(conf, "pil2M_sdd_mm")
        or _sdd_from_name_mm
        or _SAXS_DEFAULT_DISTANCE_MM
    )
    active_bs = (
        overrides.get("active_beamstop")
        or _baseline_scalar(run, "pil2M_active_beamstop")
        or _dataset_scalar(baseline, "pil2M_active_beamstop")
        or _conf_scalar(conf, "pil2M_active_beamstop")
        or "rod"
    )

    bs_pos = {
        "rod": {
            "x": (
                _baseline_scalar(run, "saxs_beamstop_x_rod_user_setpoint")
                or _dataset_scalar(baseline, "saxs_beamstop_x_rod_user_setpoint")
                or _baseline_scalar(run, "saxs_beamstop_x_rod")
                or _dataset_scalar(baseline, "saxs_beamstop_x_rod")
            ),
            "y": (
                _baseline_scalar(run, "saxs_beamstop_y_rod_user_setpoint")
                or _dataset_scalar(baseline, "saxs_beamstop_y_rod_user_setpoint")
                or _baseline_scalar(run, "saxs_beamstop_y_rod")
                or _dataset_scalar(baseline, "saxs_beamstop_y_rod")
            ),
        },
        "pin": {
            "x": (
                _baseline_scalar(run, "saxs_beamstop_x_pin_user_setpoint")
                or _dataset_scalar(baseline, "saxs_beamstop_x_pin_user_setpoint")
                or _baseline_scalar(run, "saxs_beamstop_x_pin")
                or _dataset_scalar(baseline, "saxs_beamstop_x_pin")
            ),
            "y": (
                _baseline_scalar(run, "saxs_beamstop_y_pin_user_setpoint")
                or _dataset_scalar(baseline, "saxs_beamstop_y_pin_user_setpoint")
                or _baseline_scalar(run, "saxs_beamstop_y_pin")
                or _dataset_scalar(baseline, "saxs_beamstop_y_pin")
            ),
        },
    }

    beam_row = float(beam_row)
    beam_col = float(beam_col)
    dist_mm  = float(dist_mm)

    # Apply additive distance correction (default from calibration; overridable)
    dist_delta_mm = float(
        overrides.get("distance_delta_mm", _SAXS_DEFAULT_DISTANCE_DELTA_MM)
    )
    dist_mm += dist_delta_mm

    # Apply additive beam-center corrections on top of metadata values
    beam_row += float(
        overrides.get("beam_delta_row_px", _SAXS_DEFAULT_BEAM_DELTA_ROW_PX)
    )
    beam_col += float(
        overrides.get("beam_delta_col_px", _SAXS_DEFAULT_BEAM_DELTA_COL_PX)
    )

    return SAXSGeometry(
        dist_m=dist_mm / 1000.0,
        poni1_m=beam_row * PILATUS_PIXEL_SIZE_M,
        poni2_m=beam_col * PILATUS_PIXEL_SIZE_M,
        energy_ev=energy_ev,
        wavelength_m=_energy_to_wavelength_m(energy_ev),
        beam_center_row_px=beam_row,
        beam_center_col_px=beam_col,
        active_beamstop=str(active_bs),
        beamstop_pos_mm=bs_pos,
    )


def resolve_waxs_geometry(
    run: Any,
    energy_kev: float | None = None,
    **overrides: Any,
) -> WAXSGeometry:
    """Resolve full WAXS geometry from a tiled run.

    Fallback order for each parameter:
      1. User override (``overrides`` dict)
      2. Primary stream (per-frame value, if field present)
      3. Baseline stream (start-of-scan snapshot — always present)
      4. Primary configuration metadata
      5. Start metadata / sample_name encoding
      6. Hardcoded instrument defaults
    """
    baseline = _read_baseline(run)
    conf = _primary_conf(run, "pil900KW")
    start = run.metadata.get("start", {})
    sample_name = start.get("sample_name", "")
    name_geo = parse_sample_name_geometry(sample_name)

    # Energy resolution: override > start metadata > baseline > sample_name > default
    if energy_kev is None:
        _baseline_energy_ev = _baseline_scalar(run, "energy_energy")
        _baseline_energy_kev = (
            _baseline_energy_ev / 1000.0 if _baseline_energy_ev is not None else None
        )
        energy_kev = (
            start.get("energy")
            or _baseline_energy_kev
            or name_geo.get("energy_kev")
            or DEFAULT_ENERGY_KEV
        )
    energy_ev = float(energy_kev) * 1000.0

    dist_mm = (
        overrides.get("sample_distance_mm")
        or _primary_scalar(run, "pil900KW_motor_z")
        or _baseline_scalar(run, "pil900KW_motor_z_user_setpoint")
        or _baseline_scalar(run, "pil900KW_motor_z")
        or _dataset_scalar(baseline, "pil900KW_motor_z_user_setpoint")
        or _dataset_scalar(baseline, "pil900KW_motor_z")
        or _conf_scalar(conf, "pil900KW_sdd_mm")
        or _WAXS_DEFAULT_DISTANCE_MM
    )
    # NOTE: The ophyd configuration stores WAXS beam center in the raw
    # (un-rotated) image frame, which is incompatible with the rotated
    # coordinate system used by WAXSCalibration / MultiPanelArcDetector.
    # Always use the calibrated defaults as the base; fine-tune via deltas.
    beam_row = float(
        overrides.get("beam_center_row_px")
        or _WAXS_DEFAULT_BEAM_ROW_PX
    )
    beam_col = float(
        overrides.get("beam_center_col_px")
        or _WAXS_DEFAULT_BEAM_COL_PX
    )

    # Apply additive beam-center corrections on top of metadata values
    beam_row += float(
        overrides.get("beam_delta_row_px", _WAXS_DEFAULT_BEAM_DELTA_ROW_PX)
    )
    beam_col += float(
        overrides.get("beam_delta_col_px", _WAXS_DEFAULT_BEAM_DELTA_COL_PX)
    )

    panel_offsets = overrides.get(
        "panel_offsets_deg", _WAXS_DEFAULT_PANEL_OFFSETS_DEG
    )
    panel_cols = overrides.get(
        "panel_col_ranges", _WAXS_DEFAULT_PANEL_COL_RANGES
    )
    panels = tuple(
        WAXSPanelGeometry(
            col_start=int(c0),
            col_end=int(c1),
            offset_deg=float(off),
        )
        for (c0, c1), off in zip(panel_cols, panel_offsets)
    )

    return WAXSGeometry(
        dist_m=float(dist_mm) / 1000.0,
        beam_center_row_px=beam_row,
        beam_center_col_px=beam_col,
        energy_ev=energy_ev,
        wavelength_m=_energy_to_wavelength_m(energy_ev),
        theta_zero_deg=float(overrides.get("theta_zero_deg", 0.0)),
        sample_offset_x_mm=float(overrides.get("sample_offset_x_mm", 0.0)),
        sample_offset_z_mm=float(overrides.get("sample_offset_z_mm", 0.0)),
        panels=panels,
    )


# ---------------------------------------------------------------------------
# Image loading from Tiled
# ---------------------------------------------------------------------------

def _get_primary_field_node(run: Any, field: str) -> Any:
    """Return the tiled ArrayClient (or xarray-like) node for a primary field.

    Avoids calling ``run["primary"].read()`` which would pull every variable
    in the primary stream over the network.  Tries the common bluesky/tiled
    layouts (``primary/data/<field>`` then ``primary/<field>``).
    """
    primary = run["primary"]
    # Modern bluesky-tiled layout: primary -> data -> <field>
    try:
        data_node = primary["data"]
    except Exception:
        data_node = None
    if data_node is not None:
        try:
            return data_node[field]
        except Exception:
            pass
    # Fallback: field hangs directly off primary
    try:
        return primary[field]
    except Exception as exc:  # pragma: no cover - defensive
        raise KeyError(
            f"Field '{field}' not found in primary stream of run."
        ) from exc


# Tiled chunks larger than this estimated byte size are pre-emptively read
# frame-by-frame instead of as a single bulk request.  The SMI ``pil2M_image``
# field is chunked at ~99 MB per chunk, which the tiled server has been
# observed to reject with HTTP 500 even when smaller chunks (e.g. the WAXS
# ``pil900KW_image`` at ~36 MB per chunk) succeed.  Threshold is intentionally
# conservative; the per-frame fallback path is reliable but slower.
_BULK_READ_MAX_CHUNK_BYTES = 64 * 1024 * 1024  # 64 MiB

# How many times to retry a single per-frame read on a transient server error
# before giving up.  Backoff is linear: 1s, 2s, 3s, ...
_PER_FRAME_RETRIES = 4


def _estimate_max_chunk_bytes(node: Any) -> int | None:
    """Return the byte size of the largest tiled chunk for ``node``, or None.

    Uses ``node.chunks`` (tuple of per-axis chunk-size tuples) and ``dtype``
    if available.  Returns ``None`` when the information is missing so the
    caller can fall back to a bulk read.
    """
    chunks = getattr(node, "chunks", None)
    dtype = getattr(node, "dtype", None)
    if not chunks or dtype is None:
        return None
    try:
        itemsize = int(np.dtype(dtype).itemsize)
        # Largest chunk along each axis multiplied together
        max_elems = 1
        for axis_chunks in chunks:
            if not axis_chunks:
                return None
            max_elems *= int(max(axis_chunks))
        return max_elems * itemsize
    except Exception:
        return None


def _read_array_via_http_full(node: Any) -> np.ndarray | None:
    """Fetch an entire tiled array via the raw ``/array/full`` endpoint.

    Bypasses the tiled client's slice serialiser (which in v0.2.x emits
    ``?slice=:N:1,:M:1,:K:1`` with explicit strides on every axis — a form
    the production NSLS-II tiled server rejects with HTTP 500).  Issuing
    the request with no ``slice`` query parameter at all asks for the
    whole array and works regardless of client version.

    Returns ``None`` if the node does not expose enough metadata to use
    this fast-path so the caller can fall back to ``node.read()``.
    """
    item = getattr(node, "item", None)
    links = (item or {}).get("links") or {}
    full_url = links.get("full")
    http_client = None
    ctx = getattr(node, "context", None)
    if ctx is not None:
        http_client = getattr(ctx, "http_client", None)
    dtype = getattr(node, "dtype", None)
    shape = getattr(node, "shape", None)
    if not full_url or http_client is None or dtype is None or shape is None:
        return None

    resp = http_client.get(
        full_url,
        params={"format": "application/octet-stream"},
        headers={"Accept": "application/octet-stream"},
        timeout=300.0,
    )
    if resp.status_code != 200:
        raise RuntimeError(
            f"tiled /array/full returned HTTP {resp.status_code}: "
            f"{resp.text[:200]}"
        )
    return _decode_array_response(
        resp.content, dtype, tuple(int(s) for s in shape),
    )


def _decode_array_response(content: bytes, dtype: Any, shape: tuple[int, ...]) -> np.ndarray:
    """Reconstruct a numpy array from a tiled ``/array/full`` response body."""
    return np.frombuffer(content, dtype=np.dtype(dtype)).reshape(shape)


def _read_one_frame_via_http(node: Any, i: int) -> np.ndarray | None:
    """Fetch a single frame using the raw ``/array/full`` HTTP endpoint.

    Works around a serialisation incompatibility in newer ``tiled`` clients
    (>=0.2): they format slices as ``?slice=:1:1,:N:1,:M:1`` (with explicit
    strides on every axis), which the production NSLS-II tiled server
    rejects with HTTP 500.  The plain ``?slice=i:i+1,:,:`` form works.

    Returns ``None`` if the node does not expose enough metadata to use
    this fast-path, so the caller can fall back to the high-level client.
    """
    item = getattr(node, "item", None)
    links = (item or {}).get("links") or {}
    full_url = links.get("full")
    http_client = None
    ctx = getattr(node, "context", None)
    if ctx is not None:
        http_client = getattr(ctx, "http_client", None)
    dtype = getattr(node, "dtype", None)
    shape = getattr(node, "shape", None)
    if not full_url or http_client is None or dtype is None or shape is None:
        return None
    if len(shape) < 1:
        return None

    # Build a slice spec the tiled server accepts: ``i:i+1`` on the leading
    # axis, plain ``:`` on every other axis, no strides.
    slice_parts = [f"{i}:{i + 1}"] + [":"] * (len(shape) - 1)
    slice_spec = ",".join(slice_parts)

    resp = http_client.get(
        full_url,
        params={"slice": slice_spec, "format": "application/octet-stream"},
        headers={"Accept": "application/octet-stream"},
        timeout=120.0,
    )
    if resp.status_code != 200:
        # Surface the error so the caller's retry loop can see it.
        raise RuntimeError(
            f"tiled /array/full returned HTTP {resp.status_code} for "
            f"frame {i}: {resp.text[:200]}"
        )
    frame_shape = (1, *tuple(int(s) for s in shape[1:]))
    return _decode_array_response(resp.content, dtype, frame_shape)


def _read_one_frame_with_retry(node: Any, i: int) -> np.ndarray:
    """Read frame ``i`` from ``node`` with retries for transient failures."""
    last_exc: Exception | None = None
    for attempt in range(_PER_FRAME_RETRIES):
        # Preferred path: raw HTTP with a server-friendly slice spec.
        # This avoids the tiled>=0.2 client bug where slices on multi-dim
        # arrays are serialised as ``:1:1,:N:1,:M:1`` (which the NSLS-II
        # tiled server rejects with HTTP 500).
        try:
            frame = _read_one_frame_via_http(node, i)
            if frame is not None:
                return frame
        except Exception as exc:  # noqa: BLE001
            last_exc = exc

        # Fallback 1: high-level client indexing.
        try:
            return np.asarray(node[i : i + 1])
        except Exception as exc:  # noqa: BLE001 - tiled error types vary
            last_exc = exc

        # Fallback 2: ``read(slice=...)`` for clients without __getitem__.
        try:
            return np.asarray(node.read(slice=(slice(i, i + 1),)))
        except Exception as exc2:  # noqa: BLE001
            last_exc = exc2

        if attempt < _PER_FRAME_RETRIES - 1:
            time.sleep(1.0 * (attempt + 1))
    # Exhausted retries
    assert last_exc is not None
    raise last_exc


def _read_array_chunked(node: Any, parallel: bool = True, max_workers: int | None = None) -> np.ndarray:
    """Read a tiled array node frame-by-frame to avoid server-side 500s.

    The tiled server can return HTTP 500 when asked for a large multi-frame
    detector image in a single request.  Reading one frame at a time keeps
    each request small and works around the issue.  Falls back to a single
    ``read()`` for nodes that do not support indexed access.

    Parameters
    ----------
    node : tiled ArrayClient
        The tiled node to read from.
    parallel : bool
        If True (default), fetch frames concurrently using threads.
        Each frame is an independent HTTP request, so thread-based
        parallelism yields significant speedups on multi-frame scans.
    max_workers : int | None
        Maximum number of concurrent threads.  Defaults to min(n_frames, 8).
    """
    # Determine the leading dimension length
    shape = getattr(node, "shape", None)
    if shape is None or len(shape) == 0:
        return np.asarray(node.read())

    n = int(shape[0])
    if n == 0:
        return np.asarray(node.read())

    if parallel and n > 1:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        workers = max_workers if max_workers is not None else min(n, 8)
        frames = [None] * n
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_idx = {
                executor.submit(_read_one_frame_with_retry, node, i): i
                for i in range(n)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                frames[idx] = future.result()
        return np.concatenate(frames, axis=0)

    frames: list[np.ndarray] = []
    for i in range(n):
        frames.append(_read_one_frame_with_retry(node, i))
    return np.concatenate(frames, axis=0)


def _read_primary_field(run: Any, field: str) -> np.ndarray:
    """Read a single primary-stream field as a numpy array.

    Attempts a single bulk read first; on HTTP/server errors falls back to
    a frame-by-frame chunked read so that large detector arrays still load
    successfully when the tiled backend rejects an "all at once" request.
    """
    node = _get_primary_field_node(run, field)

    # Pre-emptively skip the bulk read for nodes whose tiled chunks exceed
    # the size threshold the server has been observed to reject.  The SMI
    # SAXS ``pil2M_image`` falls in this category; WAXS ``pil900KW_image``
    # does not.
    max_chunk_bytes = _estimate_max_chunk_bytes(node)
    skip_bulk = (
        max_chunk_bytes is not None
        and max_chunk_bytes > _BULK_READ_MAX_CHUNK_BYTES
    )

    arr: np.ndarray
    if skip_bulk:
        arr = _read_array_chunked(node)
    else:
        # Preferred bulk path: raw HTTP to the ``/array/full`` endpoint.
        # Avoids the tiled>=0.2 slice-serialisation bug that otherwise
        # makes ``node.read()`` 500 against the NSLS-II tiled server.
        arr = None  # type: ignore[assignment]
        try:
            arr = _read_array_via_http_full(node)
        except Exception:  # noqa: BLE001
            arr = None
        if arr is None:
            try:
                # Tiled ArrayClient — supports .read() returning a numpy array
                if hasattr(node, "read"):
                    raw = node.read()
                else:
                    raw = node[...]
                arr = np.asarray(raw)
            except Exception:  # noqa: BLE001 - tiled/httpx error types vary
                # Any failure during the bulk read (HTTP 500, dask compute
                # failure that wraps a server error, transient network blip,
                # etc.) — retry by streaming one frame at a time.  The
                # chunked path itself retries individual frames.
                arr = _read_array_chunked(node)

    # Tiled may return 4-D: (primary_step, exposures, row, col).
    # Average over the exposures axis to get (step, row, col).
    if arr.ndim == 4:
        arr = np.nanmean(arr, axis=1)
    return arr


def _has_primary_field(run: Any, field: str) -> bool:
    """Return True if the given field exists in the primary stream.

    Uses tiled container introspection to avoid downloading the entire
    primary stream just to check for a field's presence.
    """
    try:
        primary = run["primary"]
    except Exception:
        return False
    # Try modern bluesky-tiled layout first: primary/data/<field>
    try:
        data_node = primary["data"]
        if field in list(data_node):
            return True
    except Exception:
        pass
    # Fallback: field directly under primary
    try:
        return field in list(primary)
    except Exception:
        pass
    # Last-resort: full read (slow but correct)
    try:
        ds = primary.read()
        return field in ds
    except Exception:
        return False


def _read_scan_axis(run: Any, field: str) -> np.ndarray | None:
    """Read a motor field from primary; fall back to baseline if absent.

    Fallback order:
      1. Primary stream (per-frame values — only when motor is scanned)
      2. Baseline stream via efficient per-column access (new tiled layout)
      3. Baseline stream via full xr.Dataset read (old tiled layout)
      4. Sample name parsing (last resort)

    Reads only the requested field directly from tiled (avoids pulling the
    full primary stream, which contains the multi-GB detector arrays).
    """
    # 1. Primary stream (per-frame varying values)
    if _has_primary_field(run, field):
        try:
            node = _get_primary_field_node(run, field)
            values = node.read() if hasattr(node, "read") else node[...]
            return np.asarray(values, dtype=float)
        except Exception:
            pass

    # 2. Baseline stream (efficient per-column path)
    val = _baseline_scalar(run, field)
    if val is not None:
        return np.array([float(val)], dtype=float)

    # 3. Baseline via full xr.Dataset (legacy fallback)
    baseline = _read_baseline(run)
    val = _dataset_scalar(baseline, field)
    if val is not None:
        return np.array([float(val)], dtype=float)

    # 4. Sample name parsing (last resort for waxs_arc)
    if field == WAXS_ARC_FIELD:
        start = run.metadata.get("start", {})
        name_geo = parse_sample_name_geometry(start.get("sample_name", ""))
        arc_deg = name_geo.get("waxs_arc_deg")
        if arc_deg is not None:
            return np.array([arc_deg], dtype=float)

    return None


# ---------------------------------------------------------------------------
# Public loader: SAXS
# ---------------------------------------------------------------------------

def load_saxs_raw(
    run: Any,
    geo: SAXSGeometry,
    extra_attrs: dict[str, Any] | None = None,
) -> xr.DataArray:
    """
    Load SAXS (Pilatus 2M) raw images from a tiled run as an xr.DataArray.

    Returns
    -------
    xr.DataArray
        dims: (frame, pix_y, pix_x)  or (pix_y, pix_x) if single frame.
        attrs: PyHyperScattering-compatible geometry + SMI-specific extras.
    """
    images = _read_primary_field(run, SAXS_IMAGE_FIELD)
    start = run.metadata.get("start", {})
    sample_name = start.get("sample_name", "")
    name_geo = parse_sample_name_geometry(sample_name)

    # Resolve incident angle: primary > baseline > sample_name
    incident_angle_deg = (
        _primary_scalar(run, "stage_th")
        or _baseline_scalar(run, "stage_th")
        or name_geo.get("incident_angle_deg")
        or name_geo.get("theta_deg")
    )

    attrs: dict[str, Any] = {
        # PyHyperScattering / pyFAI geometry contract
        "dist":       geo.dist_m,
        "poni1":      geo.poni1_m,
        "poni2":      geo.poni2_m,
        "rot1":       geo.rot1,
        "rot2":       geo.rot2,
        "rot3":       geo.rot3,
        "pixel1":     geo.pixel1_m,
        "pixel2":     geo.pixel2_m,
        "energy":     geo.energy_ev,
        "wavelength": geo.wavelength_m * 1e10,   # angstroms (PyHyperScattering convention)
        # SMI-specific
        "smi_detector":           "saxs_pil2M",
        "smi_energy_kev":         geo.energy_ev / 1000.0,
        "smi_beam_center_row_px": geo.beam_center_row_px,
        "smi_beam_center_col_px": geo.beam_center_col_px,
        "smi_sample_distance_mm": geo.dist_m * 1000.0,
        "smi_active_beamstop":    geo.active_beamstop,
        "smi_incident_angle_deg": incident_angle_deg,
        # Run identity
        "uid":         start.get("uid", ""),
        "scan_id":     start.get("scan_id"),
        "sample_name": start.get("sample_name", ""),
    }
    if extra_attrs:
        attrs.update(extra_attrs)

    if images.ndim == 2:
        return xr.DataArray(images, dims=["pix_y", "pix_x"], attrs=attrs)

    if images.ndim != 3:
        raise ValueError(
            f"Expected 2-D or 3-D SAXS image array, got {images.ndim}-D "
            f"with shape {images.shape}"
        )
    n_frames = images.shape[0]
    arc_angles = _read_scan_axis(run, WAXS_ARC_FIELD)
    if arc_angles is not None and arc_angles.shape[0] == n_frames:
        frame_coord = arc_angles
        frame_dim_name = WAXS_ARC_FIELD
    else:
        frame_coord = np.arange(n_frames)
        frame_dim_name = "frame"

    return xr.DataArray(
        images,
        dims=[frame_dim_name, "pix_y", "pix_x"],
        coords={frame_dim_name: frame_coord},
        attrs=attrs,
    )


# ---------------------------------------------------------------------------
# Public loader: WAXS
# ---------------------------------------------------------------------------

def load_waxs_raw(
    run: Any,
    geo: WAXSGeometry,
    extra_attrs: dict[str, Any] | None = None,
) -> xr.DataArray:
    """
    Load WAXS (900KW) raw images from a tiled run as an xr.DataArray.

    Returns
    -------
    xr.DataArray
        dims: (waxs_arc, pix_y, pix_x)
        coords: waxs_arc — arc motor angles in degrees
        attrs: PyHyperScattering-compatible + SMI WAXS panel geometry
    """
    images = _read_primary_field(run, WAXS_IMAGE_FIELD)
    start  = run.metadata.get("start", {})
    sample_name = start.get("sample_name", "")
    name_geo = parse_sample_name_geometry(sample_name)

    # Resolve incident angle: primary > baseline > sample_name
    incident_angle_deg = (
        _primary_scalar(run, "stage_th")
        or _baseline_scalar(run, "stage_th")
        or name_geo.get("incident_angle_deg")
        or name_geo.get("theta_deg")
    )

    arc_angles = _read_scan_axis(run, WAXS_ARC_FIELD)
    bsx_values = _read_scan_axis(run, WAXS_BSX_FIELD)

    if images.ndim == 2:
        images = images[np.newaxis, :, :]
    if images.ndim != 3:
        raise ValueError(
            f"Expected 2-D or 3-D WAXS image array, got {images.ndim}-D "
            f"with shape {images.shape}"
        )
    n_frames = images.shape[0]

    if arc_angles is None:
        arc_angles = np.zeros(n_frames, dtype=float)
    elif arc_angles.shape[0] == 1 and n_frames > 1:
        arc_angles = np.full(n_frames, arc_angles[0], dtype=float)
    elif arc_angles.shape[0] != n_frames:
        arc_angles = np.zeros(n_frames, dtype=float)

    if bsx_values is None:
        bsx_values = np.zeros(n_frames, dtype=float)
    elif bsx_values.shape[0] == 1 and n_frames > 1:
        bsx_values = np.full(n_frames, bsx_values[0], dtype=float)
    elif bsx_values.shape[0] != n_frames:
        bsx_values = np.zeros(n_frames, dtype=float)

    panels_attr = [
        {
            "col_start":    p.col_start,
            "col_end":      p.col_end,
            "offset_deg":   p.offset_deg,
            "row_shift_px": p.row_shift_px,
            "col_shift_px": p.col_shift_px,
        }
        for p in geo.panels
    ]

    poni1_m = geo.beam_center_row_px * geo.pixel_m
    poni2_m = geo.beam_center_col_px * geo.pixel_m

    attrs: dict[str, Any] = {
        # PyHyperScattering / pyFAI geometry contract (centre panel, arc=0)
        "dist":       geo.dist_m,
        "poni1":      poni1_m,
        "poni2":      poni2_m,
        "rot1":       0.0,
        "rot2":       0.0,
        "rot3":       0.0,
        "pixel1":     geo.pixel_m,
        "pixel2":     geo.pixel_m,
        "energy":     geo.energy_ev,
        "wavelength": geo.wavelength_m * 1e10,   # angstroms
        # SMI WAXS-specific
        "smi_detector":              "waxs_pil900KW",
        "smi_energy_kev":            geo.energy_ev / 1000.0,
        "smi_beam_center_row_px":    geo.beam_center_row_px,
        "smi_beam_center_col_px":    geo.beam_center_col_px,
        "smi_sample_distance_mm":    geo.dist_m * 1000.0,
        "smi_theta_zero_deg":        geo.theta_zero_deg,
        "smi_sample_offset_x_mm":    geo.sample_offset_x_mm,
        "smi_sample_offset_z_mm":    geo.sample_offset_z_mm,
        "smi_rotation_k":            geo.rotation_k,
        "smi_panels":                panels_attr,
        "smi_waxs_bsx_per_frame":    bsx_values.tolist(),
        "smi_incident_angle_deg":    incident_angle_deg,
        # Run identity
        "uid":         start.get("uid", ""),
        "scan_id":     start.get("scan_id"),
        "sample_name": start.get("sample_name", ""),
    }
    if extra_attrs:
        attrs.update(extra_attrs)

    return xr.DataArray(
        images,
        dims=[WAXS_ARC_FIELD, "pix_y", "pix_x"],
        coords={WAXS_ARC_FIELD: arc_angles},
        attrs=attrs,
    )


# ---------------------------------------------------------------------------
# Scan info utility
# ---------------------------------------------------------------------------

def infer_detectors_and_steps(
    run: Any, primary: xr.Dataset | None = None,
) -> dict[str, Any]:
    """Inspect a tiled run to determine detectors, scan axes, and frame count.

    Parameters
    ----------
    run :
        Bluesky/tiled run object.
    primary : xr.Dataset, optional
        If provided, used directly (legacy fast path for callers that already
        have the full primary stream loaded).  When ``None`` (preferred), the
        function introspects the tiled ``primary`` container WITHOUT calling
        ``.read()`` on the detector image fields — only field names, shapes,
        and 1-D scan axes are fetched, which avoids the multi-GB request that
        can trigger an HTTP 500 from the tiled backend.
    """
    start = run.metadata.get("start", {})

    if primary is not None:
        vars_all = sorted(map(str, primary.data_vars.keys()))

        first_dim = next(iter(primary.dims), None)
        n_frames = (
            int(primary.sizes.get(first_dim, 0)) if first_dim is not None else 0
        )
        if n_frames == 0 and vars_all:
            first_var = primary[vars_all[0]]
            n_frames = int(first_var.shape[0]) if first_var.ndim > 0 else 1

        step_candidates: list[dict[str, Any]] = []
        for name in vars_all:
            da = primary[name]
            if da.ndim != 1:
                continue
            if int(da.shape[0]) != n_frames:
                continue
            if not np.issubdtype(da.dtype, np.number):
                continue
            values = np.asarray(da.values, dtype=float)
            finite = values[np.isfinite(values)]
            unique = np.unique(finite)
            if unique.size <= 1:
                continue
            step_candidates.append(
                {
                    "name": name,
                    "n_unique": int(unique.size),
                    "min": float(np.nanmin(values)),
                    "max": float(np.nanmax(values)),
                }
            )
    else:
        # Tiled-introspection path — no bulk reads of detector arrays.
        try:
            primary_node = run["primary"]
        except Exception:
            primary_node = None

        # Modern bluesky-tiled layout: primary -> data -> <field>
        data_node = None
        if primary_node is not None:
            try:
                data_node = primary_node["data"]
            except Exception:
                data_node = None
        field_container = data_node if data_node is not None else primary_node

        vars_all: list[str] = []
        if field_container is not None:
            try:
                vars_all = sorted(map(str, list(field_container)))
            except Exception:
                vars_all = []

        def _shape_of(name: str) -> tuple[int, ...]:
            try:
                node = field_container[name]
            except Exception:
                return ()
            shape = getattr(node, "shape", None)
            if shape is None:
                try:
                    shape = node.structure().shape  # tiled ArrayClient
                except Exception:
                    shape = ()
            return tuple(int(s) for s in shape) if shape else ()

        # Determine n_frames from the first array-like field with a
        # leading dimension.  Avoids reading detector data.
        n_frames = 0
        for name in vars_all:
            shp = _shape_of(name)
            if shp:
                n_frames = int(shp[0])
                break

        step_candidates: list[dict[str, Any]] = []
        for name in vars_all:
            shp = _shape_of(name)
            if len(shp) != 1 or shp[0] != n_frames:
                continue
            try:
                node = field_container[name]
                values = np.asarray(
                    node.read() if hasattr(node, "read") else node[...],
                    dtype=float,
                )
            except Exception:
                continue
            finite = values[np.isfinite(values)]
            unique = np.unique(finite)
            if unique.size <= 1:
                continue
            step_candidates.append(
                {
                    "name": name,
                    "n_unique": int(unique.size),
                    "min": float(np.nanmin(values)),
                    "max": float(np.nanmax(values)),
                }
            )

    detector_prefixes = sorted(
        {name.split("_")[0] for name in vars_all if "_" in name}
    )

    return {
        "uid": start.get("uid"),
        "scan_id": start.get("scan_id"),
        "sample_name": start.get("sample_name"),
        "n_frames": n_frames,
        "detectors_start": start.get("detectors", []) or [],
        "detector_prefixes_in_primary": detector_prefixes,
        "step_candidates": step_candidates,
        "detector_fields": {
            "saxs": [n for n in vars_all if n.startswith("pil2M_")],
            "waxs": [n for n in vars_all if n.startswith("pil900KW_")],
            "scan_axes": [
                n for n in vars_all
                if n in {"waxs_arc", "waxs_bsx", "waxs_bsy"}
            ],
        },
    }


# ---------------------------------------------------------------------------
# Top-level loader class
# ---------------------------------------------------------------------------

class TiledSMISWAXSLoader:
    """
    Tiled-based loader for the SMI WAXS + SAXS instrument.

    Mirrors the attribute/return contract expected by PyHyperScattering
    (PFGeneralIntegrator with geomethod='template_xr').

    Parameters
    ----------
    tiled_uri : str
        Tiled server base URI.
    catalog : str
        Slash-separated catalog path, e.g. ``"smi/migration"``.
    energy_kev : float | None
        Override photon energy (keV).  Falls back to run metadata.
    """

    md_loading_is_quick = True

    def __init__(
        self,
        tiled_uri: str = DEFAULT_TILED_URI,
        catalog: str = DEFAULT_CATALOG,
        energy_kev: float | None = None,
        api_key: str | None = None,
    ) -> None:
        self.tiled_uri = tiled_uri
        self.catalog = catalog
        self.energy_kev = energy_kev
        self.api_key = api_key
        self._root_client = None
        self._catalog_client = None

    # ------------------------------------------------------------------
    # Authentication helpers
    # ------------------------------------------------------------------
    def _get_root_client(self) -> Any:
        """Return the cached root tiled client, creating it on first use."""
        if self._root_client is None:
            from tiled.client import from_uri
            kwargs: dict[str, Any] = {}
            if self.api_key is not None:
                kwargs["api_key"] = self.api_key
            self._root_client = from_uri(self.tiled_uri, **kwargs)
        return self._root_client

    def login(self, **kwargs: Any) -> Any:
        """Interactively log in to the tiled server.

        Equivalent to ``tiled.client.from_uri(uri).login()``.  Any keyword
        arguments are forwarded to the underlying tiled client's ``login``
        method (e.g. ``provider=...``).  After a successful login the
        catalog client is invalidated so the next access uses the
        authenticated session.
        """
        client = self._get_root_client()
        result = client.login(**kwargs)
        # Force re-resolution of the catalog through the now-authenticated
        # root client so subsequent reads carry the auth token.
        self._catalog_client = None
        return result

    def logout(self) -> None:
        """Log out of the tiled server and clear cached clients."""
        if self._root_client is not None:
            try:
                self._root_client.logout()
            finally:
                self._root_client = None
        self._catalog_client = None

    def _get_catalog(self) -> Any:
        if self._catalog_client is None:
            # Walk the slash-separated catalog path from the root client so
            # the same authenticated session is reused for both login and
            # data access.
            node = self._get_root_client()
            for part in self.catalog.split("/"):
                if not part:
                    continue
                node = node[part]
            self._catalog_client = node
        return self._catalog_client

    def _get_run(self, uid: str) -> Any:
        return self._get_catalog()[uid]

    def peekAtMd(self, uid: str, detector: str = "saxs") -> dict[str, Any]:
        """Return geometry metadata dict without loading images."""
        run = self._get_run(uid)
        if detector == "saxs":
            geo = resolve_saxs_geometry(run, energy_kev=self.energy_kev)
            return {
                "energy_kev":         geo.energy_ev / 1000.0,
                "dist_m":             geo.dist_m,
                "beam_center_row_px": geo.beam_center_row_px,
                "beam_center_col_px": geo.beam_center_col_px,
                "active_beamstop":    geo.active_beamstop,
            }
        geo = resolve_waxs_geometry(run, energy_kev=self.energy_kev)
        return {
            "energy_kev":         geo.energy_ev / 1000.0,
            "dist_m":             geo.dist_m,
            "beam_center_row_px": geo.beam_center_row_px,
            "beam_center_col_px": geo.beam_center_col_px,
            "n_panels":           len(geo.panels),
        }

    def loadSingleImage(
        self,
        uid: str,
        detector: str = "saxs",
        geo_overrides: dict[str, Any] | None = None,
        extra_attrs: dict[str, Any] | None = None,
    ) -> xr.DataArray | None:
        """
        Load raw images for one run.

        Parameters
        ----------
        uid : str
            Tiled run UID.
        detector : {'saxs', 'waxs'}
            Which detector to load.
        geo_overrides : dict | None
            Override specific geometry parameters.
        extra_attrs : dict | None
            Extra attrs to attach to the returned DataArray.

        Returns
        -------
        xr.DataArray or None
            None if the requested detector is not present in the run.
            SAXS: dims (pix_y, pix_x) or (frame, pix_y, pix_x)
            WAXS: dims (waxs_arc, pix_y, pix_x)
        """
        run = self._get_run(uid)
        overrides = dict(geo_overrides or {})

        if detector == "saxs":
            if not _has_primary_field(run, SAXS_IMAGE_FIELD):
                return None
            geo = resolve_saxs_geometry(
                run, energy_kev=self.energy_kev, **overrides
            )
            return load_saxs_raw(run, geo, extra_attrs=extra_attrs)

        if detector == "waxs":
            if not _has_primary_field(run, WAXS_IMAGE_FIELD):
                return None
            geo = resolve_waxs_geometry(
                run, energy_kev=self.energy_kev, **overrides
            )
            return load_waxs_raw(run, geo, extra_attrs=extra_attrs)

        raise ValueError(
            f"Unknown detector '{detector}'. Expected 'saxs' or 'waxs'."
        )

    def loadRun(
        self,
        uid: str,
        geo_overrides: dict[str, Any] | None = None,
        extra_attrs: dict[str, Any] | None = None,
    ) -> dict[str, xr.DataArray]:
        """
        Load both SAXS and WAXS raw images for one run.

        Returns
        -------
        dict with keys ``'saxs'`` and ``'waxs'``, each an xr.DataArray.
        """
        return {
            "saxs": self.loadSingleImage(
                uid, "saxs", geo_overrides=geo_overrides, extra_attrs=extra_attrs,
            ),
            "waxs": self.loadSingleImage(
                uid, "waxs", geo_overrides=geo_overrides, extra_attrs=extra_attrs,
            ),
        }

    # ------------------------------------------------------------------
    # Tiled catalog browsing convenience
    # ------------------------------------------------------------------

    def searchCatalog(
        self,
        sample: str | None = None,
        plan: str | None = None,
        scan_id: int | None = None,
        cycle: str | None = None,
        proposal: str | None = None,
        user: str | None = None,
        institution: str | None = None,
        detector: str | None = None,
        since: str | None = None,
        until: str | None = None,
        limit: int | None = None,
        outputType: str = "default",
        **kwargs: Any,
    ):
        """Search the SMI Tiled catalog and return a results table.

        Modeled on :meth:`SST1RSoXSDB.searchCatalog` so that browsers and
        notebooks have a consistent API across NSLS-II beamlines.  Each
        keyword argument is mapped to a databroker query against the
        run-start metadata; only the keywords with a non-``None`` value
        contribute to the search.

        Parameters
        ----------
        sample, plan, user, institution, cycle : str | None
            Case-insensitive substring (regex) matches against the
            corresponding ``start`` keys.
        scan_id, proposal : int | None
            Exact numeric matches.
        detector : {'saxs', 'waxs'} | None
            If given, restrict to runs whose ``start.detectors`` field
            contains the SMI image-field substring for that detector
            (``"pil2M"`` or ``"pil900KW"``).
        since, until : str | None
            ISO-8601 timestamps, forwarded to
            :meth:`tiled.client.Catalog.search` via the
            ``TimeRange`` query.
        limit : int | None
            Cap the number of result rows returned (avoids pulling
            thousands of metadata blobs over the network).
        outputType : {'default', 'scans', 'all'}
            ``'scans'`` returns a 1-column DataFrame of scan IDs;
            ``'default'`` returns the columns
            ``[scan_id, start_time, sample_name, plan_name, detectors,
            num_points, uid]``; ``'all'`` adds ``cycle, user_name,
            institution, proposal_id``.
        **kwargs
            Additional ``key=value`` pairs forwarded as
            case-insensitive regex matches against ``start[key]``.

        Returns
        -------
        pandas.DataFrame
            Empty DataFrame if no results.

        Notes
        -----
        Requires ``tiled`` to be installed.  All network access is lazy:
        the catalog is not contacted until this method is called.
        """
        import pandas as pd

        cat = self._get_catalog()

        try:
            from tiled.queries import Key, Regex, TimeRange  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dep
            raise ImportError(
                "searchCatalog requires `tiled.queries` (install `tiled[client]`)."
            ) from exc

        def _regex_search(node, field: str, value: str):
            return node.search(Regex(field, f"(?i){value}"))

        def _detector_substring(d: str) -> str:
            d = d.lower()
            if d == "saxs":
                return "pil2M"
            if d == "waxs":
                return "pil900KW"
            raise ValueError(f"detector must be 'saxs' or 'waxs', got {d!r}")

        node = cat
        if sample is not None:
            node = _regex_search(node, "sample_name", str(sample))
        if plan is not None:
            node = _regex_search(node, "plan_name", str(plan))
        if user is not None:
            node = _regex_search(node, "user_name", str(user))
        if institution is not None:
            node = _regex_search(node, "institution", str(institution))
        if cycle is not None:
            node = _regex_search(node, "cycle", str(cycle))
        if scan_id is not None:
            node = node.search(Key("scan_id") == int(scan_id))
        if proposal is not None:
            node = node.search(Key("proposal_id") == int(proposal))
        if detector is not None:
            node = _regex_search(node, "detectors", _detector_substring(detector))
        if since is not None or until is not None:
            node = node.search(TimeRange(since=since, until=until))
        for key, value in kwargs.items():
            if value is None:
                continue
            if isinstance(value, (int, float)):
                node = node.search(Key(key) == value)
            else:
                node = _regex_search(node, key, str(value))

        rows: list[dict[str, Any]] = []
        for i, (uid, run) in enumerate(node.items()):
            if limit is not None and i >= int(limit):
                break
            try:
                start = dict(run.metadata.get("start", {}))
            except Exception:
                start = {}
            try:
                stop = dict(run.metadata.get("stop", {}) or {})
            except Exception:
                stop = {}
            num_points = None
            try:
                num_points = stop.get("num_events", {}).get("primary")
            except Exception:
                pass
            row = {
                "scan_id": start.get("scan_id"),
                "start_time": start.get("time"),
                "sample_name": start.get("sample_name"),
                "plan_name": start.get("plan_name"),
                "detectors": start.get("detectors"),
                "num_points": num_points,
                "uid": uid,
            }
            if outputType == "all":
                row.update({
                    "cycle": start.get("cycle"),
                    "user_name": start.get("user_name"),
                    "institution": start.get("institution"),
                    "proposal_id": start.get("proposal_id"),
                })
            rows.append(row)

        df = pd.DataFrame(rows)
        if outputType == "scans" and not df.empty:
            df = df[["scan_id"]].copy()
        return df

    def browseCatalog(self, **kwargs: Any):
        """Interactive catalog browser (uses :class:`ipyaggrid.Grid`).

        Thin wrapper around :meth:`searchCatalog`; same kwargs.  Returns
        an ``ipyaggrid.Grid`` widget suitable for Jupyter / JupyterHub.
        """
        df = self.searchCatalog(**kwargs)
        try:
            from ipyaggrid import Grid  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dep
            raise ImportError(
                "browseCatalog requires `ipyaggrid` (pip install ipyaggrid)."
            ) from exc
        return Grid(
            grid_data=df,
            grid_options={
                "columnDefs": [{"field": c} for c in df.columns],
                "enableSorting": True,
                "enableFilter": True,
                "enableColResize": True,
            },
        )

    def summarizeRun(self, uid: str) -> dict[str, Any]:
        """Return a small dict of headline metadata for one uid.

        Useful for "what is this scan?" lookups in browser tooltips
        without paying for a full primary-stream read.
        """
        run = self._get_run(uid)
        try:
            start = dict(run.metadata.get("start", {}))
        except Exception:
            start = {}
        detectors = start.get("detectors") or []
        detector_kinds: list[str] = []
        from PyHyperScattering.smi_defaults import classify_detector_field
        for d in detectors:
            kind = classify_detector_field(d)
            if kind and kind not in detector_kinds:
                detector_kinds.append(kind)
        return {
            "uid": uid,
            "scan_id": start.get("scan_id"),
            "sample_name": start.get("sample_name"),
            "plan_name": start.get("plan_name"),
            "detectors": detectors,
            "detector_kinds": detector_kinds,
            "start_time": start.get("time"),
            "num_points": start.get("num_points"),
        }
