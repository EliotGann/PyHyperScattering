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

from dataclasses import dataclass
from typing import Any

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
    try:
        return run["baseline"].read()
    except (KeyError, Exception):
        return None


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


def _energy_to_wavelength_m(energy_ev: float) -> float:
    return _HBAR_C_EV_M / float(energy_ev)


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
    """Resolve full SAXS geometry from a tiled run."""
    baseline = _read_baseline(run)
    conf = _primary_conf(run, "pil2M")
    start = run.metadata.get("start", {})

    energy_kev = energy_kev or start.get("energy") or DEFAULT_ENERGY_KEV
    energy_ev = float(energy_kev) * 1000.0

    beam_row = (
        overrides.get("beam_center_row_px")
        or _dataset_scalar(baseline, "pil2M_beam_center_y_px")
        or _conf_scalar(conf, "pil2M_beam_center_y_px")
        or _SAXS_DEFAULT_BEAM_ROW_PX
    )
    beam_col = (
        overrides.get("beam_center_col_px")
        or _dataset_scalar(baseline, "pil2M_beam_center_x_px")
        or _conf_scalar(conf, "pil2M_beam_center_x_px")
        or _SAXS_DEFAULT_BEAM_COL_PX
    )
    dist_mm = (
        overrides.get("sample_distance_mm")
        or _dataset_scalar(baseline, "pil2M_motor_z_user_setpoint")
        or _dataset_scalar(baseline, "pil2M_motor_z")
        or _conf_scalar(conf, "pil2M_sdd_mm")
        or _SAXS_DEFAULT_DISTANCE_MM
    )
    active_bs = (
        overrides.get("active_beamstop")
        or _dataset_scalar(baseline, "pil2M_active_beamstop")
        or _conf_scalar(conf, "pil2M_active_beamstop")
        or "rod"
    )

    bs_pos = {
        "rod": {
            "x": (
                _dataset_scalar(baseline, "saxs_beamstop_x_rod_user_setpoint")
                or _dataset_scalar(baseline, "saxs_beamstop_x_rod")
            ),
            "y": (
                _dataset_scalar(baseline, "saxs_beamstop_y_rod_user_setpoint")
                or _dataset_scalar(baseline, "saxs_beamstop_y_rod")
            ),
        },
        "pin": {
            "x": (
                _dataset_scalar(baseline, "saxs_beamstop_x_pin_user_setpoint")
                or _dataset_scalar(baseline, "saxs_beamstop_x_pin")
            ),
            "y": (
                _dataset_scalar(baseline, "saxs_beamstop_y_pin_user_setpoint")
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
    """Resolve full WAXS geometry from a tiled run."""
    baseline = _read_baseline(run)
    conf = _primary_conf(run, "pil900KW")
    start = run.metadata.get("start", {})

    energy_kev = energy_kev or start.get("energy") or DEFAULT_ENERGY_KEV
    energy_ev = float(energy_kev) * 1000.0

    dist_mm = (
        overrides.get("sample_distance_mm")
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

def _read_primary_field(run: Any, field: str) -> np.ndarray:
    primary = run["primary"].read()
    if field not in primary:
        raise KeyError(
            f"Field '{field}' not found in primary stream. "
            f"Available: {list(primary.data_vars)}"
        )
    return np.asarray(primary[field].values)


def _read_scan_axis(run: Any, field: str) -> np.ndarray | None:
    try:
        primary = run["primary"].read()
        if field in primary:
            return np.asarray(primary[field].values, dtype=float)
    except Exception:
        pass
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
        # Run identity
        "uid":         start.get("uid", ""),
        "scan_id":     start.get("scan_id"),
        "sample_name": start.get("sample_name", ""),
    }
    if extra_attrs:
        attrs.update(extra_attrs)

    if images.ndim == 2:
        return xr.DataArray(images, dims=["pix_y", "pix_x"], attrs=attrs)

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

    arc_angles = _read_scan_axis(run, WAXS_ARC_FIELD)
    bsx_values = _read_scan_axis(run, WAXS_BSX_FIELD)

    if images.ndim == 2:
        images = images[np.newaxis, :, :]
    n_frames = images.shape[0]

    if arc_angles is None or arc_angles.shape[0] != n_frames:
        arc_angles = np.zeros(n_frames, dtype=float)

    if bsx_values is None or bsx_values.shape[0] != n_frames:
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

def infer_detectors_and_steps(run: Any, primary: xr.Dataset) -> dict[str, Any]:
    """Inspect a tiled run to determine detectors, scan axes, and frame count."""
    start = run.metadata.get("start", {})
    vars_all = sorted(map(str, primary.data_vars.keys()))

    detector_prefixes = sorted(
        {name.split("_")[0] for name in vars_all if "_" in name}
    )

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
    ) -> None:
        self.tiled_uri = tiled_uri
        self.catalog = catalog
        self.energy_kev = energy_kev
        self._catalog_client = None

    def _get_catalog(self) -> Any:
        if self._catalog_client is None:
            from tiled.client import from_uri
            cat = from_uri(self.tiled_uri)
            for part in self.catalog.split("/"):
                cat = cat[part]
            self._catalog_client = cat
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
    ) -> xr.DataArray:
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
        xr.DataArray
            SAXS: dims (pix_y, pix_x) or (frame, pix_y, pix_x)
            WAXS: dims (waxs_arc, pix_y, pix_x)
        """
        run = self._get_run(uid)
        overrides = dict(geo_overrides or {})

        if detector == "saxs":
            geo = resolve_saxs_geometry(
                run, energy_kev=self.energy_kev, **overrides
            )
            return load_saxs_raw(run, geo, extra_attrs=extra_attrs)

        if detector == "waxs":
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
