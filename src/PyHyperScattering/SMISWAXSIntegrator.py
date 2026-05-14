"""
SMISWAXSIntegrator
===================
pyFAI-backed azimuthal integration for the SMI WAXS+SAXS instrument,
designed to work with :class:`SMISWAXSLoader.TiledSMISWAXSLoader`.

The module is self-contained: all geometry (multi-panel arc detector,
single-panel SAXS), masking, histogram binning, and merging utilities
are included so the only external runtime dependencies are numpy, xarray,
and (optionally) pyFAI.

Architecture
------------
1.  Raw images arrive as xr.DataArray from the loader with calibration in attrs.
2.  SAXS uses a flat-panel pixel-space q-map (matching SinglePanelSAXSDetector).
3.  WAXS uses per-frame q-maps from :class:`MultiPanelArcDetector` which
    computes exact 3D pixel positions for the folded 3-panel geometry.
4.  Both detectors bin into (q, chi) grids via histogram2d.
5.  Merged I(q, chi) and I(q) are produced via weighted overlap merge.

Key classes / functions
-----------------------
- ``MultiPanelArcDetector``  – WAXS 3-panel geometry model
- ``integrate_saxs``         – SAXS integration from raw DataArray
- ``integrate_waxs``         – WAXS integration from raw DataArray
- ``reduce_smi_combined``    – Full SAXS + WAXS pipeline returning merged result
"""
from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence, Tuple

import numpy as np
import xarray as xr

from PyHyperScattering.SMISWAXSLoader import (
    SAXSGeometry,
    WAXSGeometry,
    resolve_saxs_geometry,
)


# ===================================================================
# Geometry cache – persists across calls within the same Python process
# ===================================================================
#
# GUI integration
# ---------------
# The geometry cache stores precomputed per-pixel q-maps so that repeated
# reductions with the same detector geometry (energy, distance, beam center,
# panel offsets, masks, etc.) skip the expensive trigonometry.
#
# Usage from a GUI or batch script:
#
#     from PyHyperScattering.SMISWAXSIntegrator import (
#         reduce_smi_combined,
#         clear_geometry_cache,
#         geometry_cache_info,
#     )
#
#     # Process many scans — geometry is computed once, then reused:
#     for uid in uid_list:
#         result = reduce_smi_combined(uid, cache_geometry=True, ...)
#
#     # When the user changes calibration (beam center, distance, energy,
#     # panel offsets, etc.), clear the stale cache:
#     clear_geometry_cache()
#
#     # Or inspect current cache size:
#     info = geometry_cache_info()
#     print(f"Cache holds {info['waxs_entries']} WAXS geometries, "
#           f"~{info['estimated_mb']:.1f} MB")
#
# The cache is keyed on the full set of geometry parameters, so if you
# change *any* calibration value the old entry simply won't match and a
# new one will be computed (no stale-data risk).  Call
# ``clear_geometry_cache()`` only to free memory.

_WAXS_GEOMETRY_CACHE: dict[tuple, dict[float, tuple]] = {}
_SAXS_GEOMETRY_CACHE: dict[tuple, tuple] = {}


def clear_geometry_cache() -> None:
    """Clear the module-level geometry cache to free memory.

    Call this when you want to reclaim memory in a long-running process
    (e.g. a GUI).  It is *not* necessary to call this when calibration
    parameters change — the cache is keyed on the full parameter set, so
    a changed parameter simply produces a new cache entry.

    Example (in a GUI callback when user clicks "Reset Calibration")::

        from PyHyperScattering.SMISWAXSIntegrator import clear_geometry_cache
        clear_geometry_cache()
    """
    _WAXS_GEOMETRY_CACHE.clear()
    _SAXS_GEOMETRY_CACHE.clear()


def geometry_cache_info() -> dict[str, Any]:
    """Return a summary of the current geometry cache state.

    Returns
    -------
    dict with keys:
        waxs_entries : int – number of distinct WAXS calibration keys cached
        waxs_angles_total : int – total number of cached arc-angle q-maps
        saxs_entries : int – number of distinct SAXS geometry keys cached
        estimated_mb : float – rough memory estimate in megabytes
    """
    waxs_angles = sum(len(v) for v in _WAXS_GEOMETRY_CACHE.values())
    # Each cached angle holds ~5 arrays of shape (ny, nx); estimate 619×487
    # ≈ 300k pixels × 8 bytes × 5 arrays ≈ 12 MB per angle
    est_per_angle_mb = 12.0
    # Each SAXS entry holds ~5 arrays of shape (ny, nx); estimate 1475×1679
    # ≈ 2.5M pixels × 8 bytes × 5 ≈ 100 MB per entry
    est_per_saxs_mb = 100.0
    estimated_mb = (
        waxs_angles * est_per_angle_mb
        + len(_SAXS_GEOMETRY_CACHE) * est_per_saxs_mb
    )
    return {
        "waxs_entries": len(_WAXS_GEOMETRY_CACHE),
        "waxs_angles_total": waxs_angles,
        "saxs_entries": len(_SAXS_GEOMETRY_CACHE),
        "estimated_mb": estimated_mb,
    }


def _waxs_cache_key(
    cal: "WAXSCalibration",
    image_shape: tuple[int, int],
    flip_horizontal: bool,
    qx_shift_nm: float,
    qy_shift_nm: float,
) -> tuple:
    """Build a hashable key from all parameters that affect WAXS q-maps."""
    return (
        cal.energy_kev,
        cal.sample_distance_mm,
        cal.pixel_size_mm,
        cal.beam_center_row,
        cal.beam_center_col,
        tuple(cal.panel_col_ranges),
        tuple(cal.panel_offsets_deg),
        tuple(cal.panel_row_shifts),
        tuple(cal.panel_col_shifts),
        tuple(cal.panel_delta_deg),
        cal.theta_zero_deg,
        cal.sample_offset_x_mm,
        cal.sample_offset_z_mm,
        cal.beam_col_per_arc_deg,
        cal.q_horizontal_sign,
        cal.q_vertical_sign,
        cal.rotation_k,
        image_shape,
        flip_horizontal,
        round(qx_shift_nm, 10),
        round(qy_shift_nm, 10),
    )


def _saxs_cache_key(
    dist_m: float,
    poni1_m: float,
    poni2_m: float,
    pixel1_m: float,
    pixel2_m: float,
    wavelength_m: float,
    image_shape: tuple[int, int],
) -> tuple:
    """Build a hashable key from all parameters that affect SAXS q-maps."""
    return (
        round(dist_m, 12),
        round(poni1_m, 12),
        round(poni2_m, 12),
        round(pixel1_m, 12),
        round(pixel2_m, 12),
        round(wavelength_m, 15),
        image_shape,
    )


# ===================================================================
# WAXS detector geometry – ported from waxs_reduce.py
# ===================================================================

def wavelength_nm_from_energy_kev(energy_kev: float) -> float:
    return 1.23984198 / float(energy_kev)


@dataclass(frozen=True)
class PanelSpec:
    image_cols: slice
    offset_deg: float
    row_shift_px: float = 0.0
    col_shift_px: float = 0.0
    delta_deg: float = 0.0


@dataclass
class WAXSCalibration:
    energy_kev: float = 16.1
    sample_distance_mm: float = 270.0
    pixel_size_mm: float = 0.172
    beam_center_row: float = 217.0
    beam_center_col: float = 319.0
    panel_col_ranges: Tuple = ((0, 206), (206, 413), (413, 619))
    panel_offsets_deg: Tuple = (-7.0, 0.0, 7.0)
    panel_row_shifts: Tuple = (0.0, 0.0, 0.0)
    panel_col_shifts: Tuple = (0.0, 0.0, 0.0)
    panel_delta_deg: Tuple = (0.0, 0.0, 0.0)
    theta_zero_deg: float = 0.0
    sample_offset_x_mm: float = 0.0
    sample_offset_z_mm: float = 0.0
    beam_col_per_arc_deg: float = 0.0
    q_horizontal_sign: float = -1.0
    q_vertical_sign: float = -1.0
    rotation_k: int = 3

    @property
    def wavelength_nm(self) -> float:
        return wavelength_nm_from_energy_kev(self.energy_kev)

    def make_panel_specs(self) -> list[PanelSpec]:
        specs = []
        for i, ((c0, c1), off) in enumerate(
            zip(self.panel_col_ranges, self.panel_offsets_deg)
        ):
            specs.append(
                PanelSpec(
                    image_cols=slice(int(c0), int(c1)),
                    offset_deg=float(off),
                    row_shift_px=float(self.panel_row_shifts[i]),
                    col_shift_px=float(self.panel_col_shifts[i]),
                    delta_deg=float(self.panel_delta_deg[i]),
                )
            )
        return specs

    def beam_center_at_angle(self, theta_deg: float) -> Tuple[float, float]:
        row = float(self.beam_center_row)
        col = float(self.beam_center_col)
        if self.sample_offset_x_mm != 0 or self.sample_offset_z_mm != 0:
            th = np.deg2rad(theta_deg + self.theta_zero_deg)
            dx_mm = (self.sample_offset_x_mm * (np.cos(th) - 1.0)
                     - self.sample_offset_z_mm * np.sin(th))
            col += dx_mm / self.pixel_size_mm
        if self.beam_col_per_arc_deg != 0:
            col += self.beam_col_per_arc_deg * theta_deg
        return (row, col)


# Default calibration matching legacy waxs_reduce._DEFAULT_CAL
_DEFAULT_CAL = dict(
    energy_kev=16.1,
    sample_distance_mm=274,
    beam_center_row=217.0,
    beam_center_col=319.0,
    panel_offsets_deg=(-7.0, 0.0, 7.0),
    theta_zero_deg=0,
    sample_offset_z_mm=2.0,
)


class MultiPanelArcDetector:
    """3-panel folded arc WAXS detector geometry model."""

    def __init__(
        self,
        image_shape: Tuple[int, int],
        panel_specs: Sequence[PanelSpec],
        wavelength_nm: float,
        pixel_size_mm: float = 0.172,
        sample_distance_mm: float = 300.0,
        beam_center_px: Tuple[float, float] = (0.0, 0.0),
        theta_zero_deg: float = 0.0,
        sample_offset_x_mm: float = 0.0,
        sample_offset_z_mm: float = 0.0,
    ) -> None:
        self.ny, self.nx = image_shape
        self.panel_specs = list(panel_specs)
        self.wavelength_nm = float(wavelength_nm)
        self.pixel_size_mm = float(pixel_size_mm)
        self.sample_distance_mm = float(sample_distance_mm)
        self.beam_center_row_px = float(beam_center_px[0])
        self.beam_center_col_px = float(beam_center_px[1])
        self.theta_zero_deg = float(theta_zero_deg)
        self.sample_offset_x_mm = float(sample_offset_x_mm)
        self.sample_offset_z_mm = float(sample_offset_z_mm)

    def qmap(self, theta_deg: float) -> xr.Dataset:
        """Compute per-pixel q-vectors and solid angle for a given arc angle."""
        p_mm = self.pixel_size_mm
        R    = self.sample_distance_mm
        n_panels = len(self.panel_specs)
        rows = np.arange(self.ny, dtype=float)

        panel_c0s, panel_c1s, panel_mids, alphas_det = [], [], [], []
        for ps in self.panel_specs:
            c0 = 0 if ps.image_cols.start is None else ps.image_cols.start
            c1 = self.nx if ps.image_cols.stop is None else ps.image_cols.stop
            panel_c0s.append(c0)
            panel_c1s.append(c1)
            panel_mids.append(0.5 * (c0 + c1 - 1) + ps.col_shift_px)
            alphas_det.append(np.deg2rad(ps.offset_deg + ps.delta_deg))

        ref_idx = min(
            range(n_panels),
            key=lambda i: abs(self.panel_specs[i].offset_deg),
        )
        bc_u = -(self.beam_center_col_px - panel_mids[ref_idx]) * p_mm
        alpha_r = alphas_det[ref_idx]
        centers_x = [None] * n_panels
        centers_z = [None] * n_panels
        centers_x[ref_idx] = -bc_u * np.cos(alpha_r)
        centers_z[ref_idx] = R - bc_u * np.sin(alpha_r)

        for i in range(ref_idx - 1, -1, -1):
            fold = panel_c1s[i] - 0.5
            u1 = -(fold - panel_mids[i + 1]) * p_mm
            u0 = -(fold - panel_mids[i]) * p_mm
            centers_x[i] = (
                centers_x[i + 1]
                + u1 * np.cos(alphas_det[i + 1])
                - u0 * np.cos(alphas_det[i])
            )
            centers_z[i] = (
                centers_z[i + 1]
                + u1 * np.sin(alphas_det[i + 1])
                - u0 * np.sin(alphas_det[i])
            )
        for i in range(ref_idx + 1, n_panels):
            fold = panel_c0s[i] - 0.5
            u1 = -(fold - panel_mids[i - 1]) * p_mm
            u0 = -(fold - panel_mids[i]) * p_mm
            centers_x[i] = (
                centers_x[i - 1]
                + u1 * np.cos(alphas_det[i - 1])
                - u0 * np.cos(alphas_det[i])
            )
            centers_z[i] = (
                centers_z[i - 1]
                + u1 * np.sin(alphas_det[i - 1])
                - u0 * np.sin(alphas_det[i])
            )

        theta_rad = np.deg2rad(float(theta_deg) + self.theta_zero_deg)
        cos_th, sin_th = np.cos(theta_rad), np.sin(theta_rad)

        px_mm = np.full((self.ny, self.nx), np.nan)
        py_mm = np.full((self.ny, self.nx), np.nan)
        pz_mm = np.full((self.ny, self.nx), np.nan)

        for idx, ps in enumerate(self.panel_specs):
            c0, c1 = panel_c0s[idx], panel_c1s[idx]
            cols_p = np.arange(c0, c1, dtype=float)
            _rr, _cc = np.meshgrid(rows, cols_p, indexing="ij")
            u = -(_cc - panel_mids[idx]) * p_mm
            y_det = -(
                _rr - (self.beam_center_row_px + ps.row_shift_px)
            ) * p_mm
            alpha = alphas_det[idx]
            x_det = centers_x[idx] + u * np.cos(alpha)
            z_det = centers_z[idx] + u * np.sin(alpha)
            x_lab = x_det * cos_th - z_det * sin_th
            z_lab = x_det * sin_th + z_det * cos_th
            px_mm[:, c0:c1] = x_lab - self.sample_offset_x_mm
            py_mm[:, c0:c1] = y_det
            pz_mm[:, c0:c1] = z_lab - self.sample_offset_z_mm

        r = np.sqrt(px_mm**2 + py_mm**2 + pz_mm**2)
        k = 2.0 * np.pi / self.wavelength_nm
        with np.errstate(invalid="ignore", divide="ignore"):
            qx = k * px_mm / r
            qy = k * py_mm / r
            qz = k * (pz_mm / r - 1.0)
        qabs = np.sqrt(qx**2 + qy**2 + qz**2)

        pixel_area_mm2 = p_mm * p_mm
        with np.errstate(invalid="ignore", divide="ignore"):
            solid_angle = pixel_area_mm2 * np.maximum(pz_mm, 0.0) / (r**3)

        return xr.Dataset(
            {
                "qx": (("row", "col"), qx),
                "qy": (("row", "col"), qy),
                "qz": (("row", "col"), qz),
                "qabs": (("row", "col"), qabs),
                "solid_angle": (("row", "col"), solid_angle),
            },
            coords={
                "row": np.arange(self.ny),
                "col": np.arange(self.nx),
            },
        )


def rotate_image_and_mask(
    image: np.ndarray,
    mask: np.ndarray | None = None,
    k: int = 3,
) -> tuple[np.ndarray, np.ndarray | None]:
    img_rot = np.fliplr(np.rot90(np.asarray(image), k=k))
    mask_rot = np.fliplr(np.rot90(np.asarray(mask), k=k)) if mask is not None else None
    return img_rot, mask_rot


def dezinger(
    image: np.ndarray,
    kernel_size: int = 5,
    threshold: float = 5.0,
) -> np.ndarray:
    """Detect hot/dead pixels via median-filter outlier rejection.

    Compares each pixel to a local median. Pixels deviating by more than
    ``threshold`` × σ (estimated from the MAD) are flagged.

    Parameters
    ----------
    image : ndarray
        2-D detector image.
    kernel_size : int
        Side length of the square median-filter kernel (default 5).
    threshold : float
        Number of σ above the local median to flag as bad (default 5).

    Returns
    -------
    ndarray[bool]
        True for *valid* pixels, False for outliers (matches mask convention).
    """
    from scipy.ndimage import median_filter

    img = np.asarray(image, dtype=float)
    med = median_filter(img, size=kernel_size)
    diff = img - med
    finite_diff = diff[np.isfinite(diff)]
    if finite_diff.size == 0:
        return np.ones(img.shape, dtype=bool)
    mad = np.median(np.abs(finite_diff))
    sigma_est = 1.4826 * mad  # MAD → Gaussian σ conversion
    if sigma_est < 1e-12:
        sigma_est = 1.0  # avoid division by zero for uniform images
    return ~(np.abs(diff) > threshold * sigma_est)


# ===================================================================
# Mask utilities – ported from waxs_reduce.py / saxs_reduce.py
# ===================================================================

def polygons_to_mask(
    shape: tuple[int, int],
    polygons: list,
) -> np.ndarray:
    """Build a boolean mask (True = valid) from a list of polygon regions."""
    from skimage.draw import polygon as skpoly

    mask = np.ones(shape, dtype=bool)
    ny, nx = shape
    for poly in polygons:
        if not poly:
            continue
        cols = np.array([p[0] for p in poly], dtype=float)
        rows = np.array([p[1] for p in poly], dtype=float)
        rr, cc = skpoly(rows, cols, shape=shape)
        mask[rr, cc] = False
    return mask


def shift_polygon(
    polygon: list[list[float]],
    dx_px: float = 0.0,
    dy_px: float = 0.0,
) -> list[list[float]]:
    return [[col + dx_px, row + dy_px] for col, row in polygon]


def make_mask_for_angle(
    image_shape_raw: tuple[int, int],
    static_regions: dict,
    beamstop_region: list,
    waxs_bsx: float,
    waxs_bsx_ref: float,
    pixel_size_mm: float = 0.172,
    rotation_k: int = 3,
    include_beamstop: bool = True,
) -> np.ndarray:
    """Build a per-angle WAXS mask accounting for beamstop motor position."""
    polys = list(static_regions.values())
    if include_beamstop and beamstop_region:
        bs_shift_mm = waxs_bsx - waxs_bsx_ref
        bs_shift_px = (bs_shift_mm / pixel_size_mm) * 1.088  # empirical fudge factor
        polys.append(shift_polygon(beamstop_region, dx_px=0.0, dy_px=bs_shift_px))
    raw_mask = polygons_to_mask(image_shape_raw, polys)
    mask_rot, _ = rotate_image_and_mask(raw_mask, k=rotation_k)
    return mask_rot


# ===================================================================
# SAXS mask builders
# ===================================================================

def make_saxs_mask_from_spec(
    image_shape: tuple[int, int],
    mask_path: str | Path,
    active_beamstop: str = "rod",
    beamstop_pos_mm: dict | None = None,
) -> np.ndarray:
    """Build a static SAXS mask (True = valid) from a JSON mask spec."""
    with open(mask_path) as f:
        mask_spec = json.load(f)

    static_regions = mask_spec.get("static_regions", {})
    beamstops_spec = mask_spec.get("beamstops", {})
    polys = list(static_regions.values())

    bs = beamstops_spec.get(active_beamstop, {})
    bs_ref = bs.get("reference_mm", {})
    bs_ref_x = bs_ref.get("x", 0.0)
    bs_ref_y = bs_ref.get("y", 0.0)
    bs_poly = bs.get("polygon")
    px_per_mm_map = bs.get("pixels_per_mm", {})
    px_per_mm_x = (
        px_per_mm_map.get("x", 1.0 / 0.172)
        if isinstance(px_per_mm_map, dict)
        else float(px_per_mm_map)
    )
    px_per_mm_y = (
        px_per_mm_map.get("y", 1.0 / 0.172)
        if isinstance(px_per_mm_map, dict)
        else float(px_per_mm_map)
    )

    if bs_poly is not None:
        if beamstop_pos_mm:
            cur = beamstop_pos_mm.get(active_beamstop) or {}
            cur_x = cur.get("x") or bs_ref_x
            cur_y = cur.get("y") or bs_ref_y
            dx = (cur_x - bs_ref_x) * px_per_mm_x
            dy = (cur_y - bs_ref_y) * px_per_mm_y
            polys.append(shift_polygon(bs_poly, dx_px=dx, dy_px=dy))
        else:
            polys.append(bs_poly)

    return polygons_to_mask(image_shape, polys)


def make_waxs_mask_callable(
    mask_path: str | Path,
    waxs_bsx_ref: float = 0.0,
    beamstop_max_abs_arc_deg: float | None = 15.0,
):
    """Return ``mask_fn(image_shape_raw, theta_deg, waxs_bsx) → bool mask``."""
    with open(mask_path) as f:
        mask_data = json.load(f)

    if "static_regions" in mask_data or "beamstops" in mask_data:
        static_regions = mask_data.get("static_regions", {})
        bs_entry = mask_data.get("beamstops", {}).get("beamstop", {})
        beamstop_region = (
            bs_entry.get("polygons", [[]])[0]
            if bs_entry.get("polygons")
            else []
        )
    else:
        beamstop_region = mask_data.get("beamstop", [])
        static_regions = {
            key: value
            for key, value in mask_data.items()
            if key != "beamstop"
        }

    def mask_fn(image_shape_raw, theta_deg, waxs_bsx):
        include_beamstop = (
            beamstop_max_abs_arc_deg is None
            or abs(float(theta_deg)) <= float(beamstop_max_abs_arc_deg)
        )
        return make_mask_for_angle(
            image_shape_raw=image_shape_raw,
            static_regions=static_regions,
            beamstop_region=beamstop_region,
            waxs_bsx=float(waxs_bsx),
            waxs_bsx_ref=float(waxs_bsx_ref),
            include_beamstop=include_beamstop,
        )

    return mask_fn


# ===================================================================
# Single-frame mask convenience (browser / notebook helper)
# ===================================================================

def _smi_run_field_at(run: Any, field: str, frame_idx: int) -> float:
    """Return ``run.primary[field][frame_idx]`` as a Python float.

    Tolerates several access patterns so it works against bluesky/tiled
    runs *and* dict-like fakes in tests::

        run["primary"]["data"][field]              # tiled, bluesky-tiled
        run["primary"][field]                      # legacy tiled
        run.primary[field]                         # attribute-style
        run["primary"].read()[field]               # full read fallback
        run[field]                                 # bare dict fallback (tests)
    """
    primary = None
    try:
        primary = run["primary"]
    except Exception:
        primary = getattr(run, "primary", None)

    candidates = []
    if primary is not None:
        # bluesky-tiled: primary/data/<field>
        try:
            candidates.append(primary["data"][field])
        except Exception:
            pass
        try:
            candidates.append(primary[field])
        except Exception:
            pass
        try:
            candidates.append(getattr(primary, field))
        except Exception:
            pass
    # Bare dict-like at top level (used by simple test fakes)
    try:
        candidates.append(run[field])
    except Exception:
        pass

    for node in candidates:
        if node is None:
            continue
        try:
            values = node.read() if hasattr(node, "read") else node
            arr = np.asarray(values).reshape(-1)
            if arr.size == 0:
                continue
            idx = int(frame_idx) if arr.size > 1 else 0
            return float(arr[idx])
        except Exception:
            continue

    # Fall back to a full primary.read() — slower, but always correct.
    if primary is not None:
        try:
            ds = primary.read()
            arr = np.asarray(ds[field].values).reshape(-1)
            idx = int(frame_idx) if arr.size > 1 else 0
            return float(arr[idx])
        except Exception:
            pass

    raise KeyError(field)


def _smi_run_raw_shape(run: Any, image_field: str) -> tuple[int, int]:
    """Return ``(rows, cols)`` of one frame without downloading the data."""
    primary = None
    try:
        primary = run["primary"]
    except Exception:
        primary = getattr(run, "primary", None)

    nodes = []
    if primary is not None:
        try:
            nodes.append(primary["data"][image_field])
        except Exception:
            pass
        try:
            nodes.append(primary[image_field])
        except Exception:
            pass
    try:
        nodes.append(run[image_field])
    except Exception:
        pass

    for node in nodes:
        try:
            shp = tuple(getattr(node, "shape", ()))
            if len(shp) >= 2:
                return (int(shp[-2]), int(shp[-1]))
        except Exception:
            continue
    raise KeyError(f"could not determine shape of {image_field!r} on run")


def mask_for_frame(
    run_or_uid: Any,
    frame_idx: int,
    detector: str,
    *,
    mask_path: str | Path | None = None,
    orient_for_display: bool = False,
    tiled_uri: str | None = None,
    catalog: str | None = None,
    raw_shape: tuple[int, int] | None = None,
    beamstop_max_abs_arc_deg: float | None = 15.0,
) -> np.ndarray:
    """Return the boolean validity mask (True = valid) for one frame.

    Thin wrapper over :func:`make_saxs_mask_from_spec` /
    :func:`make_waxs_mask_callable` that pulls the per-frame motor
    positions a browser/notebook would otherwise have to fetch by hand.

    Parameters
    ----------
    run_or_uid
        A bluesky/tiled run object, **or** a uid string.  If a string is
        given, ``tiled_uri`` and ``catalog`` are used to resolve it
        (defaults from :mod:`PyHyperScattering.smi_defaults`).
    frame_idx : int
        Frame index along the scan axis (e.g. ``waxs_arc``).  Ignored for
        SAXS in current SMI configuration but accepted for symmetry.
    detector : {'saxs', 'waxs'}
        Which detector's mask to build.
    mask_path : str | Path | None, optional
        Polygon-mask JSON path.  ``None`` selects the bundled default
        from :func:`smi_defaults.resolve_mask_path`.
    orient_for_display : bool, optional
        If True, return the mask already aligned with
        :func:`smi_defaults.orient_frame_for_display`.  For WAXS the
        underlying mask builder *already* returns a display-oriented
        array (it applies ``np.fliplr(np.rot90(..., k=3))`` internally),
        so no extra orientation pass is performed in that case.
    tiled_uri, catalog : str | None
        Used only when ``run_or_uid`` is a uid string.
    raw_shape : tuple[int, int] | None
        Override for the raw detector shape ``(rows, cols)``.  Useful for
        testing; normally read from the run's primary stream metadata.
    beamstop_max_abs_arc_deg : float | None
        Forwarded to :func:`make_waxs_mask_callable`.

    Returns
    -------
    np.ndarray[bool]
        Boolean mask, ``True`` where the pixel is valid for integration.

    Raises
    ------
    KeyError
        If WAXS is requested and ``waxs_arc`` / ``waxs_bsx`` cannot be
        located on the run's primary stream.
    ValueError
        If ``detector`` is not ``'saxs'`` or ``'waxs'``.
    """
    from PyHyperScattering.smi_defaults import (
        DEFAULT_TILED_URI as _DEFAULT_TILED_URI,
        DEFAULT_CATALOG as _DEFAULT_CATALOG,
        SAXS_IMAGE_FIELD as _SAXS_IMAGE_FIELD,
        WAXS_IMAGE_FIELD as _WAXS_IMAGE_FIELD,
        WAXS_ARC_FIELD as _WAXS_ARC_FIELD,
        WAXS_BSX_FIELD as _WAXS_BSX_FIELD,
        BSX_PER_ARC_DEG as _BSX_PER_ARC_DEG_PUBLIC,
        orient_frame_for_display as _orient_frame_for_display,
        resolve_mask_path as _resolve_mask_path,
    )

    det = str(detector).lower()
    if det not in ("saxs", "waxs"):
        raise ValueError(f"detector must be 'saxs' or 'waxs', got {detector!r}")

    # Resolve uid → run if necessary.
    if isinstance(run_or_uid, str):
        from tiled.client import from_uri  # heavy, lazy
        cat_path = catalog or _DEFAULT_CATALOG
        uri = tiled_uri or _DEFAULT_TILED_URI
        run = from_uri(uri)[cat_path][run_or_uid]
    else:
        run = run_or_uid

    image_field = _SAXS_IMAGE_FIELD if det == "saxs" else _WAXS_IMAGE_FIELD
    if raw_shape is None:
        raw_shape = _smi_run_raw_shape(run, image_field)
    raw_shape = (int(raw_shape[0]), int(raw_shape[1]))

    resolved_mask_path = _resolve_mask_path(mask_path, detector=det)

    if det == "saxs":
        # Pull the actual active beamstop + per-run motor positions from
        # the run's baseline / configuration so the dynamic mask reflects
        # this scan's geometry, not just the polygon file's reference
        # positions.  Falls back gracefully if the resolver fails (e.g.
        # mocked test runs without a baseline).
        active_bs = "rod"
        bs_pos: dict | None = None
        saxs_geo = None
        try:
            from PyHyperScattering.SMISWAXSLoader import resolve_saxs_geometry
            saxs_geo = resolve_saxs_geometry(run)
            active_bs = saxs_geo.active_beamstop or "rod"
            bs_pos = saxs_geo.beamstop_pos_mm
        except Exception:
            pass

        mask = make_saxs_mask_from_spec(
            image_shape=raw_shape,
            mask_path=resolved_mask_path,
            active_beamstop=active_bs,
            beamstop_pos_mm=bs_pos,
        )

        # AND in the per-frame WAXS-shadow occlusion (depends on
        # ``waxs_arc``).  This is the same shadow that
        # ``_integrate_saxs_batch`` applies during full reduction, so the
        # dynamic-mask overlay matches what the integrator actually sees.
        # Skipped silently if waxs_arc / beam_center are unavailable
        # (e.g. SAXS-only runs without the WAXS arc motor).
        try:
            waxs_arc = _smi_run_field_at(run, _WAXS_ARC_FIELD, frame_idx)
            beam_col = (
                saxs_geo.beam_center_col_px
                if saxs_geo is not None else None
            )
            if waxs_arc is not None and beam_col is not None:
                shadow = _make_waxs_shadow_mask(
                    raw_shape, [float(waxs_arc)], float(beam_col),
                )[0]
                mask = mask & shadow
        except Exception:
            pass

        if orient_for_display:
            mask = _orient_frame_for_display(mask, "saxs")
        return mask

    # WAXS: pull arc + bsx, derive bsx_ref via the SMI mechanical linkage.
    waxs_arc = _smi_run_field_at(run, _WAXS_ARC_FIELD, frame_idx)
    waxs_bsx = _smi_run_field_at(run, _WAXS_BSX_FIELD, frame_idx)
    waxs_bsx_ref = waxs_bsx - _BSX_PER_ARC_DEG_PUBLIC * waxs_arc

    mask_fn = make_waxs_mask_callable(
        resolved_mask_path,
        waxs_bsx_ref=waxs_bsx_ref,
        beamstop_max_abs_arc_deg=beamstop_max_abs_arc_deg,
    )
    # ``make_waxs_mask_callable`` returns a mask already in the display
    # orientation (rot90+fliplr applied inside ``make_mask_for_angle``).
    # Therefore for WAXS we never reapply ``orient_frame_for_display``.
    return mask_fn(raw_shape, theta_deg=waxs_arc, waxs_bsx=waxs_bsx)


# SAXS large-area masks (WAXS shadow + aperture)
_DEFAULT_SAXS_WAXS_SHADOW = {
    "enabled": True,
    "beam_visible_deg": 14.5,
    "clear_edge_deg": 18.0,
    "beam_visible_offset_px": 0.0,
    "edge_margin_px": 0.0,
}
_DEFAULT_SAXS_APERTURE = {
    "enabled": True,
    "agbh_ring_order": 5,
    "q_margin_fraction": 0.01,
    "q_cutoff": None,
}


def _silver_behenate_q_rings(max_q: float, max_order: int = 20) -> np.ndarray:
    D = 5.838  # nm
    orders = np.arange(1, max_order + 1)
    q_rings = 2.0 * np.pi / D * orders
    return q_rings[q_rings <= max_q]


def _make_waxs_shadow_mask(
    image_shape, waxs_arc, beam_center_col_px, **kwargs
) -> np.ndarray:
    ny, nx = image_shape
    enabled = kwargs.get("enabled", True)
    if not enabled or waxs_arc is None:
        return np.ones((1, ny, nx), dtype=bool)
    waxs_arc = np.asarray(waxs_arc, dtype=float).reshape(-1)
    cols = np.arange(nx, dtype=float)[np.newaxis, np.newaxis, :]
    beam_visible_deg = float(kwargs.get("beam_visible_deg", 14.5))
    clear_edge_deg = float(kwargs.get("clear_edge_deg", 18.0))
    clear_span = clear_edge_deg - beam_visible_deg
    if abs(clear_span) < 1e-6:
        return np.ones((waxs_arc.size, ny, nx), dtype=bool)
    start_col = float(beam_center_col_px) + float(
        kwargs.get("beam_visible_offset_px", 0.0)
    )
    clear_col = float(nx - 1) - float(kwargs.get("edge_margin_px", 0.0))
    boundary_col = start_col + (
        (waxs_arc - beam_visible_deg) / clear_span
    ) * (clear_col - start_col)
    boundary_col = np.clip(boundary_col, -1.0, float(nx))
    keep = cols <= boundary_col[:, np.newaxis, np.newaxis]
    keep[waxs_arc >= clear_edge_deg] = True
    return np.broadcast_to(keep, (waxs_arc.size, ny, nx)).copy()


def _make_aperture_mask(q_abs, **kwargs) -> np.ndarray:
    enabled = kwargs.get("enabled", True)
    if not enabled:
        return np.ones_like(q_abs, dtype=bool)
    q_cutoff = kwargs.get("q_cutoff")
    if q_cutoff is None:
        max_q = float(np.nanmax(q_abs))
        ring_order = max(int(kwargs.get("agbh_ring_order", 5)), 1)
        rings = _silver_behenate_q_rings(max_q=max_q, max_order=max(ring_order, 20))
        if rings.size == 0:
            return np.ones_like(q_abs, dtype=bool)
        idx = min(ring_order, rings.size) - 1
        q_margin = float(kwargs.get("q_margin_fraction", 0.08))
        q_cutoff = float(rings[idx]) * (1.0 + q_margin)
    return np.isfinite(q_abs) & (q_abs <= float(q_cutoff))


def make_saxs_large_area_masks(
    image_shape, q_abs, waxs_arc, *, beam_center_col_px, waxs_shadow=None, aperture=None
):
    shadow_cfg = dict(_DEFAULT_SAXS_WAXS_SHADOW)
    if waxs_shadow:
        shadow_cfg.update(waxs_shadow)
    aperture_cfg = dict(_DEFAULT_SAXS_APERTURE)
    if aperture:
        aperture_cfg.update(aperture)
    shadow_mask = _make_waxs_shadow_mask(
        image_shape, waxs_arc, beam_center_col_px, **shadow_cfg
    )
    aperture_mask = np.broadcast_to(
        _make_aperture_mask(q_abs, **aperture_cfg), shadow_mask.shape
    ).copy()
    return shadow_mask & aperture_mask, shadow_mask, aperture_mask


# ===================================================================
# Histogram binning helpers
# ===================================================================


def _histogram2d_pixel_split(
    q2d: np.ndarray,
    chi2d: np.ndarray,
    img: np.ndarray,
    valid: np.ndarray,
    q_edges: np.ndarray,
    chi_edges: np.ndarray,
    pixel_splitting: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Histogram with optional pyFAI-style pixel splitting.

    When *pixel_splitting* > 1 each pixel is subdivided into an
    NxN grid of sub-pixels.  The q/chi position of each sub-pixel is
    estimated via gradient-based interpolation of the full q/chi maps,
    and the pixel intensity is fractionally distributed across the bins
    the sub-pixels fall into.

    Parameters
    ----------
    q2d, chi2d : 2-D arrays
        Per-pixel q and chi maps (same shape as *img*).
    img : 2-D array
        Intensity image (may contain NaN for invalid pixels).
    valid : 2-D bool array
        Mask of pixels to include in the histogram.
    q_edges, chi_edges : 1-D arrays
        Bin edges for the output histogram.
    pixel_splitting : int
        Number of sub-pixel divisions per axis.  1 (default) disables
        splitting and falls back to the standard single-point histogram.

    Returns
    -------
    I_hist, N_hist : 2-D arrays shaped (n_q, n_chi)
    """
    if pixel_splitting <= 1:
        q_sel = q2d[valid].ravel()
        chi_sel = chi2d[valid].ravel()
        I_sel = img[valid].ravel()
        I_hist, _, _ = np.histogram2d(
            q_sel, chi_sel, bins=[q_edges, chi_edges], weights=I_sel,
        )
        N_hist, _, _ = np.histogram2d(
            q_sel, chi_sel, bins=[q_edges, chi_edges],
        )
        return I_hist, N_hist

    # Gradient-based sub-pixel interpolation
    dq_dr = np.gradient(q2d, axis=0)
    dq_dc = np.gradient(q2d, axis=1)
    dchi_dr = np.gradient(chi2d, axis=0)
    dchi_dc = np.gradient(chi2d, axis=1)

    n = pixel_splitting
    offsets = np.linspace(-0.5 + 0.5 / n, 0.5 - 0.5 / n, n)
    weight = 1.0 / (n * n)

    n_q = len(q_edges) - 1
    n_chi = len(chi_edges) - 1
    I_hist = np.zeros((n_q, n_chi), dtype=float)
    N_hist = np.zeros((n_q, n_chi), dtype=float)

    for dr in offsets:
        for dc in offsets:
            q_sub = q2d + dr * dq_dr + dc * dq_dc
            chi_sub = chi2d + dr * dchi_dr + dc * dchi_dc

            sub_valid = valid & np.isfinite(q_sub) & np.isfinite(chi_sub)
            q_sel = q_sub[sub_valid].ravel()
            chi_sel = chi_sub[sub_valid].ravel()
            I_sel = img[sub_valid].ravel() * weight

            I_h, _, _ = np.histogram2d(
                q_sel, chi_sel, bins=[q_edges, chi_edges], weights=I_sel,
            )
            N_h, _, _ = np.histogram2d(
                q_sel, chi_sel, bins=[q_edges, chi_edges],
            )
            I_hist += I_h
            N_hist += N_h * weight

    return I_hist, N_hist


def _qchi_and_iq(
    accum_I: np.ndarray,
    accum_N: np.ndarray,
    q_grid: np.ndarray,
    chi_grid: np.ndarray,
) -> dict[str, Any]:
    with np.errstate(divide="ignore", invalid="ignore"):
        mean_I = np.where(accum_N > 0, accum_I / accum_N, np.nan)
    qchi = xr.Dataset(
        {
            "intensity": (("q", "chi"), mean_I),
            "counts": (("q", "chi"), accum_N),
        },
        coords={"q": q_grid, "chi": chi_grid},
    )
    total_N = accum_N.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        I_1d = np.where(
            total_N > 0,
            (np.nan_to_num(mean_I, nan=0.0) * accum_N).sum(axis=1) / total_N,
            np.nan,
        )
    iq = xr.Dataset(
        {"I": ("q", I_1d), "counts": ("q", total_N)},
        coords={"q": q_grid},
    )
    return {"q_chi": qchi, "iq": iq}


def _stack_qchi_frames(frame_qchi: list[xr.Dataset]) -> xr.Dataset:
    first = frame_qchi[0]
    return xr.Dataset(
        {
            "intensity": (
                ("frame", "q", "chi"),
                np.stack([ds["intensity"].values for ds in frame_qchi], axis=0),
            ),
            "counts": (
                ("frame", "q", "chi"),
                np.stack([ds["counts"].values for ds in frame_qchi], axis=0),
            ),
        },
        coords={
            "frame": np.arange(len(frame_qchi), dtype=int),
            "q": np.asarray(first["q"].values, dtype=float),
            "chi": np.asarray(first["chi"].values, dtype=float),
        },
    )


def _stack_iq_frames(frame_iq: list[xr.Dataset]) -> xr.Dataset:
    first = frame_iq[0]
    return xr.Dataset(
        {
            "I": (
                ("frame", "q"),
                np.stack([ds["I"].values for ds in frame_iq], axis=0),
            ),
            "counts": (
                ("frame", "q"),
                np.stack([ds["counts"].values for ds in frame_iq], axis=0),
            ),
        },
        coords={
            "frame": np.arange(len(frame_iq), dtype=int),
            "q": np.asarray(first["q"].values, dtype=float),
        },
    )


# ===================================================================
# Merge utilities – ported from combined_reduce.py
# ===================================================================

def _interp_axis(source_axis, values, target_axis, fill_value):
    out = np.full(target_axis.shape, fill_value, dtype=float)
    finite = np.isfinite(values) & np.isfinite(source_axis)
    if np.count_nonzero(finite) == 0:
        return out
    x = np.asarray(source_axis[finite], dtype=float)
    y = np.asarray(values[finite], dtype=float)
    order = np.argsort(x)
    out = np.interp(target_axis, x[order], y[order], left=fill_value, right=fill_value)
    return out


def _empty_qchi_like(ref: xr.Dataset) -> xr.Dataset:
    """Return a zero-count q-chi dataset with the same grid as *ref*."""
    q = ref["q"].values
    chi = ref["chi"].values
    nq, nc = len(q), len(chi)
    return xr.Dataset(
        {
            "intensity": (("q", "chi"), np.full((nq, nc), np.nan)),
            "counts":    (("q", "chi"), np.zeros((nq, nc))),
        },
        coords={"q": q, "chi": chi},
    )


def _empty_iq_like(ref: xr.Dataset) -> xr.Dataset:
    """Return a NaN I(q) dataset with the same q grid as *ref*."""
    q = ref["q"].values
    return xr.Dataset(
        {
            "I":      ("q", np.full(len(q), np.nan)),
            "counts": ("q", np.zeros(len(q))),
        },
        coords={"q": q},
    )


def merge_q_chi_weighted(
    saxs_qchi: xr.Dataset | None,
    waxs_qchi: xr.Dataset | None,
    n_q: int = 1000,
    n_chi: int = 360,
) -> xr.Dataset | None:
    """Merge SAXS and WAXS q-chi maps on a common grid with count-weighting.

    Returns None if both inputs are None. Returns the single detector's data
    (re-gridded) if only one is present.
    """
    if saxs_qchi is None and waxs_qchi is None:
        return None
    # Single-detector passthrough: use the available one for both slots
    if saxs_qchi is None:
        saxs_qchi = _empty_qchi_like(waxs_qchi)
    if waxs_qchi is None:
        waxs_qchi = _empty_qchi_like(saxs_qchi)

    saxs_q = np.asarray(saxs_qchi["q"].values, dtype=float)
    waxs_q = np.asarray(waxs_qchi["q"].values, dtype=float)
    q_min = min(float(np.nanmin(saxs_q)), float(np.nanmin(waxs_q)))
    q_max = max(float(np.nanmax(saxs_q)), float(np.nanmax(waxs_q)))
    q_grid = np.linspace(q_min, q_max, n_q)

    saxs_chi = np.asarray(saxs_qchi["chi"].values, dtype=float)
    waxs_chi = np.asarray(waxs_qchi["chi"].values, dtype=float)
    chi_min = min(float(np.nanmin(saxs_chi)), float(np.nanmin(waxs_chi)))
    chi_max = max(float(np.nanmax(saxs_chi)), float(np.nanmax(waxs_chi)))
    chi_grid = np.linspace(chi_min, chi_max, n_chi)

    saxs_I = np.asarray(saxs_qchi["intensity"].values, dtype=float)
    saxs_N = np.asarray(saxs_qchi["counts"].values, dtype=float)
    waxs_I = np.asarray(waxs_qchi["intensity"].values, dtype=float)
    waxs_N = np.asarray(waxs_qchi["counts"].values, dtype=float)

    def _regrid_2d(src_q, src_chi, data, target_q, target_chi, fill):
        """Regrid a 2D (q, chi) array onto a new grid via nearest-neighbor."""
        from scipy.interpolate import RegularGridInterpolator
        finite_data = np.where(np.isfinite(data), data, fill)
        interp = RegularGridInterpolator(
            (src_q, src_chi), finite_data,
            method="nearest", bounds_error=False, fill_value=fill,
        )
        tq, tc = np.meshgrid(target_q, target_chi, indexing="ij")
        return interp((tq, tc))

    s_I_interp = _regrid_2d(saxs_q, saxs_chi, saxs_I, q_grid, chi_grid, np.nan)
    s_N_interp = _regrid_2d(saxs_q, saxs_chi, saxs_N, q_grid, chi_grid, 0.0)
    w_I_interp = _regrid_2d(waxs_q, waxs_chi, waxs_I, q_grid, chi_grid, np.nan)
    w_N_interp = _regrid_2d(waxs_q, waxs_chi, waxs_N, q_grid, chi_grid, 0.0)

    total_N = s_N_interp + w_N_interp
    with np.errstate(divide="ignore", invalid="ignore"):
        merged_I = np.where(
            total_N > 0,
            (np.nan_to_num(s_I_interp, nan=0.0) * s_N_interp
             + np.nan_to_num(w_I_interp, nan=0.0) * w_N_interp) / total_N,
            np.nan,
        )

    return xr.Dataset(
        {
            "intensity": (("q", "chi"), merged_I),
            "counts": (("q", "chi"), total_N),
            "saxs_intensity": (("q", "chi"), s_I_interp),
            "saxs_counts": (("q", "chi"), s_N_interp),
            "waxs_intensity": (("q", "chi"), w_I_interp),
            "waxs_counts": (("q", "chi"), w_N_interp),
        },
        coords={"q": q_grid, "chi": chi_grid},
    )


def merge_iq_profiles(
    merged_qchi: xr.Dataset | None,
    saxs_iq: xr.Dataset | None,
    waxs_iq: xr.Dataset | None,
) -> xr.Dataset | None:
    """Produce merged I(q) by azimuthal integration of the merged q-chi map."""
    if merged_qchi is None:
        return None
    q_grid = np.asarray(merged_qchi["q"].values, dtype=float)
    merged_I = np.asarray(merged_qchi["intensity"].values, dtype=float)
    merged_N = np.asarray(merged_qchi["counts"].values, dtype=float)

    total_N = merged_N.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        I_1d = np.where(
            total_N > 0,
            (np.nan_to_num(merged_I, nan=0.0) * merged_N).sum(axis=1) / total_N,
            np.nan,
        )

    if saxs_iq is not None:
        saxs_q_src = np.asarray(saxs_iq["q"].values, dtype=float)
        saxs_I_src = np.asarray(saxs_iq["I"].values, dtype=float)
        saxs_I_interp = _interp_axis(saxs_q_src, saxs_I_src, q_grid, np.nan)
    else:
        saxs_I_interp = np.full(len(q_grid), np.nan)

    if waxs_iq is not None:
        waxs_q_src = np.asarray(waxs_iq["q"].values, dtype=float)
        waxs_I_src = np.asarray(waxs_iq["I"].values, dtype=float)
        waxs_I_interp = _interp_axis(waxs_q_src, waxs_I_src, q_grid, np.nan)
    else:
        waxs_I_interp = np.full(len(q_grid), np.nan)

    return xr.Dataset(
        {
            "I": ("q", I_1d),
            "counts": ("q", total_N),
            "saxs_I": ("q", saxs_I_interp),
            "waxs_I": ("q", waxs_I_interp),
        },
        coords={"q": q_grid},
    )


def _build_per_frame_iq(
    merged_iq: xr.Dataset | None,
    saxs_result: dict[str, Any] | None,
    waxs_result: dict[str, Any] | None,
    scan_info: dict[str, Any] | None = None,
) -> xr.Dataset | None:
    """Build per-frame I(q) Dataset on the same q grid as merged_iq.

    Combines per-frame SAXS and WAXS I(q) via interpolation onto the
    merged q grid, then produces a count-weighted merge per frame.

    If *scan_info* is provided and contains per-frame primary-stream
    scalars (step_candidates with a 'values' key), they are attached
    as data variables on the (frame,) dimension.

    Returns
    -------
    xr.Dataset with dims (frame, q) and variables I, saxs_I, waxs_I,
    plus any per-frame primary scalars, or None if no per-frame data
    is available.
    """
    if merged_iq is None:
        return None

    q_grid = np.asarray(merged_iq["q"].values, dtype=float)
    n_q = len(q_grid)

    saxs_iq_frames = saxs_result["iq_frames"] if saxs_result else None
    waxs_iq_frames = waxs_result["iq_frames"] if waxs_result else None

    if saxs_iq_frames is None and waxs_iq_frames is None:
        return None

    # Determine number of frames from whichever detector is present
    if saxs_iq_frames is not None and waxs_iq_frames is not None:
        n_frames = max(
            len(saxs_iq_frames["frame"]),
            len(waxs_iq_frames["frame"]),
        )
    elif saxs_iq_frames is not None:
        n_frames = len(saxs_iq_frames["frame"])
    else:
        n_frames = len(waxs_iq_frames["frame"])

    saxs_I_2d = np.full((n_frames, n_q), np.nan)
    waxs_I_2d = np.full((n_frames, n_q), np.nan)

    if saxs_iq_frames is not None:
        saxs_q_src = np.asarray(saxs_iq_frames["q"].values, dtype=float)
        saxs_I_src = np.asarray(saxs_iq_frames["I"].values, dtype=float)
        n_saxs = saxs_I_src.shape[0]
        for fi in range(min(n_saxs, n_frames)):
            saxs_I_2d[fi] = _interp_axis(saxs_q_src, saxs_I_src[fi], q_grid, np.nan)

    if waxs_iq_frames is not None:
        waxs_q_src = np.asarray(waxs_iq_frames["q"].values, dtype=float)
        waxs_I_src = np.asarray(waxs_iq_frames["I"].values, dtype=float)
        n_waxs = waxs_I_src.shape[0]
        for fi in range(min(n_waxs, n_frames)):
            waxs_I_2d[fi] = _interp_axis(waxs_q_src, waxs_I_src[fi], q_grid, np.nan)

    # Count-weighted merge per frame (same logic as merge_iq_profiles)
    saxs_N_2d = np.zeros((n_frames, n_q), dtype=float)
    waxs_N_2d = np.zeros((n_frames, n_q), dtype=float)

    if saxs_iq_frames is not None:
        saxs_counts_src = np.asarray(saxs_iq_frames["counts"].values, dtype=float)
        saxs_q_src = np.asarray(saxs_iq_frames["q"].values, dtype=float)
        n_saxs = saxs_counts_src.shape[0]
        for fi in range(min(n_saxs, n_frames)):
            saxs_N_2d[fi] = _interp_axis(saxs_q_src, saxs_counts_src[fi], q_grid, 0.0)

    if waxs_iq_frames is not None:
        waxs_counts_src = np.asarray(waxs_iq_frames["counts"].values, dtype=float)
        waxs_q_src = np.asarray(waxs_iq_frames["q"].values, dtype=float)
        n_waxs = waxs_counts_src.shape[0]
        for fi in range(min(n_waxs, n_frames)):
            waxs_N_2d[fi] = _interp_axis(waxs_q_src, waxs_counts_src[fi], q_grid, 0.0)

    total_N = saxs_N_2d + waxs_N_2d
    with np.errstate(divide="ignore", invalid="ignore"):
        merged_I_2d = np.where(
            total_N > 0,
            (np.nan_to_num(saxs_I_2d, nan=0.0) * saxs_N_2d
             + np.nan_to_num(waxs_I_2d, nan=0.0) * waxs_N_2d) / total_N,
            np.nan,
        )

    data_vars: dict[str, Any] = {
        "I": (("frame", "q"), merged_I_2d),
        "saxs_I": (("frame", "q"), saxs_I_2d),
        "waxs_I": (("frame", "q"), waxs_I_2d),
    }

    # Attach per-frame primary-stream scalars as data variables
    if scan_info is not None:
        for cand in scan_info.get("step_candidates", []):
            vals = cand.get("values")
            if vals is None:
                continue
            vals = np.asarray(vals, dtype=float)
            if vals.shape[0] == n_frames:
                data_vars[cand["name"]] = ("frame", vals)

    return xr.Dataset(
        data_vars,
        coords={
            "q": q_grid,
            "frame": np.arange(n_frames, dtype=int),
        },
    )


# -------------------------------------------------------------------
# Multi-scan merging
# -------------------------------------------------------------------

def merge_multiple_qchi(
    datasets: list[xr.Dataset],
    n_q: int = 2000,
    n_chi: int = 360,
) -> xr.Dataset:
    """Count-weighted merge of N ``(q, chi)`` datasets onto a common grid.

    Each input must have ``intensity`` and ``counts`` variables with
    dimensions ``(q, chi)``.  The output grid spans the union of all
    input q/chi ranges.
    """
    from scipy.interpolate import RegularGridInterpolator

    if not datasets:
        raise ValueError("Need at least one dataset to merge")
    if len(datasets) == 1:
        return datasets[0]

    all_q = [np.asarray(ds["q"].values, dtype=float) for ds in datasets]
    all_chi = [np.asarray(ds["chi"].values, dtype=float) for ds in datasets]
    q_min = min(float(q.min()) for q in all_q)
    q_max = max(float(q.max()) for q in all_q)
    chi_min = min(float(c.min()) for c in all_chi)
    chi_max = max(float(c.max()) for c in all_chi)
    q_grid = np.linspace(q_min, q_max, n_q)
    chi_grid = np.linspace(chi_min, chi_max, n_chi)
    tq, tc = np.meshgrid(q_grid, chi_grid, indexing="ij")
    pts = (tq, tc)

    accum_IN = np.zeros((n_q, n_chi), dtype=float)
    accum_N = np.zeros((n_q, n_chi), dtype=float)

    for ds, src_q, src_chi in zip(datasets, all_q, all_chi):
        I_src = np.asarray(ds["intensity"].values, dtype=float)
        N_src = np.asarray(ds["counts"].values, dtype=float)

        N_fill = np.where(np.isfinite(N_src), N_src, 0.0)
        IN_src = np.where(np.isfinite(I_src), I_src * N_fill, 0.0)

        interp_N = RegularGridInterpolator(
            (src_q, src_chi), N_fill,
            method="nearest", bounds_error=False, fill_value=0.0,
        )
        interp_IN = RegularGridInterpolator(
            (src_q, src_chi), IN_src,
            method="nearest", bounds_error=False, fill_value=0.0,
        )
        accum_N += interp_N(pts)
        accum_IN += interp_IN(pts)

    with np.errstate(divide="ignore", invalid="ignore"):
        merged_I = np.where(accum_N > 0, accum_IN / accum_N, np.nan)

    return xr.Dataset(
        {
            "intensity": (("q", "chi"), merged_I),
            "counts": (("q", "chi"), accum_N),
        },
        coords={"q": q_grid, "chi": chi_grid},
    )


def merge_multiple_iq(
    merged_qchi: xr.Dataset,
) -> xr.Dataset:
    """Azimuthally average a merged q-chi map into I(q)."""
    q_grid = np.asarray(merged_qchi["q"].values, dtype=float)
    merged_I = np.asarray(merged_qchi["intensity"].values, dtype=float)
    merged_N = np.asarray(merged_qchi["counts"].values, dtype=float)

    total_N = merged_N.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        I_1d = np.where(
            total_N > 0,
            (np.nan_to_num(merged_I, nan=0.0) * merged_N).sum(axis=1) / total_N,
            np.nan,
        )
    return xr.Dataset(
        {"I": ("q", I_1d), "counts": ("q", total_N)},
        coords={"q": q_grid},
    )


def merge_reduction_results(
    *results: "CombinedReductionResult",
    n_q: int = 2000,
    n_chi: int = 360,
) -> Tuple[xr.Dataset, xr.Dataset]:
    """Merge multiple :class:`CombinedReductionResult` objects.

    Collects the per-scan merged q-chi maps and combines them with
    count-weighting.

    Returns ``(merged_qchi, merged_iq)``.
    """
    qchi_list = [r.merged_qchi for r in results if r.merged_qchi is not None]
    if not qchi_list:
        raise ValueError("No merged q-chi data in any of the results")
    merged_qchi = merge_multiple_qchi(qchi_list, n_q=n_q, n_chi=n_chi)
    merged_iq = merge_multiple_iq(merged_qchi)
    return merged_qchi, merged_iq


# ===================================================================
# Result dataclasses
# ===================================================================

@dataclass(frozen=True)
class CombinedReductionResult:
    uid: str
    scan_info: dict[str, Any]
    saxs: dict[str, Any] | None
    waxs: dict[str, Any] | None
    merged_qchi: xr.Dataset | None
    merged_iq: xr.Dataset | None
    per_frame_iq: xr.Dataset | None = None
    timing: dict[str, float] | None = None
    geometry: str = "transmission"
    incident_angle_deg: float = 0.0


@dataclass(frozen=True)
class GIReductionResult:
    """Result of a grazing-incidence WAXS reduction.

    Attributes
    ----------
    uid : str
        Tiled run UID.
    sample_name : str
        Sample name from the start document.
    scan_motor : str
        Name of the scanned motor (e.g. ``'piezo_th'``).
    scan_motor_values : np.ndarray
        Per-frame values of the scanned motor.
    alpha_i_deg : np.ndarray
        Per-frame incident angle (degrees).
    alpha_i_source : str
        Description of how alpha_i was determined.
    qxy_grid : np.ndarray
        1-D q_xy bin centres (nm\ :sup:`-1`).
    qz_grid : np.ndarray
        1-D q_z bin centres (nm\ :sup:`-1`).
    frames : list[np.ndarray]
        Per-frame I(qxy, qz) images (shape ``(n_qxy, n_qz)``).
    summed : np.ndarray
        Averaged I(qxy, qz) over all frames.
    timing : dict[str, float] | None
        Timing breakdown.
    """
    uid: str
    sample_name: str
    scan_motor: str
    scan_motor_values: np.ndarray
    alpha_i_deg: np.ndarray
    alpha_i_source: str
    qxy_grid: np.ndarray
    qz_grid: np.ndarray
    frames: list  # list[np.ndarray]
    summed: np.ndarray
    timing: dict[str, float] | None = None

    # -- Line cut helpers --------------------------------------------------

    def line_cut_qxy(self, qz_center: float, qz_width: float = 0.05,
                     frame: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        """I(qxy) at constant qz +/- width.  *frame*=None uses the sum."""
        img = self.summed if frame is None else self.frames[frame]
        mask = (self.qz_grid >= qz_center - qz_width) & (self.qz_grid <= qz_center + qz_width)
        if not mask.any():
            return self.qxy_grid, np.full_like(self.qxy_grid, np.nan)
        return self.qxy_grid, np.nanmean(img[:, mask], axis=1)

    def line_cut_qz(self, qxy_center: float, qxy_width: float = 0.1,
                    frame: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        """I(qz) at constant qxy +/- width.  *frame*=None uses the sum."""
        img = self.summed if frame is None else self.frames[frame]
        mask = (self.qxy_grid >= qxy_center - qxy_width) & (self.qxy_grid <= qxy_center + qxy_width)
        if not mask.any():
            return self.qz_grid, np.full_like(self.qz_grid, np.nan)
        return self.qz_grid, np.nanmean(img[mask, :], axis=0)


# ===================================================================
# Grazing-incidence helpers
# ===================================================================

def lab_to_sample_frame(
    qx_lab: np.ndarray,
    qy_lab: np.ndarray,
    qz_lab: np.ndarray,
    alpha_i_deg: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rotate lab-frame q into the sample frame for grazing incidence.

    Parameters
    ----------
    qx_lab, qy_lab, qz_lab : ndarray
        Lab-frame q components (horizontal, vertical-up, along-beam).
    alpha_i_deg : float
        Incident angle in degrees.

    Returns
    -------
    (qx_s, qy_s, qz_s) where qz_s is along the surface normal.
    """
    ai = np.deg2rad(alpha_i_deg)
    cos_ai = np.cos(ai)
    sin_ai = np.sin(ai)
    qx_s = qx_lab
    qy_s = qy_lab * sin_ai - qz_lab * cos_ai
    qz_s = qy_lab * cos_ai + qz_lab * sin_ai
    return qx_s, qy_s, qz_s


import re as _re

_AI_PATTERNS = [
    _re.compile(r'(?:^|[_\-])ai[_\-]?([0-9]+\.?[0-9]*)', _re.IGNORECASE),
    _re.compile(r'alpha[_]?i?[_\-]?([0-9]+\.?[0-9]*)', _re.IGNORECASE),
    _re.compile(r'incident[_\-]?([0-9]+\.?[0-9]*)', _re.IGNORECASE),
]


def parse_incident_angle_from_string(s: str) -> float | None:
    """Extract incident angle (degrees) from a sample-name string.

    Recognized patterns (case-insensitive):
    ``ai0.12``, ``ai_0.12``, ``alpha0.12``, ``alpha_i0.12``, ``incident0.12``.

    Returns ``float`` or ``None``.
    """
    for pat in _AI_PATTERNS:
        m = pat.search(s)
        if m:
            val = float(m.group(1))
            if 0 < val < 90:
                return val
    return None


def find_incident_angle(
    run,
    n_frames: int,
    manual_override: float | None = None,
    theta_offset: float = 0.0,
) -> tuple[np.ndarray, str]:
    """Determine per-frame incident angle (degrees).

    Only fetches individual columns from tiled — never reads the full
    primary stream.

    Priority order:
    1. ``manual_override`` if not None
    2. ``sample_name`` string parsing (``ai0.12`` etc.)
    3. Motor positions: ``stage_th + piezo_th + theta_offset``

    Returns ``(alpha_i_array, source_description)``.
    """
    start = run.metadata.get("start", {})
    sample_name = start.get("sample_name", "")

    if manual_override is not None:
        return np.full(n_frames, float(manual_override)), \
            f"manual override = {manual_override}\u00b0"

    ai_from_name = parse_incident_angle_from_string(sample_name)
    if ai_from_name is not None:
        return np.full(n_frames, ai_from_name), \
            f"sample_name parsed: ai={ai_from_name}\u00b0"

    # Motor positions — fetch individual columns only
    primary_ds = run["primary"]
    baseline_ds = run["baseline"]

    stage_th = None
    try:
        stage_th = float(baseline_ds["stage_th"].read()[0])
    except (KeyError, IndexError):
        pass

    piezo_th = None
    if "piezo_th" in primary_ds:
        piezo_th = np.asarray(primary_ds["piezo_th"].read(), dtype=float)
    else:
        try:
            piezo_th = np.full(n_frames, float(baseline_ds["piezo_th"].read()[0]))
        except (KeyError, IndexError):
            pass

    if stage_th is not None and piezo_th is not None:
        ai = stage_th + piezo_th + theta_offset
        return ai, f"stage_th({stage_th:.4f}) + piezo_th + offset({theta_offset})"

    if piezo_th is not None:
        return piezo_th + theta_offset, \
            f"piezo_th + offset({theta_offset}) [no stage_th]"

    raise RuntimeError(
        "Cannot determine incident angle. Pass incident_angle_deg manually."
    )


def integrate_waxs_gi(
    waxs_raw: xr.DataArray,
    mask_fn,
    alpha_i_deg: np.ndarray,
    n_qxy: int = 500,
    n_qz: int = 500,
    cal: WAXSCalibration | None = None,
    dezinger_threshold: float | None = None,
    dezinger_kernel: int = 5,
) -> dict[str, Any]:
    """WAXS GI reduction: bin each frame into (qxy, qz) in the sample frame.

    Parameters
    ----------
    waxs_raw : xr.DataArray
        Raw WAXS images from ``TiledSMISWAXSLoader.loadSingleImage``.
    mask_fn : callable or None
        ``mask_fn(image_shape_raw, theta_deg, waxs_bsx) -> bool mask``.
    alpha_i_deg : array-like
        Per-frame incident angle (degrees).
    n_qxy, n_qz : int
        Grid dimensions.
    cal : WAXSCalibration or None
        Detector calibration.  None uses ``_DEFAULT_CAL``.
    dezinger_threshold, dezinger_kernel
        Hot-pixel rejection parameters.

    Returns
    -------
    dict with keys ``qxy_grid``, ``qz_grid``, ``frames``, ``summed``.
    """
    if cal is None:
        cal = WAXSCalibration(**_DEFAULT_CAL)

    images = np.asarray(waxs_raw.values, dtype=float)
    if images.ndim == 2:
        images = images[np.newaxis, :, :]
    n_frames = images.shape[0]

    arc_angles = np.asarray(
        waxs_raw.coords[waxs_raw.dims[0]].values, dtype=float,
    )
    bsx_per_frame = np.asarray(
        waxs_raw.attrs.get("smi_waxs_bsx_per_frame", [0.0] * n_frames),
        dtype=float,
    )
    alpha_arr = np.asarray(alpha_i_deg, dtype=float)
    if alpha_arr.size == 1:
        alpha_arr = np.full(n_frames, float(alpha_arr))

    # Build detector geometry (constant arc for GI scans)
    img_0_rot, _ = rotate_image_and_mask(images[0], k=cal.rotation_k)
    rot_shape = img_0_rot.shape
    theta_arc = float(arc_angles[0])
    bc = cal.beam_center_at_angle(theta_arc)
    det = MultiPanelArcDetector(
        image_shape=rot_shape,
        panel_specs=cal.make_panel_specs(),
        wavelength_nm=cal.wavelength_nm,
        pixel_size_mm=cal.pixel_size_mm,
        sample_distance_mm=cal.sample_distance_mm,
        beam_center_px=bc,
        theta_zero_deg=cal.theta_zero_deg,
        sample_offset_x_mm=cal.sample_offset_x_mm,
        sample_offset_z_mm=cal.sample_offset_z_mm,
    )
    qds = det.qmap(theta_arc)
    # Apply the same sign conventions as the transmission code so that
    # "up" in the lab frame is positive qy.
    qx_lab = cal.q_horizontal_sign * np.asarray(qds["qx"].values, dtype=float)
    qy_lab = cal.q_vertical_sign * np.asarray(qds["qy"].values, dtype=float)
    qz_lab_arr = np.asarray(qds["qz"].values, dtype=float)

    # --- First pass: global qxy/qz range ---
    _qxy_mn, _qxy_mx, _qz_mn, _qz_mx = [], [], [], []
    for fi in range(n_frames):
        qx_s, qy_s, qz_s = lab_to_sample_frame(
            qx_lab, qy_lab, qz_lab_arr, alpha_arr[fi],
        )
        qxy = np.sign(qx_s) * np.sqrt(qx_s ** 2 + qy_s ** 2)
        ok = np.isfinite(qxy) & np.isfinite(qz_s)
        if ok.any():
            _qxy_mn.append(float(np.nanmin(qxy[ok])))
            _qxy_mx.append(float(np.nanmax(qxy[ok])))
            _qz_mn.append(float(np.nanmin(qz_s[ok])))
            _qz_mx.append(float(np.nanmax(qz_s[ok])))

    qxy_edges = np.linspace(min(_qxy_mn), max(_qxy_mx), n_qxy + 1)
    qz_edges = np.linspace(min(_qz_mn), max(_qz_mx), n_qz + 1)
    qxy_grid = 0.5 * (qxy_edges[:-1] + qxy_edges[1:])
    qz_grid = 0.5 * (qz_edges[:-1] + qz_edges[1:])

    # --- Second pass: histogram each frame ---
    accum_I = np.zeros((n_qxy, n_qz), dtype=float)
    accum_N = np.zeros((n_qxy, n_qz), dtype=float)
    frame_maps: list[np.ndarray] = []

    for fi in range(n_frames):
        theta_f = float(arc_angles[fi])
        ai = alpha_arr[fi]
        bsx = float(bsx_per_frame[fi]) if fi < len(bsx_per_frame) else 0.0

        img_rot, _ = rotate_image_and_mask(images[fi], k=cal.rotation_k)

        mask_rot = None
        if mask_fn is not None:
            try:
                mask_rot = mask_fn(images[fi].shape, theta_f, bsx)
            except Exception as exc:
                warnings.warn(
                    f"mask_fn failed for frame {fi}: {exc}", stacklevel=2,
                )

        if dezinger_threshold is not None:
            dz = dezinger(
                img_rot, kernel_size=dezinger_kernel,
                threshold=dezinger_threshold,
            )
            mask_rot = (mask_rot & dz) if mask_rot is not None else dz

        qx_s, qy_s, qz_s = lab_to_sample_frame(
            qx_lab, qy_lab, qz_lab_arr, ai,
        )
        qxy = np.sign(qx_s) * np.sqrt(qx_s ** 2 + qy_s ** 2)

        valid = np.isfinite(qxy) & np.isfinite(qz_s) & np.isfinite(img_rot)
        if mask_rot is not None:
            valid &= mask_rot

        qxy_v = qxy[valid].ravel()
        qz_v = qz_s[valid].ravel()
        I_v = img_rot[valid].ravel()

        I_hist, _, _ = np.histogram2d(
            qxy_v, qz_v, bins=[qxy_edges, qz_edges], weights=I_v,
        )
        N_hist, _, _ = np.histogram2d(
            qxy_v, qz_v, bins=[qxy_edges, qz_edges],
        )
        accum_I += I_hist
        accum_N += N_hist

        with np.errstate(invalid="ignore", divide="ignore"):
            frame_maps.append(np.where(N_hist > 0, I_hist / N_hist, np.nan))

    with np.errstate(invalid="ignore", divide="ignore"):
        summed = np.where(accum_N > 0, accum_I / accum_N, np.nan)

    return {
        "qxy_grid": qxy_grid,
        "qz_grid": qz_grid,
        "frames": frame_maps,
        "summed": summed,
    }


# ===================================================================
# Grazing-incidence reduction entry point
# ===================================================================

def reduce_smi_gi(
    uid: str,
    tiled_uri: str = "https://tiled.nsls2.bnl.gov",
    catalog: str = "smi/migration",
    waxs_mask_path: str | Path | None = None,
    n_qxy: int = 500,
    n_qz: int = 500,
    incident_angle_deg: float | None = None,
    theta_offset: float = -0.5,
    waxs_beam_col_per_arc_deg: float = 0.08,
    beamstop_max_abs_arc_deg: float = 15.0,
    dezinger_threshold: float | None = 30000.0,
    dezinger_kernel: int = 5,
    waxs_cal_overrides: dict[str, Any] | None = None,
) -> GIReductionResult:
    """Full grazing-incidence WAXS reduction pipeline.

    Parameters
    ----------
    uid : str
        Tiled run UID.
    tiled_uri, catalog : str
        Tiled connection parameters.
    waxs_mask_path : str or Path or None
        Path to the WAXS mask JSON.  ``None`` (the default) uses the
        bundled SMI default mask shipped with PyHyperScattering
        (``PyHyperScattering.smi_defaults.default_waxs_mask_path``).
    n_qxy, n_qz : int
        Output grid dimensions.
    incident_angle_deg : float or None
        Manual incident-angle override.  None = auto-detect from
        sample_name or motor positions.
    theta_offset : float
        Added to ``stage_th + piezo_th`` when auto-detecting the
        incident angle.
    waxs_beam_col_per_arc_deg : float
        Beam-centre drift per degree of waxs_arc.
    beamstop_max_abs_arc_deg : float
        Mask beamstop only for ``|arc| <= this``.
    dezinger_threshold, dezinger_kernel
        Hot-pixel rejection parameters.
    waxs_cal_overrides : dict or None
        Extra overrides for ``WAXSCalibration`` fields.

    Returns
    -------
    GIReductionResult
    """
    import time as _time
    from PyHyperScattering.SMISWAXSLoader import (
        TiledSMISWAXSLoader,
        resolve_waxs_geometry,
    )

    t0 = _time.perf_counter()

    # --- Connect & get metadata ---
    from tiled.client import from_uri
    client = from_uri(tiled_uri)
    run = client[catalog + "/" + uid]
    start = run.metadata.get("start", {})
    sample_name = start.get("sample_name", "")
    n_frames = start.get("num_points", 1)
    scan_motor = (start.get("motors") or ["unknown"])[0]

    # --- Incident angle ---
    alpha_i, ai_source = find_incident_angle(
        run, n_frames,
        manual_override=incident_angle_deg,
        theta_offset=theta_offset,
    )

    # --- Scan motor values (for labelling) ---
    scan_motor_values = alpha_i.copy()  # default: use alpha_i
    try:
        primary_ds = run["primary"]
        if scan_motor in primary_ds:
            scan_motor_values = np.asarray(
                primary_ds[scan_motor].read(), dtype=float,
            )
    except Exception:
        pass

    # --- Load WAXS images ---
    loader = TiledSMISWAXSLoader(tiled_uri=tiled_uri, catalog=catalog)
    waxs_raw = loader.loadSingleImage(uid, detector="waxs")
    if waxs_raw is None:
        raise RuntimeError(f"No WAXS data in scan {uid}")
    t_load = _time.perf_counter()

    # --- Mask ---
    from PyHyperScattering.smi_defaults import resolve_mask_path
    waxs_mask_path = resolve_mask_path(waxs_mask_path, detector="waxs")
    waxs_mask_fn = None
    if waxs_mask_path is not None:
        waxs_mask_fn = make_waxs_mask_callable(
            waxs_mask_path,
            beamstop_max_abs_arc_deg=beamstop_max_abs_arc_deg,
        )

    # --- WAXS calibration ---
    cal_dict: dict[str, Any] = dict(_DEFAULT_CAL)
    cal_dict["beam_col_per_arc_deg"] = waxs_beam_col_per_arc_deg
    if waxs_cal_overrides:
        cal_dict.update(waxs_cal_overrides)
    waxs_cal = WAXSCalibration(**cal_dict)

    # --- Integrate ---
    t_int = _time.perf_counter()
    gi_out = integrate_waxs_gi(
        waxs_raw=waxs_raw,
        mask_fn=waxs_mask_fn,
        alpha_i_deg=alpha_i,
        n_qxy=n_qxy,
        n_qz=n_qz,
        cal=waxs_cal,
        dezinger_threshold=dezinger_threshold,
        dezinger_kernel=dezinger_kernel,
    )
    t_done = _time.perf_counter()

    return GIReductionResult(
        uid=uid,
        sample_name=sample_name,
        scan_motor=scan_motor,
        scan_motor_values=scan_motor_values,
        alpha_i_deg=alpha_i,
        alpha_i_source=ai_source,
        qxy_grid=gi_out["qxy_grid"],
        qz_grid=gi_out["qz_grid"],
        frames=gi_out["frames"],
        summed=gi_out["summed"],
        timing={
            "total": t_done - t0,
            "tiled_load": t_load - t0,
            "integrate": t_done - t_int,
        },
    )


# ===================================================================
# SAXS integration
# ===================================================================

def integrate_saxs(
    saxs_raw: xr.DataArray,
    mask: np.ndarray | None,
    n_q: int = 1000,
    n_chi: int = 360,
    solid_angle_correction: bool = False,
    rotate_cw_90: bool = False,
    waxs_arc: np.ndarray | None = None,
    beam_center_col_px: float | None = None,
    dynamic_saxs_mask: bool = False,
    dynamic_saxs_kwargs: dict[str, Any] | None = None,
    dezinger_threshold: float | None = None,
    dezinger_kernel: int = 5,
    cache_geometry: bool = True,
    pixel_splitting: int = 1,
) -> dict[str, Any]:
    """SAXS reduction via direct pixel-space q-map and histogram binning."""
    attrs = saxs_raw.attrs
    # Build geometry arrays from attrs
    dist_m = float(attrs["dist"])
    poni1_m = float(attrs["poni1"])
    poni2_m = float(attrs["poni2"])
    pixel1_m = float(attrs["pixel1"])
    pixel2_m = float(attrs["pixel2"])
    wavelength_m = float(attrs["wavelength"]) * 1e-10
    wavelength_nm = wavelength_m * 1e9

    images = np.asarray(saxs_raw.values, dtype=float)
    if images.ndim == 2:
        images = images[np.newaxis, :, :]

    mask_use = None if mask is None else np.asarray(mask, dtype=bool)

    shape = images.shape[-2:]
    ny, nx = shape

    # Check persistent geometry cache
    _saxs_key = _saxs_cache_key(dist_m, poni1_m, poni2_m, pixel1_m, pixel2_m, wavelength_m, shape)
    _cached = _SAXS_GEOMETRY_CACHE.get(_saxs_key) if cache_geometry else None

    if _cached is not None:
        q2d, qh2d, qv2d, chi_deg_2d, sa_base = _cached
        # Recompute sa based on current solid_angle_correction setting
        sa = sa_base if solid_angle_correction else None
    else:
        rr, cc = np.meshgrid(
            np.arange(ny, dtype=float),
            np.arange(nx, dtype=float),
            indexing="ij",
        )
        bc_row = poni1_m / pixel1_m
        bc_col = poni2_m / pixel2_m

        x_m = (cc - bc_col) * pixel2_m
        y_m = -(rr - bc_row) * pixel1_m
        r_m = np.sqrt(x_m**2 + y_m**2 + dist_m**2)
        k = 2.0 * np.pi / wavelength_nm

        qh2d = k * x_m / r_m
        qv2d = k * y_m / r_m
        qz2d = k * (dist_m / r_m - 1.0)
        q2d = np.sqrt(qh2d**2 + qv2d**2 + qz2d**2)
        chi_deg_2d = np.rad2deg(np.arctan2(qh2d, qv2d))

        pixel_area_m2 = pixel1_m * pixel2_m
        with np.errstate(invalid="ignore", divide="ignore"):
            sa_base = pixel_area_m2 * np.maximum(dist_m, 0.0) / (r_m**3)
        sa = sa_base if solid_angle_correction else None

        if cache_geometry:
            _SAXS_GEOMETRY_CACHE[_saxs_key] = (q2d, qh2d, qv2d, chi_deg_2d, sa_base)

    bc_col = poni2_m / pixel2_m

    base_valid = np.isfinite(q2d) & np.isfinite(chi_deg_2d)
    if mask_use is not None:
        base_valid &= mask_use

    # Dynamic large-area masks
    if waxs_arc is None:
        dim0 = saxs_raw.dims[0] if saxs_raw.ndim > 2 else None
        if dim0 == "waxs_arc":
            waxs_arc = np.asarray(
                saxs_raw.coords["waxs_arc"].values, dtype=float
            )
    if beam_center_col_px is None:
        beam_center_col_px = bc_col

    mask_options = dict(dynamic_saxs_kwargs or {})
    ws_kw = dict(mask_options.pop("waxs_shadow", {}) or {})
    ap_kw = dict(mask_options.pop("aperture", {}) or {})

    large_area_mask, _, _ = make_saxs_large_area_masks(
        shape,
        q2d,
        waxs_arc,
        beam_center_col_px=float(beam_center_col_px),
        waxs_shadow=ws_kw,
        aperture=ap_kw,
    )

    n_frames = images.shape[0]
    per_frame_valid = np.broadcast_to(base_valid, images.shape).copy()
    if large_area_mask.shape[0] == 1:
        per_frame_valid &= large_area_mask
    elif large_area_mask.shape[0] >= n_frames:
        per_frame_valid &= large_area_mask[:n_frames]

    # Per-frame dezinger: flag hot pixels
    if dezinger_threshold is not None:
        for _di in range(n_frames):
            dz_mask = dezinger(images[_di], kernel_size=dezinger_kernel,
                                threshold=dezinger_threshold)
            per_frame_valid[_di] &= dz_mask

    # Bin edges
    q_vals = q2d[base_valid]
    chi_vals = chi_deg_2d[base_valid]
    if q_vals.size > 0:
        q_min, q_max = np.percentile(q_vals, [0.5, 99.5])
        q_edges = np.linspace(float(q_min), float(q_max), n_q + 1)
    else:
        q_edges = np.linspace(0.0, 10.0, n_q + 1)
    q_grid = 0.5 * (q_edges[:-1] + q_edges[1:])

    if chi_vals.size > 0:
        chi_min, chi_max = np.percentile(chi_vals, [0.5, 99.5])
        chi_edges = np.linspace(float(chi_min), float(chi_max), n_chi + 1)
    else:
        chi_edges = np.linspace(-180.0, 180.0, n_chi + 1)
    chi_grid = 0.5 * (chi_edges[:-1] + chi_edges[1:])

    accum_I = np.zeros((n_q, n_chi), dtype=float)
    accum_N = np.zeros((n_q, n_chi), dtype=float)
    frame_qchi: list[xr.Dataset] = []
    frame_iq: list[xr.Dataset] = []

    for idx in range(n_frames):
        img = images[idx].astype(float)
        if sa is not None:
            with np.errstate(divide="ignore", invalid="ignore"):
                sa_valid = (mask_use if mask_use is not None else np.ones(shape, bool)) & (sa > 0)
                img = np.where(sa_valid, img / sa, np.nan)

        valid = per_frame_valid[idx] & np.isfinite(img)
        i_hist = np.zeros((n_q, n_chi), dtype=float)
        n_hist = np.zeros((n_q, n_chi), dtype=float)
        if np.any(valid):
            i_hist, n_hist = _histogram2d_pixel_split(
                q2d, chi_deg_2d, img, valid, q_edges, chi_edges,
                pixel_splitting=pixel_splitting,
            )
        accum_I += i_hist
        accum_N += n_hist
        frame_out = _qchi_and_iq(i_hist, n_hist, q_grid, chi_grid)
        frame_qchi.append(frame_out["q_chi"])
        frame_iq.append(frame_out["iq"])

    out = _qchi_and_iq(accum_I, accum_N, q_grid, chi_grid)
    out["q_chi_frames"] = _stack_qchi_frames(frame_qchi)
    out["iq_frames"] = _stack_iq_frames(frame_iq)

    # Build detector-space xr.Dataset
    ds_images = images.astype(float)
    ds_q2d = q2d
    ds_qh2d = qh2d
    ds_qv2d = qv2d
    ds_masks = per_frame_valid

    if rotate_cw_90:
        ds_images = np.rot90(ds_images, k=-1, axes=(-2, -1))
        ds_q2d = np.rot90(ds_q2d, k=-1)
        ds_qh2d = np.rot90(ds_qh2d, k=-1)
        ds_qv2d = np.rot90(ds_qv2d, k=-1)
        ds_masks = np.rot90(ds_masks, k=-1, axes=(-2, -1))

    ds = xr.Dataset(
        {
            "intensity": (("frame", "row", "col"), ds_images),
            "q_abs": (
                ("frame", "row", "col"),
                np.repeat(ds_q2d[np.newaxis], n_frames, axis=0),
            ),
            "q_horizontal": (
                ("frame", "row", "col"),
                np.repeat(ds_qh2d[np.newaxis], n_frames, axis=0),
            ),
            "q_vertical": (
                ("frame", "row", "col"),
                np.repeat(ds_qv2d[np.newaxis], n_frames, axis=0),
            ),
            "mask": (("frame", "row", "col"), ds_masks),
        },
        coords={"frame": np.arange(n_frames, dtype=int)},
    )
    out["ds"] = ds
    return out


# ===================================================================
# WAXS integration
# ===================================================================

def integrate_waxs(
    waxs_raw: xr.DataArray,
    mask_fn,
    n_q: int = 1000,
    n_chi: int = 360,
    cal: WAXSCalibration | None = None,
    solid_angle_correction: bool = False,
    flip_horizontal: bool = False,
    qx_shift_nm: float = 0.0,
    qy_shift_nm: float = 0.0,
    dezinger_threshold: float | None = None,
    dezinger_kernel: int = 5,
    cache_geometry: bool = True,
    pixel_splitting: int = 1,
) -> dict[str, Any]:
    """WAXS reduction via MultiPanelArcDetector per arc-angle frame."""
    attrs = waxs_raw.attrs
    if cal is None:
        cal = WAXSCalibration(**_DEFAULT_CAL)

    images = np.asarray(waxs_raw.values, dtype=float)
    if images.ndim == 2:
        images = images[np.newaxis, :, :]
    arc_angles = np.asarray(
        waxs_raw.coords[waxs_raw.dims[0]].values, dtype=float
    )
    bsx_per_frame = np.asarray(
        attrs.get("smi_waxs_bsx_per_frame", [0.0] * images.shape[0]),
        dtype=float,
    )

    # Per-frame energy (eV) — used for wavelength in q-map computation
    energy_per_frame_ev = np.asarray(
        attrs.get("smi_energy_per_frame_ev", [cal.energy_kev * 1000.0] * images.shape[0]),
        dtype=float,
    )

    img_0_rot, _ = rotate_image_and_mask(images[0], k=cal.rotation_k)
    rot_shape = img_0_rot.shape

    def build_detector_for_angle(theta_deg: float, wavelength_nm: float | None = None):
        bc = cal.beam_center_at_angle(float(theta_deg))
        wl = wavelength_nm if wavelength_nm is not None else cal.wavelength_nm
        return MultiPanelArcDetector(
            image_shape=rot_shape,
            panel_specs=cal.make_panel_specs(),
            wavelength_nm=wl,
            pixel_size_mm=cal.pixel_size_mm,
            sample_distance_mm=cal.sample_distance_mm,
            beam_center_px=bc,
            theta_zero_deg=cal.theta_zero_deg,
            sample_offset_x_mm=cal.sample_offset_x_mm,
            sample_offset_z_mm=cal.sample_offset_z_mm,
        )

    # Pre-compute global q/chi range across all angles
    _all_q_min, _all_q_max = [], []

    # Use module-level persistent cache if requested
    _cache_key = None
    if cache_geometry:
        _cache_key = _waxs_cache_key(cal, rot_shape, flip_horizontal, qx_shift_nm, qy_shift_nm)
        _geo_cache = _WAXS_GEOMETRY_CACHE.setdefault(_cache_key, {})
    else:
        _geo_cache = {}

    # Determine per-frame wavelength (nm) from energy
    _HC_EV_NM = 1239.84198  # eV·nm
    wavelength_per_frame_nm = _HC_EV_NM / energy_per_frame_ev

    for fi, theta_val in enumerate(arc_angles):
        theta_f = float(theta_val)
        wl_nm = float(wavelength_per_frame_nm[fi])
        # Cache key includes both arc angle and wavelength
        key = (round(theta_f, 6), round(wl_nm, 8))
        if key in _geo_cache:
            # Still need q-range info even from cached entries
            qabs_px = _geo_cache[key][0]
            chi_px = _geo_cache[key][3]
            finite = np.isfinite(qabs_px) & np.isfinite(chi_px)
            if finite.any():
                _all_q_min.append(float(np.nanmin(qabs_px[finite])))
                _all_q_max.append(float(np.nanmax(qabs_px[finite])))
            continue
        det = build_detector_for_angle(theta_f, wavelength_nm=wl_nm)
        qds = det.qmap(theta_f)
        qx_px = cal.q_horizontal_sign * np.asarray(qds["qx"].values, dtype=float)
        qy_px = cal.q_vertical_sign * np.asarray(qds["qy"].values, dtype=float)
        qabs_px = np.asarray(qds["qabs"].values, dtype=float)
        sa_px = np.asarray(qds["solid_angle"].values, dtype=float)
        if flip_horizontal:
            qx_px = np.fliplr(qx_px)
            qy_px = np.fliplr(qy_px)
            qabs_px = np.fliplr(qabs_px)
            sa_px = np.fliplr(sa_px)
        qx_px += qx_shift_nm
        qy_px += qy_shift_nm
        chi_px = np.rad2deg(np.arctan2(qx_px, qy_px))
        _geo_cache[key] = (qabs_px, qx_px, qy_px, chi_px, sa_px)

        finite = np.isfinite(qabs_px) & np.isfinite(chi_px)
        if finite.any():
            _all_q_min.append(float(np.nanmin(qabs_px[finite])))
            _all_q_max.append(float(np.nanmax(qabs_px[finite])))

    if _all_q_min:
        q_edges = np.linspace(min(_all_q_min), max(_all_q_max), n_q + 1)
    else:
        q_edges = np.linspace(0, 20, n_q + 1)
    q_grid = 0.5 * (q_edges[:-1] + q_edges[1:])
    chi_edges = np.linspace(-180.0, 180.0, n_chi + 1)
    chi_grid = 0.5 * (chi_edges[:-1] + chi_edges[1:])

    accum_I = np.zeros((n_q, n_chi), dtype=float)
    accum_N = np.zeros((n_q, n_chi), dtype=float)
    frame_qchi: list[xr.Dataset] = []
    frame_iq: list[xr.Dataset] = []

    ds_int_frames, ds_qabs_frames = [], []
    ds_qh_frames, ds_qv_frames, ds_mask_frames = [], [], []

    for fi, theta in enumerate(arc_angles):
        theta_f = float(theta)
        wl_nm = float(wavelength_per_frame_nm[fi])
        img_raw = images[fi]
        bsx = float(bsx_per_frame[fi]) if fi < len(bsx_per_frame) else 0.0
        img_rot, _ = rotate_image_and_mask(img_raw, k=cal.rotation_k)

        mask_rot = None
        if mask_fn is not None:
            try:
                mask_rot = mask_fn(img_raw.shape, theta_f, bsx)
            except Exception as exc:
                warnings.warn(
                    f"mask_fn failed for frame {fi}: {exc}", stacklevel=2
                )

        # Dezinger: flag hot pixels on the rotated image
        if dezinger_threshold is not None:
            dz_mask = dezinger(img_rot, kernel_size=dezinger_kernel,
                               threshold=dezinger_threshold)
            if mask_rot is not None:
                mask_rot = mask_rot & dz_mask
            else:
                mask_rot = dz_mask

        if flip_horizontal:
            img_rot = np.fliplr(img_rot)
            if mask_rot is not None:
                mask_rot = np.fliplr(mask_rot)

        geo_key = (round(theta_f, 6), round(wl_nm, 8))
        qabs_px, qx_px, qy_px, chi_px, sa_px = _geo_cache[geo_key]

        if solid_angle_correction:
            with np.errstate(divide="ignore", invalid="ignore"):
                valid_sa = (
                    (mask_rot if mask_rot is not None else np.ones(img_rot.shape, bool))
                    & np.isfinite(sa_px)
                    & (sa_px > 0)
                )
                img_rot = np.where(valid_sa, img_rot / sa_px, np.nan)

        valid = np.isfinite(qabs_px) & np.isfinite(chi_px) & np.isfinite(img_rot)
        if mask_rot is not None:
            valid &= mask_rot

        mask_use = valid if mask_rot is None else mask_rot
        ds_int_frames.append(np.where(mask_use, img_rot, np.nan))
        ds_qabs_frames.append(qabs_px)
        ds_qh_frames.append(qx_px)
        ds_qv_frames.append(qy_px)
        ds_mask_frames.append(mask_use.astype(bool))

        q_sel = qabs_px[valid].ravel()
        chi_sel = chi_px[valid].ravel()
        I_sel = img_rot[valid].ravel()

        I_hist, N_hist = _histogram2d_pixel_split(
            qabs_px, chi_px, img_rot, valid, q_edges, chi_edges,
            pixel_splitting=pixel_splitting,
        )
        accum_I += I_hist
        accum_N += N_hist
        frame_out = _qchi_and_iq(I_hist, N_hist, q_grid, chi_grid)
        frame_qchi.append(frame_out["q_chi"])
        frame_iq.append(frame_out["iq"])

    out = _qchi_and_iq(accum_I, accum_N, q_grid, chi_grid)
    out["q_chi_frames"] = _stack_qchi_frames(frame_qchi)
    out["iq_frames"] = _stack_iq_frames(frame_iq)
    out["ds"] = xr.Dataset(
        {
            "intensity": (
                ("frame", "row", "col"),
                np.asarray(ds_int_frames, dtype=float),
            ),
            "q_abs": (
                ("frame", "row", "col"),
                np.asarray(ds_qabs_frames, dtype=float),
            ),
            "q_horizontal": (
                ("frame", "row", "col"),
                np.asarray(ds_qh_frames, dtype=float),
            ),
            "q_vertical": (
                ("frame", "row", "col"),
                np.asarray(ds_qv_frames, dtype=float),
            ),
            "mask": (
                ("frame", "row", "col"),
                np.asarray(ds_mask_frames, dtype=bool),
            ),
            "waxs_arc": (("frame",), np.asarray(arc_angles, dtype=float)),
        },
        coords={"frame": np.arange(len(arc_angles), dtype=int)},
    )
    return out


# ===================================================================
# Combined reduction entry point
# ===================================================================

def reduce_smi_combined(
    uid: str,
    tiled_uri: str = "https://tiled.nsls2.bnl.gov",
    catalog: str = "smi/migration",
    n_q: int = 1000,
    n_chi: int = 360,
    solid_angle_correction: bool = False,
    saxs_mask_path: str | Path | None = None,
    waxs_mask_path: str | Path | None = None,
    saxs_kwargs: dict[str, Any] | None = None,
    waxs_kwargs: dict[str, Any] | None = None,
    backend_options: dict[str, Any] | None = None,
    geometry: str = "transmission",
    incident_angle_deg: float = 0.0,
    saxs_beam_delta_px: Tuple[float, float] | None = None,
    waxs_beam_delta_px: Tuple[float, float] | None = None,
    saxs_distance_delta_mm: float | None = None,
    saxs_q_cutoff: float | None = None,
    saxs_agbh_ring_order: int = 5,
    saxs_q_margin_fraction: float = 0.01,
    dezinger_threshold: float | None = 3000.0,
    dezinger_kernel: int = 5,
    waxs_beam_col_per_arc_deg: float = 0.0,
    cache_geometry: bool = True,
    pixel_splitting: int = 1,
) -> CombinedReductionResult:
    """
    Full SAXS + WAXS reduction pipeline.

    Parameters
    ----------
    uid : str
        Tiled run UID.
    tiled_uri, catalog : str
        Tiled connection parameters.
    n_q, n_chi : int
        Output grid dimensions.
    solid_angle_correction : bool
        Apply solid-angle correction to intensities.
    saxs_mask_path, waxs_mask_path : str or Path, optional
        JSON mask specification files. ``None`` (the default) uses the
        bundled SMI default masks shipped with PyHyperScattering
        (see ``PyHyperScattering.smi_defaults``). Pass an explicit path
        to override.
    saxs_kwargs, waxs_kwargs : dict, optional
        Extra options passed to SAXS / WAXS integrators.
    backend_options : dict, optional
        Options: ``saxs_rotate_cw_90``, ``waxs_flip_horizontal``,
        ``waxs_qx_shift_nm``, ``waxs_qy_shift_nm``.
    geometry : str
        ``'transmission'`` or ``'grazing_incidence'``.
    incident_angle_deg : float
        Incident angle for GI geometry.
    saxs_beam_delta_px : (delta_row, delta_col) or None
        Additive correction to the SAXS beam center read from metadata.
        None uses the built-in defaults from SMISWAXSLoader.
    waxs_beam_delta_px : (delta_row, delta_col) or None
        Additive correction to the WAXS beam center read from metadata.
        None uses the built-in defaults from SMISWAXSLoader.
    saxs_distance_delta_mm : float or None
        Additive correction to the SAXS sample-detector distance read from
        the motor (mm). None uses the built-in default from SMISWAXSLoader.
    saxs_q_cutoff : float or None
        Explicit SAXS q cutoff in nm⁻¹. Overrides silver-behenate-based
        auto-calculation. None (default) uses silver behenate rings.
    saxs_agbh_ring_order : int
        Silver behenate ring order used for auto q cutoff (default 5).
        Ignored if ``saxs_q_cutoff`` is set.
    saxs_q_margin_fraction : float
        Fractional margin above the selected AgBh ring for q cutoff
        (default 0.08, i.e. 8%). Ignored if ``saxs_q_cutoff`` is set.
    dezinger_threshold : float or None
        Sigma threshold for median-filter hot-pixel rejection. Applied to
        both SAXS and WAXS per frame. None (default) disables dezingering.
    dezinger_kernel : int
        Kernel size for the dezinger median filter (default 5).
    cache_geometry : bool
        If True (default), cache precomputed q-maps in a module-level dict
        so that subsequent calls with the same geometry parameters skip the
        expensive pixel-position trigonometry.  Safe across scans that share
        calibration.  Call :func:`clear_geometry_cache` to free memory or
        after programmatically changing calibration parameters.
    pixel_splitting : int
        Number of sub-pixel divisions per axis for fractional pixel
        splitting during histogram binning.  1 (default) disables splitting
        (each pixel contributes to a single bin).  Values > 1 subdivide each
        pixel into an NxN grid and distribute intensity fractionally across
        bins using gradient-based interpolation of the q/chi maps.  Typical
        values are 2–4.

    Returns
    -------
    CombinedReductionResult
    """
    import time as _time
    from PyHyperScattering.SMISWAXSLoader import (
        TiledSMISWAXSLoader,
        clear_baseline_cache,
        infer_detectors_and_steps,
        resolve_saxs_geometry,
        resolve_waxs_geometry,
    )

    saxs_kw = dict(saxs_kwargs or {})
    waxs_kw = dict(waxs_kwargs or {})
    opts = dict(backend_options or {})
    t0 = _time.perf_counter()

    # Load raw data — reuse a single loader (and its tiled session) for
    # everything so we don't call from_uri / authenticate twice.
    loader = TiledSMISWAXSLoader(tiled_uri=tiled_uri, catalog=catalog)
    run = loader._get_run(uid)

    # Avoid run["primary"].read() — that pulls every variable in the primary
    # stream including the multi-frame detector arrays, which can trigger
    # an HTTP 500 from the tiled backend.  infer_detectors_and_steps now
    # introspects the tiled containers directly.
    scan_info = infer_detectors_and_steps(run, None)

    saxs_raw = loader.loadSingleImage(uid, detector="saxs")
    waxs_raw = loader.loadSingleImage(uid, detector="waxs")
    has_saxs = saxs_raw is not None
    has_waxs = waxs_raw is not None
    t_load = _time.perf_counter()

    # -- SAXS branch --
    saxs_result: dict[str, Any] | None = None
    saxs_geo = None
    t_saxs_start = t_saxs_end = _time.perf_counter()
    if has_saxs:
        _saxs_geo_kw: dict[str, Any] = {}
        if saxs_beam_delta_px is not None:
            _saxs_geo_kw["beam_delta_row_px"] = saxs_beam_delta_px[0]
            _saxs_geo_kw["beam_delta_col_px"] = saxs_beam_delta_px[1]
        if saxs_distance_delta_mm is not None:
            _saxs_geo_kw["distance_delta_mm"] = saxs_distance_delta_mm
        saxs_geo = resolve_saxs_geometry(run, **_saxs_geo_kw)

        # Update saxs_raw attrs with corrected geometry
        _pixel1 = float(saxs_raw.attrs["pixel1"])
        _pixel2 = float(saxs_raw.attrs["pixel2"])
        new_attrs = dict(saxs_raw.attrs)
        new_attrs["poni1"] = saxs_geo.beam_center_row_px * _pixel1
        new_attrs["poni2"] = saxs_geo.beam_center_col_px * _pixel2
        new_attrs["dist"] = saxs_geo.dist_m
        saxs_raw.attrs.update(new_attrs)

        # SAXS mask
        if saxs_mask_path is None:
            saxs_mask_path = saxs_kw.pop("mask_path", None)
        from PyHyperScattering.smi_defaults import resolve_mask_path
        saxs_mask_path = resolve_mask_path(saxs_mask_path, detector="saxs")
        saxs_mask = None
        if saxs_mask_path is not None:
            saxs_mask = make_saxs_mask_from_spec(
                image_shape=saxs_raw.shape[-2:],
                mask_path=saxs_mask_path,
                active_beamstop=saxs_geo.active_beamstop,
                beamstop_pos_mm=saxs_geo.beamstop_pos_mm,
            )

        # Integrate SAXS
        t_saxs_start = _time.perf_counter()
        _dyn_kw = dict(saxs_kw.get("dynamic_saxs_kwargs") or {})
        _ap = dict(_dyn_kw.pop("aperture", {}) or {})
        _ap.setdefault("agbh_ring_order", saxs_agbh_ring_order)
        _ap.setdefault("q_margin_fraction", saxs_q_margin_fraction)
        if saxs_q_cutoff is not None:
            _ap["q_cutoff"] = saxs_q_cutoff
        _dyn_kw["aperture"] = _ap

        saxs_result = integrate_saxs(
            saxs_raw=saxs_raw,
            mask=saxs_mask,
            n_q=n_q,
            n_chi=n_chi,
            solid_angle_correction=solid_angle_correction,
            rotate_cw_90=bool(opts.get("saxs_rotate_cw_90", False)),
            beam_center_col_px=saxs_geo.beam_center_col_px,
            dynamic_saxs_mask=bool(saxs_kw.get("dynamic_saxs_mask", False)),
            dynamic_saxs_kwargs=_dyn_kw,
            dezinger_threshold=dezinger_threshold,
            dezinger_kernel=dezinger_kernel,
            cache_geometry=cache_geometry,
            pixel_splitting=pixel_splitting,
        )
        t_saxs_end = _time.perf_counter()

    # -- WAXS branch --
    waxs_result: dict[str, Any] | None = None
    t_waxs_start = t_waxs_end = _time.perf_counter()
    if has_waxs:
        _waxs_geo_kw: dict[str, Any] = {}
        if waxs_beam_delta_px is not None:
            _waxs_geo_kw["beam_delta_row_px"] = waxs_beam_delta_px[0]
            _waxs_geo_kw["beam_delta_col_px"] = waxs_beam_delta_px[1]
        waxs_geo = resolve_waxs_geometry(run, **_waxs_geo_kw)

        # WAXS mask callable
        waxs_mask_fn = None
        if waxs_mask_path is None:
            waxs_mask_path = waxs_kw.pop("mask_path", None)
        from PyHyperScattering.smi_defaults import resolve_mask_path
        waxs_mask_path = resolve_mask_path(waxs_mask_path, detector="waxs")
        if waxs_mask_path is not None:
            waxs_bsx_pf = np.asarray(
                waxs_raw.attrs.get("smi_waxs_bsx_per_frame", []),
                dtype=float,
            )
            # waxs_bsx_ref is the bsx position where the mask polygon was
            # drawn (typically arc ≈ 0°).  If the scan started at a different
            # arc angle the first-frame bsx will be offset and we must NOT
            # use it as the reference.  Prefer an explicit value from
            # waxs_kwargs; fall back to computing the arc-0 bsx from the
            # known linear bsx-vs-arc relationship (~-4.4 mm/deg at SMI).
            _BSX_PER_ARC_DEG = -4.39  # mm/deg, SMI mechanical linkage
            if "waxs_bsx_ref" in waxs_kw:
                waxs_bsx_ref = float(waxs_kw.pop("waxs_bsx_ref"))
            elif waxs_bsx_pf.size >= 2:
                arc_pf = np.asarray(
                    waxs_raw.coords[waxs_raw.dims[0]].values, dtype=float
                )
                if arc_pf.shape[0] == waxs_bsx_pf.shape[0] and (arc_pf.max() - arc_pf.min()) > 0.5:
                    # Arc was scanned — fit slope and extrapolate to arc=0
                    slope = np.polyfit(arc_pf, waxs_bsx_pf, 1)[0]
                    waxs_bsx_ref = float(waxs_bsx_pf[0] - slope * arc_pf[0])
                else:
                    # Fixed arc with multiple frames — use known slope
                    arc_val = float(arc_pf[0])
                    waxs_bsx_ref = float(
                        waxs_bsx_pf[0] - _BSX_PER_ARC_DEG * arc_val
                    )
            elif waxs_bsx_pf.size == 1:
                # Single-frame fixed arc — use known slope
                arc_val = float(
                    waxs_raw.coords[waxs_raw.dims[0]].values[0]
                )
                waxs_bsx_ref = float(
                    waxs_bsx_pf[0] - _BSX_PER_ARC_DEG * arc_val
                )
            else:
                waxs_bsx_ref = 0.0
            waxs_mask_fn = make_waxs_mask_callable(
                waxs_mask_path,
                waxs_bsx_ref=waxs_bsx_ref,
                beamstop_max_abs_arc_deg=waxs_kw.pop(
                    "beamstop_max_abs_arc_deg", 15.0
                ),
            )

        # Build WAXS calibration
        cal_dict: dict[str, Any] = dict(_DEFAULT_CAL)
        cal_dict["beam_center_row"] = waxs_geo.beam_center_row_px
        cal_dict["beam_center_col"] = waxs_geo.beam_center_col_px
        cal_dict["energy_kev"] = waxs_geo.energy_ev / 1000.0
        cal_dict["sample_distance_mm"] = waxs_geo.dist_m * 1000.0
        if waxs_beam_col_per_arc_deg != 0:
            cal_dict["beam_col_per_arc_deg"] = waxs_beam_col_per_arc_deg
        cal_override_keys = set(WAXSCalibration.__dataclass_fields__.keys())
        for k in list(waxs_kw.keys()):
            if k in cal_override_keys:
                cal_dict[k] = waxs_kw.pop(k)
        waxs_cal = WAXSCalibration(**cal_dict)

        t_waxs_start = _time.perf_counter()
        waxs_result = integrate_waxs(
            waxs_raw=waxs_raw,
            mask_fn=waxs_mask_fn,
            n_q=n_q,
            n_chi=n_chi,
            cal=waxs_cal,
            solid_angle_correction=solid_angle_correction,
            flip_horizontal=bool(opts.get("waxs_flip_horizontal", False)),
            qx_shift_nm=float(opts.get("waxs_qx_shift_nm", 0.0)),
            qy_shift_nm=float(opts.get("waxs_qy_shift_nm", 0.0)),
            dezinger_threshold=dezinger_threshold,
            dezinger_kernel=dezinger_kernel,
            cache_geometry=cache_geometry,
            pixel_splitting=pixel_splitting,
        )
        t_waxs_end = _time.perf_counter()

    t_mask = _time.perf_counter()

    # Merge (handles None gracefully)
    t_merge_start = _time.perf_counter()
    saxs_qchi = saxs_result["q_chi"] if saxs_result else None
    waxs_qchi = waxs_result["q_chi"] if waxs_result else None
    saxs_iq = saxs_result["iq"] if saxs_result else None
    waxs_iq = waxs_result["iq"] if waxs_result else None

    merged_qchi = merge_q_chi_weighted(saxs_qchi, waxs_qchi, n_q=n_q, n_chi=n_chi)
    merged_iq = merge_iq_profiles(merged_qchi, saxs_iq, waxs_iq)
    per_frame_iq = _build_per_frame_iq(merged_iq, saxs_result, waxs_result, scan_info=scan_info)
    t_merge_end = _time.perf_counter()

    timing = {
        "total": t_merge_end - t0,
        "tiled_load": t_load - t0,
        "mask_setup": t_mask - t_load,
        "saxs_integrate": t_saxs_end - t_saxs_start,
        "waxs_integrate": t_waxs_end - t_waxs_start,
        "merge": t_merge_end - t_merge_start,
    }

    # Free cached baseline data for this run
    clear_baseline_cache()

    return CombinedReductionResult(
        uid=uid,
        scan_info=scan_info,
        saxs=saxs_result,
        waxs=waxs_result,
        merged_qchi=merged_qchi,
        merged_iq=merged_iq,
        per_frame_iq=per_frame_iq,
        timing=timing,
        geometry=geometry,
        incident_angle_deg=incident_angle_deg,
    )
