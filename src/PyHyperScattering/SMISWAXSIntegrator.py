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
        return (float(self.beam_center_row), float(self.beam_center_col))


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
        dx_px = (waxs_bsx - waxs_bsx_ref) / pixel_size_mm
        polys.append(shift_polygon(beamstop_region, dx_px=dx_px, dy_px=0.0))
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
    beamstop_max_abs_arc_deg: float | None = 6.0,
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
    "q_margin_fraction": 0.08,
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


def merge_q_chi_weighted(
    saxs_qchi: xr.Dataset,
    waxs_qchi: xr.Dataset,
    n_q: int = 1000,
    n_chi: int = 360,
) -> xr.Dataset:
    """Merge SAXS and WAXS q-chi maps on a common grid with count-weighting."""
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

    s_I_interp = np.full((n_q, n_chi), np.nan, dtype=float)
    s_N_interp = np.zeros((n_q, n_chi), dtype=float)
    w_I_interp = np.full((n_q, n_chi), np.nan, dtype=float)
    w_N_interp = np.zeros((n_q, n_chi), dtype=float)

    for j in range(saxs_I.shape[1]):
        idx = min(j, len(chi_grid) - 1)
        s_I_interp[:, idx] = _interp_axis(saxs_q, saxs_I[:, j], q_grid, np.nan)
        s_N_interp[:, idx] = _interp_axis(saxs_q, saxs_N[:, j], q_grid, 0.0)
    for j in range(waxs_I.shape[1]):
        idx = min(j, len(chi_grid) - 1)
        w_I_interp[:, idx] = _interp_axis(waxs_q, waxs_I[:, j], q_grid, np.nan)
        w_N_interp[:, idx] = _interp_axis(waxs_q, waxs_N[:, j], q_grid, 0.0)

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
    merged_qchi: xr.Dataset,
    saxs_iq: xr.Dataset,
    waxs_iq: xr.Dataset,
) -> xr.Dataset:
    """Produce merged I(q) by azimuthal integration of the merged q-chi map."""
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

    saxs_q_src = np.asarray(saxs_iq["q"].values, dtype=float)
    saxs_I_src = np.asarray(saxs_iq["I"].values, dtype=float)
    waxs_q_src = np.asarray(waxs_iq["q"].values, dtype=float)
    waxs_I_src = np.asarray(waxs_iq["I"].values, dtype=float)

    saxs_I_interp = _interp_axis(saxs_q_src, saxs_I_src, q_grid, np.nan)
    waxs_I_interp = _interp_axis(waxs_q_src, waxs_I_src, q_grid, np.nan)

    return xr.Dataset(
        {
            "I": ("q", I_1d),
            "counts": ("q", total_N),
            "saxs_I": ("q", saxs_I_interp),
            "waxs_I": ("q", waxs_I_interp),
        },
        coords={"q": q_grid},
    )


# ===================================================================
# Result dataclass
# ===================================================================

@dataclass(frozen=True)
class CombinedReductionResult:
    uid: str
    scan_info: dict[str, Any]
    saxs: dict[str, Any]
    waxs: dict[str, Any]
    merged_qchi: xr.Dataset
    merged_iq: xr.Dataset
    timing: dict[str, float] | None = None
    geometry: str = "transmission"
    incident_angle_deg: float = 0.0


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

    if solid_angle_correction:
        pixel_area_m2 = pixel1_m * pixel2_m
        with np.errstate(invalid="ignore", divide="ignore"):
            sa = pixel_area_m2 * np.maximum(dist_m, 0.0) / (r_m**3)
    else:
        sa = None

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
            q_sel = q2d[valid].ravel()
            chi_sel = chi_deg_2d[valid].ravel()
            i_sel = img[valid].ravel()
            i_hist, _, _ = np.histogram2d(
                q_sel, chi_sel, bins=[q_edges, chi_edges], weights=i_sel
            )
            n_hist, _, _ = np.histogram2d(
                q_sel, chi_sel, bins=[q_edges, chi_edges]
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

    img_0_rot, _ = rotate_image_and_mask(images[0], k=cal.rotation_k)
    rot_shape = img_0_rot.shape

    def build_detector_for_angle(theta_deg: float):
        bc = cal.beam_center_at_angle(float(theta_deg))
        return MultiPanelArcDetector(
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

    # Pre-compute global q/chi range across all angles
    _all_q_min, _all_q_max = [], []
    _geo_cache: dict[float, tuple] = {}

    for theta_val in arc_angles:
        theta_f = float(theta_val)
        key = round(theta_f, 6)
        if key in _geo_cache:
            continue
        det = build_detector_for_angle(theta_f)
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

        geo_key = round(theta_f, 6)
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

        I_hist, _, _ = np.histogram2d(
            q_sel, chi_sel, bins=[q_edges, chi_edges], weights=I_sel
        )
        N_hist, _, _ = np.histogram2d(
            q_sel, chi_sel, bins=[q_edges, chi_edges]
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
    saxs_q_margin_fraction: float = 0.08,
    dezinger_threshold: float | None = None,
    dezinger_kernel: int = 5,
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
        JSON mask specification files.
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

    Returns
    -------
    CombinedReductionResult
    """
    import time as _time
    from PyHyperScattering.SMISWAXSLoader import (
        TiledSMISWAXSLoader,
        infer_detectors_and_steps,
        resolve_saxs_geometry,
        resolve_waxs_geometry,
    )

    saxs_kw = dict(saxs_kwargs or {})
    waxs_kw = dict(waxs_kwargs or {})
    opts = dict(backend_options or {})
    t0 = _time.perf_counter()

    # Load raw data
    from tiled.client import from_uri

    cat = from_uri(tiled_uri)
    for part in catalog.split("/"):
        cat = cat[part]
    run = cat[uid]

    primary = run["primary"].read()
    scan_info = infer_detectors_and_steps(run, primary)

    loader = TiledSMISWAXSLoader(tiled_uri=tiled_uri, catalog=catalog)
    saxs_raw = loader.loadSingleImage(uid, detector="saxs")
    waxs_raw = loader.loadSingleImage(uid, detector="waxs")
    t_load = _time.perf_counter()

    # Resolve geometry with delta corrections applied
    _saxs_geo_kw: dict[str, Any] = {}
    if saxs_beam_delta_px is not None:
        _saxs_geo_kw["beam_delta_row_px"] = saxs_beam_delta_px[0]
        _saxs_geo_kw["beam_delta_col_px"] = saxs_beam_delta_px[1]
    if saxs_distance_delta_mm is not None:
        _saxs_geo_kw["distance_delta_mm"] = saxs_distance_delta_mm
    saxs_geo = resolve_saxs_geometry(run, **_saxs_geo_kw)

    _waxs_geo_kw: dict[str, Any] = {}
    if waxs_beam_delta_px is not None:
        _waxs_geo_kw["beam_delta_row_px"] = waxs_beam_delta_px[0]
        _waxs_geo_kw["beam_delta_col_px"] = waxs_beam_delta_px[1]
    waxs_geo = resolve_waxs_geometry(run, **_waxs_geo_kw)

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
    saxs_mask = None
    if saxs_mask_path is not None:
        saxs_mask = make_saxs_mask_from_spec(
            image_shape=saxs_raw.shape[-2:],
            mask_path=saxs_mask_path,
            active_beamstop=saxs_geo.active_beamstop,
            beamstop_pos_mm=saxs_geo.beamstop_pos_mm,
        )

    # WAXS mask callable
    waxs_mask_fn = None
    if waxs_mask_path is None:
        waxs_mask_path = waxs_kw.pop("mask_path", None)
    if waxs_mask_path is not None:
        waxs_bsx_pf = np.asarray(
            waxs_raw.attrs.get("smi_waxs_bsx_per_frame", []),
            dtype=float,
        )
        waxs_bsx_ref = float(
            waxs_kw.pop(
                "waxs_bsx_ref",
                waxs_bsx_pf[0] if waxs_bsx_pf.size else 0.0,
            )
        )
        waxs_mask_fn = make_waxs_mask_callable(
            waxs_mask_path,
            waxs_bsx_ref=waxs_bsx_ref,
            beamstop_max_abs_arc_deg=waxs_kw.pop(
                "beamstop_max_abs_arc_deg", 6.0
            ),
        )
    t_mask = _time.perf_counter()

    # Build WAXS calibration — use geometry-resolved beam center
    cal_dict: dict[str, Any] = dict(_DEFAULT_CAL)
    cal_dict["beam_center_row"] = waxs_geo.beam_center_row_px
    cal_dict["beam_center_col"] = waxs_geo.beam_center_col_px
    cal_override_keys = set(WAXSCalibration.__dataclass_fields__.keys())
    for k in list(waxs_kw.keys()):
        if k in cal_override_keys:
            cal_dict[k] = waxs_kw.pop(k)
    waxs_cal = WAXSCalibration(**cal_dict)

    # Integrate SAXS
    t_saxs_start = _time.perf_counter()
    # Merge user-supplied aperture overrides with top-level q-cutoff params
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
    )
    t_saxs_end = _time.perf_counter()

    # Integrate WAXS
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
    )
    t_waxs_end = _time.perf_counter()

    # Merge
    t_merge_start = _time.perf_counter()
    merged_qchi = merge_q_chi_weighted(
        saxs_result["q_chi"], waxs_result["q_chi"], n_q=n_q, n_chi=n_chi
    )
    merged_iq = merge_iq_profiles(
        merged_qchi, saxs_result["iq"], waxs_result["iq"]
    )
    t_merge_end = _time.perf_counter()

    timing = {
        "total": t_merge_end - t0,
        "tiled_load": t_load - t0,
        "mask_setup": t_mask - t_load,
        "saxs_integrate": t_saxs_end - t_saxs_start,
        "waxs_integrate": t_waxs_end - t_waxs_start,
        "merge": t_merge_end - t_merge_start,
    }

    return CombinedReductionResult(
        uid=uid,
        scan_info=scan_info,
        saxs=saxs_result,
        waxs=waxs_result,
        merged_qchi=merged_qchi,
        merged_iq=merged_iq,
        timing=timing,
        geometry=geometry,
        incident_angle_deg=incident_angle_deg,
    )
