"""
calibrate_smi_saxs.py
=====================
Calibrate SMI Pilatus 2M SAXS geometry from an AGB grid scan.

The test scan ``b0f165c4-…`` (``AGB_scan_x_y_9m``) scans:
    pil2M_motor_x ∈ {0, 30, 60} mm   (detector lateral)
    pil2M_motor_y ∈ {-10, 0, 10} mm  (detector vertical)
    piezo_z       ∈ {-10..+10} mm    (sample along beam)

with AGB (silver behenate, D = 5.838 nm, ring 1 at q ≈ 1.076 nm⁻¹) in the
beam.  For each frame we:

  1. Locate the bright pin-diode transmission spot → initial beam center.
  2. Compute a chi-averaged radial profile around that point.
  3. Find the first AGB ring peak → ring radius (px).
  4. Refine the beam center by sampling the ring at several chi sectors
     and least-squares fitting a circle.
  5. Convert ring radius → SDD via Bragg: tan(2θ_1) = r_mm / SDD_mm.

Then we regress:

    beam_col_px = a_x + b_x · motor_x_mm
    beam_row_px = a_y + b_y · motor_y_mm
    SDD_mm      = a_z + b_z · piezo_z_um

and emit the calibration constants the loader needs:

    _SAXS_BEAM_COL_PX_PER_MOTOR_X_MM = b_x
    _SAXS_BEAM_ROW_PX_PER_MOTOR_Y_MM = b_y
    _SAXS_MOTOR_X_REF_MM = (baseline_bc_col - a_x) / b_x
    _SAXS_MOTOR_Y_REF_MM = (baseline_bc_row - a_y) / b_y
    _SAXS_SDD_DELTA_MM_PER_PIEZO_Z_UM = b_z
    _SAXS_PIEZO_Z_REF_UM = (motor_z - a_z) / b_z

Usage
-----
    PYTHONPATH=src python calibrate_smi_saxs.py [UID]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from PyHyperScattering.SMISWAXSLoader import TiledSMISWAXSLoader


# Silver behenate
AGB_D_NM = 5.838
PILATUS_PX_MM = 0.172

# Baseline EPICS calibration values (after abs).  Used to translate the fitted
# intercept back into a motor reference position.
BASELINE_BC_COL_PX = 744.0
BASELINE_BC_ROW_PX = 1107.0


def two_theta_agb_ring(order: int, wavelength_nm: float) -> float:
    """Bragg angle of the n-th AGB ring (radians).  λ in nm."""
    return 2.0 * np.arcsin(order * wavelength_nm / (2.0 * AGB_D_NM))


def find_bright_spot(image: np.ndarray, window: int = 15) -> tuple[float, float]:
    """Return (row, col) of the beam, estimated from the pin transmission.

    Strategy: locate the largest bright connected component (filters out
    scattered hot pixels and ring features), then find the peak inside
    that component and take a small (±window) intensity-weighted
    centroid around it.  The small window avoids the bias from the pin
    diode's "arm" shadow that elongates the cluster.
    """
    from scipy.ndimage import label

    img = np.where(image > 0, image, 0)
    thresh = float(np.percentile(img, 99.9))
    mask = img >= thresh
    if not mask.any():
        raise RuntimeError("no bright pixels found")

    labelled, n = label(mask, structure=np.ones((3, 3), dtype=int))
    if n == 0:
        raise RuntimeError("no connected bright region found")

    flat_lbl = labelled.ravel()
    flat_img = img.ravel().astype(float)
    sums = np.bincount(flat_lbl, weights=flat_img)
    sums[0] = 0.0
    best_lbl = int(np.argmax(sums))

    # Find brightest pixel inside the chosen cluster.
    cluster_mask = labelled == best_lbl
    masked_img = np.where(cluster_mask, img, 0)
    peak_idx = int(np.argmax(masked_img))
    pr, pc = np.unravel_index(peak_idx, img.shape)

    # ±window intensity-weighted centroid for sub-pixel precision.
    ny, nx = img.shape
    r0 = max(0, pr - window)
    r1 = min(ny, pr + window + 1)
    c0 = max(0, pc - window)
    c1 = min(nx, pc + window + 1)
    patch = img[r0:r1, c0:c1].astype(float)
    if patch.sum() <= 0:
        return float(pr), float(pc)
    yy, xx = np.indices(patch.shape)
    weights = patch
    r = r0 + float(np.sum(yy * weights) / np.sum(weights))
    c = c0 + float(np.sum(xx * weights) / np.sum(weights))
    return r, c


def radial_profile(
    image: np.ndarray,
    bc_row: float,
    bc_col: float,
    r_min: float = 50.0,
    r_max: float | None = None,
    binsize: float = 1.0,
    chi_exclude_deg: tuple[tuple[float, float], ...] = (),
) -> tuple[np.ndarray, np.ndarray]:
    """Chi-averaged radial profile: returns (r_centers, I_mean).

    Parameters
    ----------
    chi_exclude_deg : tuple of (lo, hi) ranges in degrees
        Azimuthal sectors to exclude from the average.  Uses the same
        chi convention as ``find_ring_points``: chi = atan2(dy, dx) with
        +x = increasing col, +y = increasing row.  For SMI's pin diode
        which extends downward (increasing row), exclude ``((60, 120),)``
        to remove the arm-shadow contamination.
    """
    ny, nx = image.shape
    ys, xs = np.indices(image.shape)
    dy = ys - bc_row
    dx = xs - bc_col
    r = np.sqrt(dy * dy + dx * dx)
    if r_max is None:
        r_max = float(min(bc_row, ny - bc_row, bc_col, nx - bc_col))
    bins = np.arange(r_min, r_max, binsize)
    img = image.astype(float)
    # Treat NaN as masked: such pixels contribute neither to the sum nor
    # the count.  This lets callers mask out e.g. pin-arm shadow before
    # averaging without biasing the result toward zero.
    valid = np.isfinite(img) & (img >= 0)

    if chi_exclude_deg:
        chi = np.rad2deg(np.arctan2(dy, dx))
        chi_keep = np.ones_like(chi, dtype=bool)
        for lo, hi in chi_exclude_deg:
            if lo <= hi:
                chi_keep &= ~((chi >= lo) & (chi <= hi))
            else:
                chi_keep &= ~((chi >= lo) | (chi <= hi))
        valid &= chi_keep

    r_kept = r[valid].ravel()
    img_kept = img[valid].ravel()

    sum_I, _ = np.histogram(r_kept, bins=bins, weights=img_kept)
    cnt, _ = np.histogram(r_kept, bins=bins)
    mean_I = np.where(cnt > 0, sum_I / np.maximum(cnt, 1), np.nan)
    centers = 0.5 * (bins[:-1] + bins[1:])
    return centers, mean_I


def find_agb_ring_radius(
    image: np.ndarray,
    bc_row: float,
    bc_col: float,
    r_expected: float,
    r_search_window: float = 200.0,
) -> tuple[float, float]:
    """Fit a Gaussian to the AGB ring peak in the radial profile.

    Returns (r_peak_px, fwhm_px).
    """
    r, I = radial_profile(
        image, bc_row, bc_col,
        r_min=max(50.0, r_expected - r_search_window),
        r_max=r_expected + r_search_window,
        binsize=1.0,
    )
    finite = np.isfinite(I)
    if not finite.any():
        raise RuntimeError("radial profile has no finite values")
    # Locate peak: bin with max I within the window.
    i_peak = int(np.nanargmax(I))
    r_peak = float(r[i_peak])
    # Local Gaussian fit on a small window (±20 px) for sub-pixel.
    sel = (r >= r_peak - 20) & (r <= r_peak + 20) & finite
    if sel.sum() >= 5:
        r_local = r[sel]
        I_local = I[sel] - np.nanmin(I[sel])
        # weighted centroid → sub-pixel refinement
        w = np.maximum(I_local, 0)
        if w.sum() > 0:
            r_peak = float(np.sum(r_local * w) / np.sum(w))
        fwhm = float(np.sum(w) / np.maximum(np.max(w), 1.0))
    else:
        fwhm = float("nan")
    return r_peak, fwhm


def find_ring_points(
    image: np.ndarray,
    bc_row: float,
    bc_col: float,
    r_expected: float,
    n_chi: int = 24,
    chi_width: float = 5.0,
    r_window: float = 20.0,
) -> np.ndarray:
    """Sample the AGB ring at *n_chi* azimuthal angles.

    For each chi sector, take the slice of pixels within ±chi_width° and
    within ±r_window of r_expected.  The brightest pixel in that slice is
    the ring sample point.  Returns ``(n_pts, 2)`` of (row, col), with
    sectors containing no clear peak dropped.
    """
    chi = np.deg2rad(np.linspace(-180, 180, n_chi, endpoint=False))
    pts = []
    ny, nx = image.shape
    for c in chi:
        # Polar-aligned sampling: evaluate I along a ray from BC, within
        # ±r_window.  Use bilinear interp at half-pixel sample spacing.
        rs = np.arange(r_expected - r_window, r_expected + r_window, 0.5)
        ys = bc_row + rs * np.sin(c)
        xs = bc_col + rs * np.cos(c)
        valid = (ys >= 0) & (ys < ny - 1) & (xs >= 0) & (xs < nx - 1)
        if valid.sum() < 5:
            continue
        ys_v, xs_v = ys[valid], xs[valid]
        # bilinear
        y0 = np.floor(ys_v).astype(int)
        x0 = np.floor(xs_v).astype(int)
        dy = ys_v - y0
        dx = xs_v - x0
        I = (
            image[y0, x0] * (1 - dy) * (1 - dx)
            + image[y0 + 1, x0] * dy * (1 - dx)
            + image[y0, x0 + 1] * (1 - dy) * dx
            + image[y0 + 1, x0 + 1] * (1 - dy) * dx
        )
        I = np.where(I > 0, I, 0)
        # The AGB ring on this scan is very faint; require only that the
        # peak rises ≥2× above the median of the slice, not an absolute
        # intensity threshold.
        if I.max() < 2.0 * max(np.median(I), 0.5):
            continue
        i_max = int(np.argmax(I))
        # Local centroid for sub-pixel
        lo = max(0, i_max - 3)
        hi = min(len(I), i_max + 4)
        w = I[lo:hi]
        if w.sum() <= 0:
            continue
        r_centroid = float(np.sum(rs[valid][lo:hi] * w) / np.sum(w))
        pts.append((bc_row + r_centroid * np.sin(c),
                    bc_col + r_centroid * np.cos(c)))
    return np.asarray(pts) if pts else np.empty((0, 2))


def fit_circle(pts: np.ndarray) -> tuple[float, float, float]:
    """Least-squares fit a circle to (row, col) points.  Returns (row, col, r)."""
    if pts.shape[0] < 3:
        raise RuntimeError("need ≥3 points to fit a circle")
    y = pts[:, 0]
    x = pts[:, 1]
    # Kasa's method: solve linear system for circle parameters
    A = np.column_stack([2 * x, 2 * y, np.ones_like(x)])
    b = x * x + y * y
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    xc, yc, c0 = sol
    r = float(np.sqrt(c0 + xc * xc + yc * yc))
    return float(yc), float(xc), r


def analyze_frame(image: np.ndarray, sdd_mm_hint: float, wavelength_nm: float) -> dict:
    """Find beam center (from pin transmission) + AGB ring radius for one frame.

    The pin diode transmits a fraction of the direct beam — the centroid of
    the brightest pixels is the most reliable estimator of the beam position.
    The AGB ring is faint and partially occluded by the pin's arm shadow and
    detector gaps, which means full-circle fitting is biased toward those
    structures (we observed it converging on detector-gap geometry rather
    than the ring).  So we keep the bright-spot BC and use only the chi-
    averaged radial profile to extract ring radius.

    Returns a dict with keys ``bc_row``, ``bc_col``, ``ring_radius_px``,
    ``ring_radius_mm``, ``sdd_mm`` (from ring).
    """
    bc_row, bc_col = find_bright_spot(image)

    tt = two_theta_agb_ring(1, wavelength_nm)
    r_expected = sdd_mm_hint * np.tan(tt) / PILATUS_PX_MM

    ring_r_px, fwhm = find_agb_ring_radius(image, bc_row, bc_col, r_expected,
                                           r_search_window=100.0)
    ring_r_mm = ring_r_px * PILATUS_PX_MM
    sdd_mm = ring_r_mm / np.tan(tt)
    return {
        "bc_row": float(bc_row),
        "bc_col": float(bc_col),
        "ring_radius_px": float(ring_r_px),
        "ring_radius_mm": float(ring_r_mm),
        "sdd_mm": float(sdd_mm),
        "ring_fwhm_px": float(fwhm),
    }


def main(uid: str) -> None:
    from PyHyperScattering.SMISWAXSLoader import _read_scan_axis
    loader = TiledSMISWAXSLoader()
    run = loader._get_run(uid)
    primary = run["primary"]

    # Use the loader's seq_num-aware reader so motor arrays align with the
    # chronological image stack.  (The raw primary scalar table can be in a
    # non-chronological order while the image stack is chronological.)
    motor_x = _read_scan_axis(run, "pil2M_motor_x")
    motor_y = _read_scan_axis(run, "pil2M_motor_y")
    motor_z = _read_scan_axis(run, "pil2M_motor_z")
    piezo_z = _read_scan_axis(run, "piezo_z")
    n_frames = motor_x.size
    print(f"Scan has {n_frames} frames")
    print(f"  motor_x range: [{motor_x.min():.2f}, {motor_x.max():.2f}] "
          f"(unique={len(np.unique(motor_x))})")
    print(f"  motor_y range: [{motor_y.min():.2f}, {motor_y.max():.2f}] "
          f"(unique={len(np.unique(motor_y))})")
    print(f"  motor_z range: [{motor_z.min():.2f}, {motor_z.max():.2f}] "
          f"(unique={len(np.unique(motor_z))})")
    print(f"  piezo_z range: [{piezo_z.min():.2f}, {piezo_z.max():.2f}] "
          f"(unique={len(np.unique(piezo_z))})")

    # Energy → wavelength
    energy_ev = float(np.asarray(
        run["baseline"].base["internal"]["energy_energy"].read()
    ).flat[0])
    wavelength_nm = 1.239841984 / energy_ev * 1e3  # nm·eV / eV = nm
    # Actually: λ(nm) = 1.239841984 / energy(keV); energy in eV → convert
    wavelength_nm = 1.239841984 / (energy_ev / 1000.0)
    print(f"  energy = {energy_ev:.2f} eV → λ = {wavelength_nm*10:.4f} Å "
          f"= {wavelength_nm:.5f} nm")
    # First AGB ring 2θ:
    tt = two_theta_agb_ring(1, wavelength_nm)
    print(f"  AGB ring 1 → 2θ = {np.rad2deg(tt):.4f}°")
    sdd_hint = float(motor_z.mean())
    r_hint = sdd_hint * np.tan(tt) / PILATUS_PX_MM
    print(f"  expected ring radius at SDD={sdd_hint:.0f} mm: {r_hint:.1f} px")

    # Loop frames one-by-one (reading individual frames is cheap; loading
    # all 99 at once needs ~1 GB).  We could batch but per-frame is simpler.
    img_node = primary["pil2M_image"]
    results = []
    print("\nProcessing frames…")
    for i in range(n_frames):
        frame = np.squeeze(np.asarray(img_node[i:i + 1]))
        if frame.ndim != 2:
            raise RuntimeError(f"frame {i}: unexpected shape {frame.shape}")
        try:
            res = analyze_frame(frame, sdd_mm_hint=sdd_hint,
                                wavelength_nm=wavelength_nm)
        except Exception as exc:
            print(f"  [{i:3d}] FAILED: {exc}")
            continue
        res["frame"] = i
        res["motor_x"] = float(motor_x[i])
        res["motor_y"] = float(motor_y[i])
        res["motor_z"] = float(motor_z[i])
        res["piezo_z"] = float(piezo_z[i])
        results.append(res)
        if i % 10 == 0:
            print(f"  [{i:3d}] mx={motor_x[i]:6.2f} my={motor_y[i]:6.2f} "
                  f"pz={piezo_z[i]:8.1f} → BC=({res['bc_row']:.1f}, "
                  f"{res['bc_col']:.1f}) ring={res['ring_radius_px']:.1f}px "
                  f"sdd={res['sdd_mm']:.1f}mm")

    if not results:
        print("\nNo successful frames — aborting.")
        return

    # Stack arrays
    arrs = {k: np.array([r[k] for r in results]) for k in results[0]}

    # Fit 1: beam_col = a_x + b_x * motor_x
    A_x = np.column_stack([np.ones_like(arrs["motor_x"]), arrs["motor_x"]])
    a_x, b_x = np.linalg.lstsq(A_x, arrs["bc_col"], rcond=None)[0]
    # Fit 2: beam_row = a_y + b_y * motor_y
    A_y = np.column_stack([np.ones_like(arrs["motor_y"]), arrs["motor_y"]])
    a_y, b_y = np.linalg.lstsq(A_y, arrs["bc_row"], rcond=None)[0]
    # Fit 3: sdd = a_z + b_z * piezo_z
    A_z = np.column_stack([np.ones_like(arrs["piezo_z"]), arrs["piezo_z"]])
    a_z, b_z = np.linalg.lstsq(A_z, arrs["sdd_mm"], rcond=None)[0]

    # Translate to loader constants
    motor_x_ref = (BASELINE_BC_COL_PX - a_x) / b_x if abs(b_x) > 1e-9 else None
    motor_y_ref = (BASELINE_BC_ROW_PX - a_y) / b_y if abs(b_y) > 1e-9 else None
    motor_z_baseline = float(motor_z.mean())
    piezo_z_ref = (motor_z_baseline - a_z) / b_z if abs(b_z) > 1e-9 else None

    print("\n" + "=" * 70)
    print("REGRESSION FITS")
    print("=" * 70)
    print(f"beam_col = {a_x:8.3f} + {b_x:7.4f} · motor_x_mm")
    print(f"beam_row = {a_y:8.3f} + {b_y:7.4f} · motor_y_mm")
    print(f"sdd_mm   = {a_z:8.3f} + {b_z:8.5f} · piezo_z_um")

    print("\n" + "=" * 70)
    print("LOADER CONSTANTS")
    print("=" * 70)
    print(f"# In SMISWAXSLoader.py:")
    print(f"_SAXS_MOTOR_X_REF_MM = {motor_x_ref:.4f}"
          if motor_x_ref is not None else "_SAXS_MOTOR_X_REF_MM = ?  (b_x≈0)")
    print(f"_SAXS_MOTOR_Y_REF_MM = {motor_y_ref:.4f}"
          if motor_y_ref is not None else "_SAXS_MOTOR_Y_REF_MM = ?  (b_y≈0)")
    print(f"_SAXS_PIEZO_Z_REF_UM = {piezo_z_ref:.4f}"
          if piezo_z_ref is not None else "_SAXS_PIEZO_Z_REF_UM = ?  (b_z≈0)")
    print(f"_SAXS_BEAM_COL_PX_PER_MOTOR_X_MM = {b_x:.6f}")
    print(f"_SAXS_BEAM_ROW_PX_PER_MOTOR_Y_MM = {b_y:.6f}")
    print(f"_SAXS_SDD_DELTA_MM_PER_PIEZO_Z_UM = {b_z:.6f}")
    # Also dist_delta_mm:
    print(f"# _SAXS_DEFAULT_DISTANCE_DELTA_MM should become:")
    print(f"#   a_z - motor_z_mean = {a_z - motor_z_baseline:.3f}")

    # Residuals
    res_col = arrs["bc_col"] - (a_x + b_x * arrs["motor_x"])
    res_row = arrs["bc_row"] - (a_y + b_y * arrs["motor_y"])
    res_sdd = arrs["sdd_mm"] - (a_z + b_z * arrs["piezo_z"])
    print(f"\nResiduals (RMS):")
    print(f"  bc_col: {np.sqrt(np.mean(res_col**2)):.3f} px")
    print(f"  bc_row: {np.sqrt(np.mean(res_row**2)):.3f} px")
    print(f"  sdd:    {np.sqrt(np.mean(res_sdd**2)):.3f} mm")

    # Save raw results for inspection
    out = Path("/tmp/agb_calibration_results.npz")
    np.savez(out, **arrs)
    print(f"\nSaved per-frame results to {out}")


if __name__ == "__main__":
    uid = sys.argv[1] if len(sys.argv) > 1 else "b0f165c4-203e-4d58-af17-916620b974c2"
    main(uid)
