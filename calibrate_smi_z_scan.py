"""
calibrate_smi_z_scan.py
=======================
Calibrate SMI Pilatus 2M SAXS geometry as a function of detector distance
(``pil2M_motor_z``) and sample-z (``piezo_z``), using an AGB grid scan.

The reference scan ``b900e711-…`` (``AGB_scan_z``) scans:

    pil2M_motor_z ∈ [1700, 9300] mm   (~79 unique values)
    piezo_z       ∈ [-10000, +10000] μm   (11 values)
    pil2M_motor_x = 60 mm (fixed)
    pil2M_motor_y = 10 mm (fixed)

Active beamstop in this scan: ``pin`` (constants below).

For each frame we extract:

  1. Pin transmission bright spot (initial BC guess via
     :func:`calibrate_smi_saxs.find_bright_spot`).
  2. AGB ring radius from chi-averaged radial profile.
  3. Refined BC from sampling the ring at multiple chi angles and
     least-squares fitting a circle.
  4. SDD from the ring radius via Bragg: ``tan(2θ_AGB) = r_mm / SDD_mm``.
  5. Pin beamstop SHADOW centroid (NEW) — found by locating the
     low-intensity connected region around the BC.  Used to compute
     beamstop motor offsets needed to keep the beamstop centered on
     the beam.

Then we regress (model components):

    BC_col(mx, my, mz)         = a_c + bx_c · mx + by_c · my + bz_c · mz
    BC_row(mx, my, mz)         = a_r + bx_r · mx + by_r · my + bz_r · mz
    SDD(mz, pz)                = a_s + bz_s · mz + bp_s · pz
    pin_shadow_col(mx, mz)     = a_psc + bx_psc · mx + bz_psc · mz
    pin_shadow_row(my, mz)     = a_psr + by_psr · my + bz_psr · mz

This scan has motor_x, motor_y essentially constant, so their
coefficients are not well-determined here (taken from
``calibrate_smi_saxs.py`` on a complementary grid scan).

Outputs
-------

    /tmp/agb_z_calibration_results.npz
        Per-frame measurements.

    /tmp/saxs_calibration.json
        Suggested values for the loader constants + JSON override file
        that ``SMISWAXSLoader`` can read at runtime.

    /tmp/smi_beamstop_offsets.json
        Table of (motor_z) → (d_bsx, d_bsy) for the collection code to
        keep the pin beamstop centered.

Usage
-----

    PYTHONPATH=src python calibrate_smi_z_scan.py [UID]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from PyHyperScattering.SMISWAXSLoader import (
    TiledSMISWAXSLoader,
    _baseline_scalar,
    _read_scan_axis,
)

# Reuse helpers from the existing calibration script.
from calibrate_smi_saxs import (
    AGB_D_NM,
    PILATUS_PX_MM,
    find_bright_spot,
    radial_profile,
    find_agb_ring_radius,
    find_ring_points,
    fit_circle,
    two_theta_agb_ring,
)


# ---------------------------------------------------------------------------
# Pin beamstop shadow detection
# ---------------------------------------------------------------------------

def find_pin_shadow_centroid(
    image: np.ndarray,
    bc_row_hint: float,
    bc_col_hint: float,
    *,
    window: int = 60,
    pin_radius_px: float = 12.0,
) -> tuple[float, float, int]:
    """Locate the pin beamstop's shadow centroid on the detector.

    The pin is an opaque disk with a small calibrated hole that
    transmits a known fraction of the direct beam.  We see:

      - one bright cluster of pixels (the pin transmission), AND
      - a surrounding disk of near-zero intensity (the opaque body).

    The bright-spot detector (``find_bright_spot``) already finds the
    transmission; the centroid of the surrounding low-intensity disk is
    a separate quantity that tells us where the pin SHADOW sits on the
    detector.  Comparing the shadow centroid to the true beam center
    (from the AGB ring fit) reveals how far off-center the beamstop is.

    Strategy:
      1. Take a square patch around the bright-spot hint.
      2. Identify pixels with intensity below a low quantile (i.e. inside
         the pin's opaque shadow but NOT the bright transmission pixel).
      3. Restrict to pixels within ``2 × pin_radius_px`` of any seed point.
      4. Return the (weighted) centroid of those shadow pixels.

    Parameters
    ----------
    image : 2-D array
        The raw detector image.
    bc_row_hint, bc_col_hint : float
        Initial guess for the pin position (typically the bright-spot
        centroid from :func:`find_bright_spot`).
    window : int
        Search half-width in pixels (default 60).  Should comfortably
        contain the pin shadow.
    pin_radius_px : float
        Approximate pin radius in pixels; used only to mask candidate
        shadow pixels to a reasonable spatial extent around the hint.

    Returns
    -------
    (row, col, n_shadow_pixels)
        The shadow centroid in image coordinates plus how many pixels
        contributed.  Returns ``(nan, nan, 0)`` if no clear shadow is
        found.
    """
    ny, nx = image.shape
    r0 = max(0, int(bc_row_hint) - window)
    r1 = min(ny, int(bc_row_hint) + window + 1)
    c0 = max(0, int(bc_col_hint) - window)
    c1 = min(nx, int(bc_col_hint) + window + 1)
    patch = image[r0:r1, c0:c1].astype(float)
    if patch.size == 0:
        return float("nan"), float("nan"), 0

    # The opaque pin body is "very dark" relative to the local mean.
    # Use a threshold that is robust to detector-gap pixels (which can be
    # negative or 0) by demanding the patch contain reasonable signal.
    median = float(np.median(patch[patch > 0])) if (patch > 0).any() else 0.0
    if median <= 0:
        return float("nan"), float("nan"), 0

    # Shadow pixels: ≤ ~1% of local median, AND > some sentinel (negative
    # pixels are dead/gap, not shadow).
    shadow_thresh = max(median * 0.01, 1.0)
    # Build a coordinate grid for the patch
    pr_grid, pc_grid = np.indices(patch.shape)
    in_radius = (
        (pr_grid - (bc_row_hint - r0)) ** 2 + (pc_grid - (bc_col_hint - c0)) ** 2
        <= (3 * pin_radius_px) ** 2
    )
    is_shadow = (patch >= 0) & (patch < shadow_thresh) & in_radius
    n_shadow = int(is_shadow.sum())
    if n_shadow < 10:
        return float("nan"), float("nan"), n_shadow

    # Equal-weighted centroid of shadow pixels.
    pr_centroid = float(pr_grid[is_shadow].mean()) + r0
    pc_centroid = float(pc_grid[is_shadow].mean()) + c0
    return pr_centroid, pc_centroid, n_shadow


# ---------------------------------------------------------------------------
# Per-frame analysis
# ---------------------------------------------------------------------------

def analyze_frame_z(
    image: np.ndarray,
    sdd_mm_hint: float,
    wavelength_nm: float,
    refine_with_ring: bool = True,
) -> dict:
    """Find beam center, ring radius, SDD, and pin shadow on one frame.

    The pin beamstop creates a large, well-defined dark disk on the
    detector — its centroid is a robust seed for the beam center.  The
    pin transmission bright spot is NOT used as the BC seed because
    its detected centroid is sensitive to which side of the small pin
    hole the brightest pixel lands on (it can wobble by ~40 px between
    frames at different motor_z, which propagates into the ring fit).

    Flow:
      1. find_bright_spot for an initial seed (used only for the shadow
         search window).
      2. find_pin_shadow_centroid → stable BC seed.
      3. find_agb_ring_radius around the shadow → ring radius.
      4. find_ring_points + fit_circle → refined BC (the true beam
         center, independent of the pin geometry).
      5. SDD from ring radius via Bragg.

    The pin shadow centroid is also reported on its own — comparing it
    to the refined BC gives the offset that should be applied to the
    beamstop motors to keep it centered on the beam.

    Returns
    -------
    dict with keys
        bc_row_pin, bc_col_pin          — pin transmission centroid (raw seed)
        pin_shadow_row, pin_shadow_col  — opaque pin centroid (= BC seed)
        pin_shadow_n                    — number of shadow pixels found
        bc_row, bc_col                  — refined beam center (ring fit if
                                           successful, else shadow centroid)
        ring_radius_px, ring_radius_mm  — AGB ring 1 radius
        sdd_mm                          — derived from ring
        ring_fwhm_px                    — ring sharpness
        bc_method                       — 'ring_fit' or 'shadow_only'
    """
    bc_row_pin, bc_col_pin = find_bright_spot(image)

    # Step 1: pin shadow centroid (stable across motor_z).
    psr, psc, psn = find_pin_shadow_centroid(image, bc_row_pin, bc_col_pin)
    if not np.isfinite(psr) or not np.isfinite(psc):
        # Fall back to bright-spot seed if shadow detection failed.
        bc_row_seed, bc_col_seed = bc_row_pin, bc_col_pin
    else:
        bc_row_seed, bc_col_seed = psr, psc

    tt = two_theta_agb_ring(1, wavelength_nm)
    r_expected = sdd_mm_hint * np.tan(tt) / PILATUS_PX_MM

    bc_row, bc_col = bc_row_seed, bc_col_seed
    bc_method = "shadow_only"
    ring_r_px = np.nan
    ring_fwhm = np.nan

    if r_expected > 30:
        try:
            ring_r_px, ring_fwhm = find_agb_ring_radius(
                image, bc_row_seed, bc_col_seed, r_expected,
                r_search_window=min(150.0, 0.3 * r_expected),
            )
        except Exception:
            ring_r_px = np.nan
            ring_fwhm = np.nan

    if refine_with_ring and np.isfinite(ring_r_px) and ring_r_px > 20:
        try:
            pts = find_ring_points(
                image, bc_row_seed, bc_col_seed, ring_r_px,
                n_chi=36, chi_width=4.0, r_window=15.0,
            )
            if pts.shape[0] >= 6:
                yc, xc, r_fit = fit_circle(pts)
                # Sanity: refined BC should be near the shadow seed (the
                # pin is mounted within a few mm of the beam, so the
                # refined BC and shadow centroid differ only by a small
                # beamstop-centering error, typically < 30 px).
                if (abs(yc - bc_row_seed) < 50 and abs(xc - bc_col_seed) < 50
                        and abs(r_fit - ring_r_px) < 15):
                    bc_row, bc_col = yc, xc
                    ring_r_px = r_fit
                    bc_method = "ring_fit"
        except Exception:
            pass

    ring_r_mm = ring_r_px * PILATUS_PX_MM
    sdd_mm = ring_r_mm / np.tan(tt) if np.isfinite(ring_r_mm) else np.nan

    return {
        "bc_row_pin": float(bc_row_pin),
        "bc_col_pin": float(bc_col_pin),
        "bc_row": float(bc_row),
        "bc_col": float(bc_col),
        "ring_radius_px": float(ring_r_px),
        "ring_radius_mm": float(ring_r_mm),
        "sdd_mm": float(sdd_mm),
        "ring_fwhm_px": float(ring_fwhm),
        "pin_shadow_row": float(psr),
        "pin_shadow_col": float(psc),
        "pin_shadow_n": int(psn),
        "bc_method": bc_method,
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main(uid: str, max_frames: int | None = None) -> None:
    loader = TiledSMISWAXSLoader()
    run = loader._get_run(uid)
    primary = run["primary"]

    motor_x = _read_scan_axis(run, "pil2M_motor_x")
    motor_y = _read_scan_axis(run, "pil2M_motor_y")
    motor_z = _read_scan_axis(run, "pil2M_motor_z")
    piezo_z = _read_scan_axis(run, "piezo_z")
    n_frames_meta = motor_z.size

    img_node = primary["pil2M_image"]
    n_frames_img = int(img_node.shape[0])
    n_frames = min(n_frames_meta, n_frames_img)
    if max_frames is not None:
        n_frames = min(n_frames, max_frames)
    print(f"Analyzing {n_frames} frames (metadata={n_frames_meta}, "
          f"image_stack={n_frames_img})")
    print(f"  motor_z range: [{motor_z.min():.1f}, {motor_z.max():.1f}] "
          f"({len(np.unique(motor_z))} unique)")
    print(f"  piezo_z range: [{piezo_z.min():.1f}, {piezo_z.max():.1f}] "
          f"({len(np.unique(piezo_z))} unique)")

    # Energy
    energy_ev = float(_baseline_scalar(run, "energy_energy"))
    wavelength_nm = 1.239841984 / (energy_ev / 1000.0)
    print(f"  energy = {energy_ev:.2f} eV → λ = {wavelength_nm*10:.4f} Å "
          f"= {wavelength_nm:.5f} nm")

    # Active beamstop confirmation
    active_bs = _baseline_scalar(run, "pil2M_active_beamstop")
    print(f"  active beamstop: {active_bs}")
    if str(active_bs).lower() != "pin":
        print(f"  WARNING: this script assumes the pin beamstop is active "
              f"(found {active_bs!r}).  Shadow detection may misbehave.")

    print("\nProcessing frames…")
    results = []
    for i in range(n_frames):
        frame = np.squeeze(np.asarray(img_node[i:i + 1]))
        if frame.ndim != 2:
            print(f"  [{i:3d}] skipping — unexpected shape {frame.shape}")
            continue
        try:
            res = analyze_frame_z(
                frame,
                sdd_mm_hint=float(motor_z[i]),
                wavelength_nm=wavelength_nm,
            )
        except Exception as exc:
            print(f"  [{i:3d}] FAILED: {exc}")
            continue
        res["frame"] = i
        res["motor_x"] = float(motor_x[i])
        res["motor_y"] = float(motor_y[i])
        res["motor_z"] = float(motor_z[i])
        res["piezo_z"] = float(piezo_z[i])
        results.append(res)
        if i % 50 == 0 or i == n_frames - 1:
            print(f"  [{i:4d}] mz={motor_z[i]:6.0f}mm pz={piezo_z[i]:7.0f}μm "
                  f"BC=({res['bc_row']:7.2f}, {res['bc_col']:7.2f}) "
                  f"ring={res['ring_radius_px']:6.1f}px "
                  f"sdd={res['sdd_mm']:6.1f}mm "
                  f"pin_shadow=({res['pin_shadow_row']:7.2f}, "
                  f"{res['pin_shadow_col']:7.2f}) "
                  f"[{res['bc_method']}]")

    if not results:
        print("\nNo successful frames — aborting.")
        return

    print(f"\n{len(results)}/{n_frames} frames analyzed.")

    arrs = {k: np.array([r[k] for r in results
                         if isinstance(r[k], (int, float, np.integer, np.floating))])
            for k in results[0] if k != "bc_method"}

    # Filter to "good" frames where the ring fit converged
    good = np.array([r["bc_method"] == "ring_fit" for r in results])
    print(f"Ring-fit succeeded: {good.sum()}/{len(results)}")
    print(f"Pin shadow found (>=10 px): "
          f"{(np.array([r['pin_shadow_n'] for r in results]) >= 10).sum()}"
          f"/{len(results)}")

    # ----- Regressions -----
    print("\n" + "=" * 70)
    print("REGRESSIONS (motor_x and motor_y are ~constant here, so they")
    print("are held out — see calibrate_smi_saxs.py for those coefficients).")
    print("=" * 70)
    # Use the refined BC (ring fit when available, shadow centroid otherwise)
    valid_bc = (np.isfinite(arrs["bc_col"]) & np.isfinite(arrs["bc_row"])
                & good)

    def regress_mz_pz(y, valid=valid_bc):
        """Fit y = a + bz·motor_z + bp·piezo_z (excluding motor_x, motor_y)."""
        X = np.column_stack([
            np.ones(arrs["motor_z"].shape),
            arrs["motor_z"],
            arrs["piezo_z"],
        ])
        coef, *_ = np.linalg.lstsq(X[valid], y[valid], rcond=None)
        pred = X @ coef
        rms = float(np.sqrt(np.nanmean((y[valid] - pred[valid]) ** 2)))
        return coef, rms, pred

    print("\n--- Beam center (refined via ring fit) ---")
    coef_c, rms_c, _ = regress_mz_pz(arrs["bc_col"])
    coef_r, rms_r, _ = regress_mz_pz(arrs["bc_row"])
    print(f"bc_col = {coef_c[0]:9.3f} + {coef_c[1]:11.6f}·mz + {coef_c[2]:12.7f}·pz")
    print(f"  RMS = {rms_c:.3f} px ({valid_bc.sum()} frames)")
    print(f"bc_row = {coef_r[0]:9.3f} + {coef_r[1]:11.6f}·mz + {coef_r[2]:12.7f}·pz")
    print(f"  RMS = {rms_r:.3f} px ({valid_bc.sum()} frames)")
    # Mean BC (used to update _SAXS_DEFAULT_BEAM_DELTA_* if needed)
    bc_col_mean = float(np.mean(arrs["bc_col"][valid_bc]))
    bc_row_mean = float(np.mean(arrs["bc_row"][valid_bc]))
    print(f"  mean BC at this fixed (motor_x, motor_y) = "
          f"({bc_row_mean:.2f}, {bc_col_mean:.2f})")

    print("\n--- SDD from ring ---")
    valid_sdd = np.isfinite(arrs["sdd_mm"]) & good
    coef_s, rms_s, _ = regress_mz_pz(arrs["sdd_mm"], valid=valid_sdd)
    print(f"sdd_mm = {coef_s[0]:9.3f} + {coef_s[1]:9.6f}·mz + "
          f"{coef_s[2]:11.7f}·pz")
    print(f"  RMS = {rms_s:.3f} mm ({valid_sdd.sum()} frames)")
    print(f"  Implied loader constants:")
    print(f"    _SAXS_DEFAULT_DISTANCE_DELTA_MM ≈ a_s = {coef_s[0]:.3f} "
          f"(if bz≈1)")
    print(f"    _SAXS_SDD_DELTA_MM_PER_PIEZO_Z_UM ≈ bp = {coef_s[2]:.6f}")
    print(f"    (bz - 1) = {coef_s[1] - 1:+.6f} — multiplicative motor_z "
          f"correction (small ⇒ motor_z ≈ SDD)")

    print("\n--- Pin beamstop SHADOW position ---")
    arr_psr = arrs["pin_shadow_row"]
    arr_psc = arrs["pin_shadow_col"]
    valid_ps = np.isfinite(arr_psr) & np.isfinite(arr_psc)
    if valid_ps.sum() > 0:
        coef_psc, rms_psc, _ = regress_mz_pz(arr_psc, valid=valid_ps)
        coef_psr, rms_psr, _ = regress_mz_pz(arr_psr, valid=valid_ps)
        print(f"pin_shadow_col = {coef_psc[0]:9.3f} + {coef_psc[1]:11.6f}·mz + "
              f"{coef_psc[2]:12.7f}·pz")
        print(f"  RMS = {rms_psc:.3f} px")
        print(f"pin_shadow_row = {coef_psr[0]:9.3f} + {coef_psr[1]:11.6f}·mz + "
              f"{coef_psr[2]:12.7f}·pz")
        print(f"  RMS = {rms_psr:.3f} px")

        # ---- Beamstop centering offset ----
        # The "ideal" beamstop position would have the shadow centered on
        # the beam.  Offset is (beam_center - shadow_center) in pixels →
        # convert to mm via PILATUS_PX_MM and report vs motor_z.
        print("\n--- BEAMSTOP CENTERING OFFSET vs motor_z ---")
        print("(positive means: BC is to the +row/+col side of the shadow,")
        print(" so the beamstop motor should move that direction to center)")
        offset_col_px = arrs["bc_col"] - arr_psc
        offset_row_px = arrs["bc_row"] - arr_psr
        offset_col_mm = offset_col_px * PILATUS_PX_MM
        offset_row_mm = offset_row_px * PILATUS_PX_MM
        # Per unique motor_z, report mean and stddev
        unique_z = np.unique(arrs["motor_z"][valid_ps & valid_bc])
        print(f"{'motor_z(mm)':>11} {'d_col(mm)':>11} {'d_row(mm)':>11} "
              f"{'d_col(px)':>11} {'d_row(px)':>11}    N")
        for z in unique_z:
            sel = (arrs["motor_z"] == z) & valid_ps & valid_bc
            if sel.sum() == 0:
                continue
            dcm = np.nanmean(offset_col_mm[sel])
            drm = np.nanmean(offset_row_mm[sel])
            dcp = np.nanmean(offset_col_px[sel])
            drp = np.nanmean(offset_row_px[sel])
            print(f"{z:11.1f} {dcm:11.4f} {drm:11.4f} {dcp:11.2f} {drp:11.2f} "
                  f"{sel.sum():4d}")
    else:
        print("Pin shadow not found in any frames — cannot calibrate "
              "beamstop offsets.")

    # ----- Save outputs -----
    out_npz = Path("/tmp/agb_z_calibration_results.npz")
    np.savez(out_npz, **arrs)
    print(f"\nSaved per-frame measurements → {out_npz}")

    # Build calibration JSON — emits BOTH a regressions section (for
    # human inspection) and a "constants" block in the format the
    # SMISWAXSLoader.py JSON-override reader recognizes.
    motor_x_fixed = float(np.mean(arrs["motor_x"]))
    motor_y_fixed = float(np.mean(arrs["motor_y"]))
    calib = {
        "_doc": (
            "SMI SAXS calibration derived from z-grid scan "
            f"{uid}.  Generated by calibrate_smi_z_scan.py."
        ),
        "source_uid": uid,
        "energy_ev": energy_ev,
        "wavelength_nm": wavelength_nm,
        "active_beamstop": str(active_bs),
        "scan_fixed_motor_x_mm": motor_x_fixed,
        "scan_fixed_motor_y_mm": motor_y_fixed,
        "regressions": {
            "bc_col": {
                "intercept": float(coef_c[0]),
                "per_motor_z_mm": float(coef_c[1]),
                "per_piezo_z_um": float(coef_c[2]),
                "rms_px": float(rms_c),
                "_note": (
                    "Fit AT motor_x={:.3f}, motor_y={:.3f}.  motor_x/y "
                    "coefficients come from b0f165c4 grid scan."
                ).format(motor_x_fixed, motor_y_fixed),
            },
            "bc_row": {
                "intercept": float(coef_r[0]),
                "per_motor_z_mm": float(coef_r[1]),
                "per_piezo_z_um": float(coef_r[2]),
                "rms_px": float(rms_r),
            },
            "sdd_mm": {
                "intercept": float(coef_s[0]),
                "per_motor_z_mm": float(coef_s[1]),
                "per_piezo_z_um": float(coef_s[2]),
                "rms_mm": float(rms_s),
            },
        },
        # Constants block: read by SMISWAXSLoader._apply_calibration_override
        "constants": {
            "_SAXS_DEFAULT_DISTANCE_DELTA_MM": float(coef_s[0]),
            "_SAXS_BEAM_COL_PX_PER_MOTOR_Z_MM": float(coef_c[1]),
            "_SAXS_BEAM_ROW_PX_PER_MOTOR_Z_MM": float(coef_r[1]),
            "_SAXS_SDD_DELTA_MM_PER_PIEZO_Z_UM": float(coef_s[2]),
        },
    }
    out_json = Path("/tmp/saxs_calibration.json")
    out_json.write_text(json.dumps(calib, indent=2))
    print(f"Saved calibration → {out_json}")

    if valid_ps.sum() > 0:
        # Beamstop offset table keyed by motor_z
        bs_offsets = []
        z_arr_list = []
        dx_arr_list = []
        dy_arr_list = []
        for z in unique_z:
            sel = (arrs["motor_z"] == z) & valid_ps & valid_bc
            if sel.sum() == 0:
                continue
            dcm = float(np.nanmean(offset_col_mm[sel]))
            drm = float(np.nanmean(offset_row_mm[sel]))
            bs_offsets.append({
                "motor_z_mm": float(z),
                "d_bsx_mm": dcm,
                "d_bsy_mm": drm,
                "d_bsx_px": float(np.nanmean(offset_col_px[sel])),
                "d_bsy_px": float(np.nanmean(offset_row_px[sel])),
                "n_frames": int(sel.sum()),
            })
            z_arr_list.append(float(z))
            dx_arr_list.append(dcm)
            dy_arr_list.append(drm)

        # ----- Smoothing spline fit -----
        # The offset vs motor_z is not a simple linear function — there
        # are real meanders (~0.5 mm) that a linear fit would miss.  A
        # smoothing cubic B-spline tracks the structure while filtering
        # out per-z noise (residual after averaging over piezo_z).
        spline_payload = {"available": False}
        try:
            from scipy.interpolate import UnivariateSpline
            z_arr = np.asarray(z_arr_list, dtype=float)
            dx_arr = np.asarray(dx_arr_list, dtype=float)
            dy_arr = np.asarray(dy_arr_list, dtype=float)
            order = np.argsort(z_arr)
            z_s = z_arr[order]
            dx_s = dx_arr[order]
            dy_s = dy_arr[order]
            # Estimate per-z scatter to set the smoothing factor s = n*σ².
            # Use the median absolute deviation between adjacent points
            # as a rough σ estimate; this is robust to true structure.
            dx_diff = np.diff(dx_s)
            dy_diff = np.diff(dy_s)
            sigma_dx = float(1.4826 * np.median(np.abs(dx_diff - np.median(dx_diff))) / np.sqrt(2))
            sigma_dy = float(1.4826 * np.median(np.abs(dy_diff - np.median(dy_diff))) / np.sqrt(2))
            # Floor σ to avoid s=0 (interpolating spline) when the data
            # is suspiciously smooth.
            sigma_dx = max(sigma_dx, 0.05)
            sigma_dy = max(sigma_dy, 0.05)
            spl_dx = UnivariateSpline(z_s, dx_s, k=3, s=len(z_s) * sigma_dx ** 2)
            spl_dy = UnivariateSpline(z_s, dy_s, k=3, s=len(z_s) * sigma_dy ** 2)
            # Evaluate on a dense grid for the JSON consumer that can't
            # reconstruct the spline directly.
            z_dense = np.linspace(z_s[0], z_s[-1], 200)
            dx_dense = spl_dx(z_dense)
            dy_dense = spl_dy(z_dense)
            # Export the FULL B-spline knot vectors (with boundary
            # multiplicity) so the JSON consumer can rebuild the spline
            # directly via scipy.interpolate.BSpline(t, c, k).
            # UnivariateSpline.get_knots() returns boundaries once, but
            # a clamped B-spline of degree k needs them repeated k+1
            # times on each side.  We get the canonical (t, c, k) tuple
            # from the internal _eval_args attribute.
            t_dx, c_dx, k_dx = spl_dx._eval_args
            t_dy, c_dy, k_dy = spl_dy._eval_args
            spline_payload = {
                "available": True,
                "model": "scipy.interpolate.BSpline (cubic, smoothing)",
                "degree": int(k_dx),
                "sigma_dx_mm": sigma_dx,
                "sigma_dy_mm": sigma_dy,
                "knots_d_bsx_mm": [float(v) for v in t_dx],
                "coefs_d_bsx_mm": [float(v) for v in c_dx],
                "knots_d_bsy_mm": [float(v) for v in t_dy],
                "coefs_d_bsy_mm": [float(v) for v in c_dy],
                "_consumer_note": (
                    "To evaluate: from scipy.interpolate import BSpline; "
                    "f = BSpline(knots_d_bsx_mm, coefs_d_bsx_mm, degree); "
                    "d_bsx_at_z = f(motor_z_mm)."
                ),
                "dense_grid": {
                    "motor_z_mm": [float(v) for v in z_dense],
                    "d_bsx_mm":   [float(v) for v in dx_dense],
                    "d_bsy_mm":   [float(v) for v in dy_dense],
                },
            }
            # Quick RMS of raw points vs spline (sanity)
            rms_dx = float(np.sqrt(np.mean((dx_s - spl_dx(z_s)) ** 2)))
            rms_dy = float(np.sqrt(np.mean((dy_s - spl_dy(z_s)) ** 2)))
            spline_payload["rms_residual_dx_mm"] = rms_dx
            spline_payload["rms_residual_dy_mm"] = rms_dy
            print(f"\nSpline fit residuals (raw → smooth):")
            print(f"  d_bsx: RMS = {rms_dx:.4f} mm "
                  f"(σ≈{sigma_dx:.4f}, knots={len(spl_dx.get_knots())})")
            print(f"  d_bsy: RMS = {rms_dy:.4f} mm "
                  f"(σ≈{sigma_dy:.4f}, knots={len(spl_dy.get_knots())})")
        except Exception as exc:
            print(f"\nSpline fit failed: {exc}")

        bs_json = {
            "_doc": (
                "Pin beamstop centering offsets vs detector motor_z, "
                f"derived from {uid}.  d_bsx, d_bsy are the shadow "
                "displacement in mm that the beamstop motors should move "
                "to center the beamstop on the beam.  Sign convention: "
                "positive = move beamstop in the +col / +row direction "
                "on the detector.  Both per-point measurements and a "
                "smoothing cubic B-spline are provided — collection code "
                "should evaluate the spline at the current motor_z for "
                "the smoothest correction."
            ),
            "source_uid": uid,
            "active_beamstop": str(active_bs),
            "current_motor_positions": {
                "saxs_beamstop_x_pin": float(
                    _baseline_scalar(run, "saxs_beamstop_x_pin")
                ),
                "saxs_beamstop_y_pin": float(
                    _baseline_scalar(run, "saxs_beamstop_y_pin")
                ),
            },
            "offsets_by_motor_z": bs_offsets,
            "spline": spline_payload,
        }
        bs_out = Path("/tmp/smi_beamstop_offsets.json")
        bs_out.write_text(json.dumps(bs_json, indent=2))
        print(f"Saved beamstop offsets → {bs_out}")


if __name__ == "__main__":
    uid = sys.argv[1] if len(sys.argv) > 1 else "b900e711-35a8-4dbc-8afa-2a1e20056608"
    max_frames = int(sys.argv[2]) if len(sys.argv) > 2 else None
    main(uid, max_frames=max_frames)
