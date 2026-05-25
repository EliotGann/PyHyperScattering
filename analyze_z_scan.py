"""
analyze_z_scan.py
=================
Comprehensive AGB-based geometry analysis for SMI SAXS, using the long
z-sweep scan b900e711-… (``AGB_scan_z``).  The scan moves
pil2M_motor_z through 77 values (9300 → 1700 mm) and at each step
sweeps piezo_z through 11 values (-10 → +10 mm), for 847 frames total.

Outputs (saved to /tmp/zscan_results/):

  per_frame.csv        — one row per frame:
                         motor_z, piezo_z, bc_row, bc_col, fitted ring
                         radii r_1..r_n, max visible radius (where the
                         radial profile drops below the noise floor).

  zscan_summary.png    — 4-panel summary:
                         (a) bc_col, bc_row vs motor_z (lookup table)
                         (b) SDD(motor_z) at piezo_z=0 with piezo spread
                         (c) energy estimated from multi-ring fit
                         (d) max usable q vs SDD (beam-pipe occlusion)

  lookup_table.csv     — per motor_z step:
                         motor_z, bc_col(pz=0), bc_row(pz=0), SDD(pz=0),
                         piezo_to_SDD_slope, piezo_precision_um,
                         max_q_visible_per_nm

Algorithm per frame:

  1. Find pin transmission centroid → initial beam center.
  2. Build chi-averaged radial profile in 1-px bins.
  3. scipy.signal.find_peaks against the radial profile to detect all
     visible AGB rings (multiple orders at short SDD).
  4. For the first peak (always present), do a Gaussian sub-pixel fit
     for ring radius; the higher-order rings get centroid refinement.
  5. Refine BC via least-squares circle fit on ring samples at multiple
     chi sectors (when ≥ 5 sectors usable).
  6. From ring radii r_n at known SDD: solve for SDD via Bragg + AGB
     d-spacing.  When ≥ 2 ring orders present, also extract λ.
  7. Determine max visible q from the radial profile's outer envelope.

Usage:
    PYTHONPATH=src python analyze_z_scan.py [UID]
"""
from __future__ import annotations

import sys
import time as _time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from PyHyperScattering.SMISWAXSLoader import (
    TiledSMISWAXSLoader, _read_scan_axis,
)

# Local helpers from the previous calibration script
sys.path.insert(0, str(Path(__file__).parent))
from calibrate_smi_saxs import (
    find_bright_spot, radial_profile, two_theta_agb_ring,
    fit_circle, AGB_D_NM, PILATUS_PX_MM,
)

OUT_DIR = Path("/tmp/zscan_results")
OUT_DIR.mkdir(exist_ok=True)


def find_ring_peaks(
    rs: np.ndarray, Is: np.ndarray,
    expected_r1_px: float,
    n_max_order: int = 12,
    r_match_tol: float = 0.15,
    prominence_factor: float = 0.3,
) -> list[tuple[float, float, int]]:
    """Find AGB ring peaks at integer multiples of *expected_r1_px*.

    Returns ``[(r_peak_px, intensity, order_n), …]`` sorted by order.
    Each accepted peak must lie within ``r_match_tol·expected_r1_px`` of
    an integer-multiple position (n·r1), filtering out the pin-streak
    pseudo-peaks at small radii and other detector artefacts.
    """
    from scipy.signal import find_peaks

    finite = np.isfinite(Is)
    if not finite.any() or expected_r1_px <= 0:
        return []
    Is_f = Is.copy()
    Is_f[~finite] = 0
    # Median over background-ish pixels — robust to peaks.
    background = np.median(Is_f) if (Is_f > 0).any() else 1.0
    peaks, _ = find_peaks(
        Is_f,
        prominence=prominence_factor * max(background, 1.0),
        distance=max(int(expected_r1_px * 0.5), 4),
    )
    if peaks.size == 0:
        return []

    accepted: dict[int, tuple[float, float]] = {}
    tol = r_match_tol * expected_r1_px
    for p in peaks:
        # Sub-pixel parabolic refinement
        if 0 < p < len(rs) - 1:
            y0, y1, y2 = Is_f[p - 1], Is_f[p], Is_f[p + 1]
            denom = (y0 - 2 * y1 + y2)
            shift = 0.5 * (y0 - y2) / denom if abs(denom) > 1e-9 else 0.0
            r_refined = float(rs[p] + shift)
        else:
            r_refined = float(rs[p])
        # Which integer order is this peak closest to?
        n = int(round(r_refined / expected_r1_px))
        if n < 1 or n > n_max_order:
            continue
        if abs(r_refined - n * expected_r1_px) > tol:
            continue
        intensity = float(Is_f[p])
        # Keep the strongest peak per order
        if n not in accepted or intensity > accepted[n][1]:
            accepted[n] = (r_refined, intensity)
    return [(r, i, n) for n, (r, i) in sorted(accepted.items())]


def estimate_max_visible_q(
    rs: np.ndarray, Is: np.ndarray, sdd_mm: float, wavelength_nm: float,
    rel_threshold: float = 0.01,
) -> float | None:
    """Estimate the largest radius at which the radial profile is above
    `rel_threshold` of its peak — i.e., the beam-pipe / detector aperture
    cutoff, converted to q.
    """
    finite = np.isfinite(Is) & (Is > 0)
    if not finite.any():
        return None
    Is_f = Is.copy()
    Is_f[~finite] = 0
    peak = float(Is_f.max())
    above = np.where(Is_f >= rel_threshold * peak)[0]
    if above.size == 0:
        return None
    r_max_px = float(rs[above[-1]])
    r_max_mm = r_max_px * PILATUS_PX_MM
    two_theta = np.arctan2(r_max_mm, sdd_mm)
    q_max = 4.0 * np.pi * np.sin(two_theta / 2.0) / wavelength_nm
    return float(q_max)


# The pin-diode arm casts a vertical shadow extending downward from the
# beam.  We mask it geometrically (a fixed-pixel-width vertical strip)
# rather than via a fixed-angle chi exclusion — the latter is too wide at
# small ring radii and too narrow at large ones.  The arm is ~30 px wide
# (≈ 5 mm physical) and extends ~640 px downward.
PIN_ARM_HALF_WIDTH_PX: float = 15.0


def mask_pin_arm(
    image: np.ndarray, bc_row: float, bc_col: float,
    half_width_px: float = PIN_ARM_HALF_WIDTH_PX,
) -> np.ndarray:
    """Return a copy of *image* with the pin-arm vertical strip set to NaN.

    The pin arm extends DOWN from beam center (increasing row), so we
    mask pixels where ``|col - bc_col| < half_width AND row > bc_row``.
    NaN values are treated as missing by ``radial_profile`` (histogram
    skips them, but using NaN keeps the array shape intact).
    """
    out = image.astype(float, copy=True)
    ys, xs = np.indices(image.shape)
    arm_mask = (np.abs(xs - bc_col) < half_width_px) & (ys > bc_row)
    out[arm_mask] = np.nan
    return out


def fit_ring_circles(
    image: np.ndarray, bc_row: float, bc_col: float, ring_radii_px: list[float],
    n_chi: int = 72, r_window_px: float = 12.0,
    chi_exclude_deg: tuple[tuple[float, float], ...] = (),
) -> tuple[float, float, list[float]]:
    """Refine BC and ring radii by circle-fitting samples at many chi.

    Samples *n_chi* azimuthal sectors (default 72 = every 5°) and skips
    the chi ranges in *chi_exclude_deg* (the pin-diode arm casts a strong
    vertical shadow that biases ring detection in those sectors).
    """
    ny, nx = image.shape
    chi_deg = np.linspace(-180, 180, n_chi, endpoint=False)

    def _is_excluded(angle_deg: float) -> bool:
        for lo, hi in chi_exclude_deg:
            if lo <= hi:
                if lo <= angle_deg <= hi:
                    return True
            else:
                if angle_deg >= lo or angle_deg <= hi:
                    return True
        return False

    bc_estimates_row: list[float] = []
    bc_estimates_col: list[float] = []
    refined_radii: list[float] = []
    for r_expected in ring_radii_px:
        pts = []
        for cd in chi_deg:
            if _is_excluded(cd):
                continue
            c = np.deg2rad(cd)
            rs = np.arange(r_expected - r_window_px, r_expected + r_window_px, 0.5)
            ys = bc_row + rs * np.sin(c)
            xs = bc_col + rs * np.cos(c)
            valid = (ys >= 0) & (ys < ny - 1) & (xs >= 0) & (xs < nx - 1)
            if valid.sum() < 5:
                continue
            ys_v, xs_v = ys[valid], xs[valid]
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
            if I.max() < 2.0 * max(np.median(I), 0.5):
                continue
            i_max = int(np.argmax(I))
            lo = max(0, i_max - 3)
            hi = min(len(I), i_max + 4)
            w = I[lo:hi]
            if w.sum() <= 0:
                continue
            r_centroid = float(np.sum(rs[valid][lo:hi] * w) / np.sum(w))
            pts.append((bc_row + r_centroid * np.sin(c),
                        bc_col + r_centroid * np.cos(c)))
        if len(pts) >= 5:
            arr = np.asarray(pts)
            try:
                bcr, bcc, rr = fit_circle(arr)
                bc_estimates_row.append(bcr)
                bc_estimates_col.append(bcc)
                refined_radii.append(rr)
            except Exception:
                refined_radii.append(r_expected)
        else:
            refined_radii.append(r_expected)
    if bc_estimates_row:
        return (float(np.median(bc_estimates_row)),
                float(np.median(bc_estimates_col)),
                refined_radii)
    return float(bc_row), float(bc_col), refined_radii


def analyze_one_frame(
    image: np.ndarray, sdd_hint_mm: float, wavelength_nm: float,
) -> dict:
    """Run the per-frame analysis: BC, ring peaks, max-q.  Returns a dict."""
    bc_row, bc_col = find_bright_spot(image)

    # Predict where AGB ring 1 should land at the *hinted* SDD.  This
    # anchors the peak finder (otherwise it locks onto the pin streak).
    tt1 = two_theta_agb_ring(1, wavelength_nm)
    expected_r1 = sdd_hint_mm * np.tan(tt1) / PILATUS_PX_MM

    # Mask the pin-diode arm's vertical shadow strip so it doesn't bias
    # the radial average (especially at short SDD where the first AGB
    # ring sits inside the arm length).  The geometric mask is more
    # accurate than a fixed chi-angle exclusion which over-cuts at small
    # radii and under-cuts at large.
    image_masked = mask_pin_arm(image, bc_row, bc_col)

    # Compute the radial profile out to the farthest corner.  The histogram
    # averages over whatever pixels are available at each radius — at radii
    # beyond the nearest edge, only some chi sectors contribute, but that's
    # fine.  Skip the inner zone where the pin-diode bright spot dominates.
    ny, nx = image.shape
    far_row = max(bc_row, ny - bc_row)
    far_col = max(bc_col, nx - bc_col)
    r_max = float(np.hypot(far_row, far_col)) - 5.0
    r_min = max(40.0, 0.5 * expected_r1)
    rs, Is = radial_profile(image_masked, bc_row, bc_col, r_min=r_min,
                            r_max=r_max, binsize=1.0)

    peaks = find_ring_peaks(rs, Is, expected_r1_px=expected_r1)
    ring_radii_px = [p[0] for p in peaks]
    ring_orders = [p[2] for p in peaks]

    # Refine BC via circle fit on the rings.
    if ring_radii_px:
        bc_row, bc_col, refined_radii = fit_ring_circles(
            image, bc_row, bc_col, ring_radii_px[:6],
        )
        ring_radii_px = refined_radii

    # Derive SDD from each ring's q and apparent radius:
    #    r_n_mm = SDD * tan(2θ_n),  2θ_n = 2*arcsin(n λ / 2D)
    sdds: list[float] = []
    for n, r_pk in zip(ring_orders, ring_radii_px):
        tt = two_theta_agb_ring(n, wavelength_nm)
        if np.tan(tt) > 0:
            sdds.append(r_pk * PILATUS_PX_MM / np.tan(tt))
    sdd_estimate = float(np.median(sdds)) if sdds else float("nan")

    q_max = estimate_max_visible_q(rs, Is, sdd_estimate or sdd_hint_mm,
                                    wavelength_nm)

    return {
        "bc_row": float(bc_row),
        "bc_col": float(bc_col),
        "n_rings": len(ring_radii_px),
        "ring_radii_px": ring_radii_px,
        "ring_orders": ring_orders,
        "sdd_from_rings_mm": sdd_estimate,
        "q_max_per_nm": q_max,
        "max_pixel_intensity": float(image.max()),
    }


def fetch_and_analyze(
    img_node, idx: int, sdd_hint: float, wavelength_nm: float,
) -> tuple[int, dict | None]:
    try:
        frame = np.squeeze(np.asarray(img_node[idx:idx + 1])).astype(float)
        if frame.ndim != 2:
            return idx, None
        frame[frame < 0] = 0
        res = analyze_one_frame(frame, sdd_hint, wavelength_nm)
        return idx, res
    except Exception as exc:  # noqa: BLE001
        return idx, {"error": repr(exc)}


def main(uid: str, n_workers: int = 8) -> None:
    loader = TiledSMISWAXSLoader()
    run = loader._get_run(uid)
    primary = run["primary"]
    img_node = primary["pil2M_image"]

    mz = _read_scan_axis(run, "pil2M_motor_z")
    pz = _read_scan_axis(run, "piezo_z")
    n_frames = mz.size
    print(f"Scan: {n_frames} frames")
    print(f"  motor_z: {mz.min():.1f}..{mz.max():.1f} ({len(np.unique(np.round(mz, 1)))} unique)")
    print(f"  piezo_z: {pz.min():.1f}..{pz.max():.1f} ({len(np.unique(np.round(pz, 0)))} unique)")

    energy_ev = float(np.asarray(
        run["baseline"].base["internal"]["energy_energy"].read()
    ).flat[0])
    wavelength_nm = 1.239841984 / (energy_ev / 1000.0)
    print(f"  energy: {energy_ev:.2f} eV → λ = {wavelength_nm:.5f} nm")

    # Process frames in parallel using threads (I/O-bound on tiled reads).
    results: dict[int, dict | None] = {}
    t0 = _time.perf_counter()
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        futures = {
            ex.submit(fetch_and_analyze, img_node, i,
                       float(mz[i]) - 193.0, wavelength_nm): i
            for i in range(n_frames)
        }
        done = 0
        for fut in as_completed(futures):
            idx, res = fut.result()
            results[idx] = res
            done += 1
            if done % 25 == 0 or done == n_frames:
                elapsed = _time.perf_counter() - t0
                rate = done / elapsed
                eta = (n_frames - done) / max(rate, 0.01)
                print(f"  [{done:3d}/{n_frames}] elapsed={elapsed:.1f}s "
                      f"rate={rate:.1f} fr/s eta={eta:.0f}s")

    # Build DataFrame
    rows = []
    for i in range(n_frames):
        r = results.get(i)
        if r is None or "error" in r:
            continue
        rows.append({
            "frame": i,
            "motor_z": float(mz[i]),
            "piezo_z": float(pz[i]),
            "bc_row": r["bc_row"],
            "bc_col": r["bc_col"],
            "n_rings": r["n_rings"],
            "sdd_from_rings_mm": r["sdd_from_rings_mm"],
            "q_max_per_nm": r["q_max_per_nm"],
            "ring_radii_px": ";".join(f"{x:.2f}" for x in r["ring_radii_px"]),
            "ring_orders": ";".join(str(x) for x in r["ring_orders"]),
            "max_pixel_intensity": r["max_pixel_intensity"],
        })
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "per_frame.csv", index=False)
    print(f"\nSaved per-frame CSV: {OUT_DIR / 'per_frame.csv'} "
          f"({len(df)} rows)")

    # ---------- Lookup table per motor_z step ----------
    lookup_rows: list[dict] = []
    for mz_val, sub in df.groupby(np.round(df["motor_z"], 1)):
        bc_col_at_pz0 = float(sub.loc[sub["piezo_z"].abs() < 100, "bc_col"].mean())
        bc_row_at_pz0 = float(sub.loc[sub["piezo_z"].abs() < 100, "bc_row"].mean())
        sdd_at_pz0 = float(sub.loc[sub["piezo_z"].abs() < 100,
                                    "sdd_from_rings_mm"].mean())
        # piezo→SDD slope via linear fit within this motor_z step
        valid = sub.dropna(subset=["sdd_from_rings_mm"])
        if len(valid) >= 3:
            slope, intercept = np.polyfit(
                valid["piezo_z"].values,
                valid["sdd_from_rings_mm"].values, 1,
            )
            residual = valid["sdd_from_rings_mm"].values - (
                intercept + slope * valid["piezo_z"].values
            )
            sdd_noise_mm = float(np.std(residual))
            piezo_precision_um = sdd_noise_mm / abs(slope) if abs(slope) > 1e-9 else np.nan
        else:
            slope = piezo_precision_um = np.nan
        max_q = float(sub["q_max_per_nm"].dropna().median())
        lookup_rows.append({
            "motor_z": float(mz_val),
            "bc_col_at_pz0": bc_col_at_pz0,
            "bc_row_at_pz0": bc_row_at_pz0,
            "sdd_at_pz0_mm": sdd_at_pz0,
            "piezo_to_sdd_slope": slope,
            "piezo_precision_um": piezo_precision_um,
            "max_q_visible_per_nm": max_q,
        })
    lookup = pd.DataFrame(lookup_rows).sort_values("motor_z")
    lookup.to_csv(OUT_DIR / "lookup_table.csv", index=False)
    print(f"Saved lookup table: {OUT_DIR / 'lookup_table.csv'} "
          f"({len(lookup)} motor_z steps)")

    # ---------- Energy refinement ----------
    # From rows with ≥ 2 rings, the ratios r_n/r_1 should follow Bragg.
    # But we already mapped peaks to orders n.  We can also re-fit λ
    # GIVEN the assumption that motor_z is correct (linearly).
    # SDD_true = motor_z - offset (where offset is constant)
    # r_n = SDD_true * tan(2θ_n) = SDD_true * tan(2 arcsin(n λ / (2D)))
    # For small 2θ:  r_n ≈ SDD_true * n λ / D
    # So r_n / (SDD_true * n) = λ / D  →  λ = D * r_n / (SDD_true * n)
    # Compute λ for every (n, r_n, SDD_true) and aggregate.
    lambda_estimates = []
    if not lookup.empty:
        sdd_offset = float((lookup["motor_z"] - lookup["sdd_at_pz0_mm"]).median())
        for _, r in df.iterrows():
            sdd_true = r["motor_z"] - sdd_offset + 0.001 * r["piezo_z"]  # rough piezo correction
            if not (r["ring_radii_px"] and r["ring_orders"]):
                continue
            try:
                radii = [float(x) for x in str(r["ring_radii_px"]).split(";") if x]
                orders = [int(x) for x in str(r["ring_orders"]).split(";") if x]
            except ValueError:
                continue
            for n, r_px in zip(orders, radii):
                if n <= 0:
                    continue
                r_mm = r_px * PILATUS_PX_MM
                tt = np.arctan2(r_mm, sdd_true)
                lam = 2.0 * AGB_D_NM * np.sin(tt / 2.0) / n
                lambda_estimates.append((n, lam))
        if lambda_estimates:
            lam_arr = np.array([x[1] for x in lambda_estimates])
            print(f"\nEnergy refinement (assuming motor_z linear, offset {sdd_offset:.1f} mm):")
            print(f"  λ samples: {lam_arr.size}, median = {np.median(lam_arr)*10:.5f} Å")
            E_kev = 1.239841984 / np.median(lam_arr)
            print(f"  → energy = {E_kev:.5f} keV (catalog says {energy_ev/1000:.5f} keV)")
            print(f"  λ stdev = {np.std(lam_arr)*10:.6f} Å "
                  f"({100*np.std(lam_arr)/np.median(lam_arr):.3f}%)")

    # ---------- Summary plot ----------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    ax.scatter(df["motor_z"], df["bc_col"], s=4, c=df["piezo_z"], cmap="coolwarm")
    ax.plot(lookup["motor_z"], lookup["bc_col_at_pz0"], "k-", lw=1, label="pz=0 lookup")
    ax.set_xlabel("motor_z (mm)")
    ax.set_ylabel("bc_col (px)")
    ax.set_title("Beam center column vs motor_z")
    ax.legend()
    ax_b = ax.twinx()
    ax_b.scatter(df["motor_z"], df["bc_row"], s=4, c="orange", alpha=0.3)
    ax_b.plot(lookup["motor_z"], lookup["bc_row_at_pz0"], "darkorange", lw=1)
    ax_b.set_ylabel("bc_row (px)", color="darkorange")

    ax = axes[0, 1]
    ax.scatter(df["motor_z"], df["sdd_from_rings_mm"], s=3, alpha=0.4)
    ax.plot(lookup["motor_z"], lookup["sdd_at_pz0_mm"], "r-", lw=1, label="pz=0 fit")
    ax.plot([1500, 9500], [1500 - 193, 9500 - 193], "k--", lw=0.5,
            label="motor_z - 193")
    ax.set_xlabel("motor_z (mm)")
    ax.set_ylabel("SDD from rings (mm)")
    ax.set_title("Distance from AGB ring radius")
    ax.legend()

    ax = axes[1, 0]
    ax.scatter(lookup["motor_z"], lookup["piezo_precision_um"], s=10)
    ax.set_xlabel("motor_z (mm)")
    ax.set_ylabel("piezo_z precision (μm) per step")
    ax.set_title("How precisely can piezo_z=0 be located at each SDD?")
    ax.set_yscale("log")

    ax = axes[1, 1]
    valid = lookup.dropna(subset=["max_q_visible_per_nm", "sdd_at_pz0_mm"])
    ax.scatter(valid["sdd_at_pz0_mm"], valid["max_q_visible_per_nm"], s=15)
    ax.set_xlabel("SDD (mm)")
    ax.set_ylabel("max usable q (nm⁻¹)")
    ax.set_title("Beam-pipe / aperture occlusion")
    # Annotate AGB ring lines
    for n in range(1, 12):
        q_n = 2 * np.pi * n / AGB_D_NM
        ax.axhline(q_n, color="r", alpha=0.2, ls="--")
        ax.text(valid["sdd_at_pz0_mm"].max() * 0.99, q_n, f"AGB n={n}",
                ha="right", va="bottom", fontsize=8, color="r")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "zscan_summary.png", dpi=100, bbox_inches="tight")
    print(f"Saved summary plot: {OUT_DIR / 'zscan_summary.png'}")


if __name__ == "__main__":
    uid = sys.argv[1] if len(sys.argv) > 1 else "b900e711-35a8-4dbc-8afa-2a1e20056608"
    n_workers = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    main(uid, n_workers)
