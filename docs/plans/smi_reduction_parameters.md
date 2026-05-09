# SMI SAXS + WAXS Reduction Parameters Reference

All parameters available for `reduce_smi_combined()` and `reduce_smi_gi()` in
`PyHyperScattering.SMISWAXSIntegrator`, organized for GUI integration.

---

## Entry Point: `reduce_smi_combined()`

The main transmission-geometry pipeline. Handles SAXS, WAXS, or both.

### Connection / Data Selection

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `uid` | `str` | *(required)* | Tiled run UID |
| `tiled_uri` | `str` | `"https://tiled.nsls2.bnl.gov"` | Tiled server URL |
| `catalog` | `str` | `"smi/migration"` | Tiled catalog path |

### Integration Grid

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_q` | `int` | `1000` | Number of q bins in output |
| `n_chi` | `int` | `360` | Number of azimuthal (chi) bins in output |

### Masks

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `saxs_mask_path` | `str \| Path \| None` | `None` (bundled default) | Path to SAXS polygon-mask JSON |
| `waxs_mask_path` | `str \| Path \| None` | `None` (bundled default) | Path to WAXS polygon-mask JSON |

### SAXS Geometry Corrections

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `saxs_beam_delta_px` | `(float, float) \| None` | `None` → `(2.0, 3.0)` | Additive (row, col) correction to SAXS beam center from metadata |
| `saxs_distance_delta_mm` | `float \| None` | `None` → `-20.0` | Additive correction to SAXS detector distance (mm) |

### SAXS Q-Range / Aperture

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `saxs_q_cutoff` | `float \| None` | `None` (auto from AgBh) | Explicit max-q cutoff (nm⁻¹). Overrides silver-behenate auto-calculation |
| `saxs_agbh_ring_order` | `int` | `5` | Silver behenate ring order for auto q-cutoff |
| `saxs_q_margin_fraction` | `float` | `0.01` | Fractional margin above the AgBh ring |

### WAXS Geometry Corrections

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `waxs_beam_delta_px` | `(float, float) \| None` | `None` → `(0.0, -2.0)` | Additive (row, col) correction to WAXS beam center |
| `waxs_beam_col_per_arc_deg` | `float` | `0.0` | Beam-center column drift per degree of waxs_arc |

### Hot-Pixel Rejection (Dezingering)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dezinger_threshold` | `float \| None` | `3000.0` | Sigma threshold for median-filter outlier rejection. `None` disables |
| `dezinger_kernel` | `int` | `5` | Kernel size for the dezinger median filter |

### Intensity Corrections

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `solid_angle_correction` | `bool` | `False` | Apply solid-angle correction to pixel intensities |

### Geometry Mode

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `geometry` | `str` | `"transmission"` | `"transmission"` or `"grazing_incidence"` |
| `incident_angle_deg` | `float` | `0.0` | Incident angle for GI geometry (degrees) |

### Backend / Display Options (via `backend_options` dict)

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `saxs_rotate_cw_90` | `bool` | `False` | Rotate SAXS output 90° clockwise for display |
| `waxs_flip_horizontal` | `bool` | `False` | Flip WAXS image horizontally |
| `waxs_qx_shift_nm` | `float` | `0.0` | Global shift to WAXS q_x (nm⁻¹) |
| `waxs_qy_shift_nm` | `float` | `0.0` | Global shift to WAXS q_y (nm⁻¹) |

### Caching

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `cache_geometry` | `bool` | `True` | Cache q-maps across calls with identical geometry. Call `clear_geometry_cache()` to free memory |

### Advanced WAXS Calibration (via `waxs_kwargs` dict)

These override `WAXSCalibration` fields when passed inside `waxs_kwargs`:

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `energy_kev` | `float` | `16.1` | Photon energy (keV) |
| `sample_distance_mm` | `float` | `274.0` | Sample-to-detector distance (mm) |
| `pixel_size_mm` | `float` | `0.172` | Pixel size (mm) |
| `beam_center_row` | `float` | `217.0` | Beam center row (px, rotated frame) |
| `beam_center_col` | `float` | `319.0` | Beam center col (px, rotated frame) |
| `panel_col_ranges` | `tuple` | `((0,206),(206,413),(413,619))` | Column ranges for 3 panels |
| `panel_offsets_deg` | `tuple` | `(-7.0, 0.0, 7.0)` | Panel tilt offsets (degrees) |
| `panel_row_shifts` | `tuple` | `(0.0, 0.0, 0.0)` | Per-panel row alignment shifts (px) |
| `panel_col_shifts` | `tuple` | `(0.0, 0.0, 0.0)` | Per-panel column alignment shifts (px) |
| `panel_delta_deg` | `tuple` | `(0.0, 0.0, 0.0)` | Per-panel additional tilt correction (degrees) |
| `theta_zero_deg` | `float` | `0.0` | Zero-offset of the arc motor (degrees) |
| `sample_offset_x_mm` | `float` | `0.0` | Sample mis-alignment in x (mm) |
| `sample_offset_z_mm` | `float` | `2.0` | Sample mis-alignment in z (beam direction, mm) |
| `beam_col_per_arc_deg` | `float` | `0.0` | Beam-center column drift per arc degree |
| `q_horizontal_sign` | `float` | `-1.0` | Sign convention for q_x |
| `q_vertical_sign` | `float` | `-1.0` | Sign convention for q_y |
| `rotation_k` | `int` | `3` | `np.rot90` k-value for raw→display orientation |

### Advanced WAXS Masking (via `waxs_kwargs` dict)

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `waxs_bsx_ref` | `float` | auto-derived | Beamstop x-motor reference position (mm at arc=0°) |
| `beamstop_max_abs_arc_deg` | `float` | `15.0` | Only apply beamstop mask for \|arc\| ≤ this |

### Advanced SAXS Dynamic Masking (via `saxs_kwargs` dict)

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `dynamic_saxs_mask` | `bool` | `False` | Enable per-frame WAXS-shadow + aperture masking |
| `dynamic_saxs_kwargs.waxs_shadow.enabled` | `bool` | `True` | Enable WAXS shadow on SAXS |
| `dynamic_saxs_kwargs.waxs_shadow.beam_visible_deg` | `float` | `14.5` | Arc angle below which WAXS detector blocks SAXS |
| `dynamic_saxs_kwargs.waxs_shadow.clear_edge_deg` | `float` | `18.0` | Arc angle above which shadow fully clears |
| `dynamic_saxs_kwargs.aperture.enabled` | `bool` | `True` | Enable q-aperture mask |
| `dynamic_saxs_kwargs.aperture.agbh_ring_order` | `int` | `5` | AgBh ring for aperture auto q-cutoff |
| `dynamic_saxs_kwargs.aperture.q_margin_fraction` | `float` | `0.01` | Margin above AgBh ring for cutoff |
| `dynamic_saxs_kwargs.aperture.q_cutoff` | `float \| None` | `None` | Manual q cutoff (overrides AgBh) |

---

## Entry Point: `reduce_smi_gi()`

Grazing-incidence WAXS pipeline. Bins into (q_xy, q_z) in the sample frame.

### Connection / Data Selection

Same as `reduce_smi_combined`: `uid`, `tiled_uri`, `catalog`.

### GI-Specific Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_qxy` | `int` | `500` | Number of q_xy bins |
| `n_qz` | `int` | `500` | Number of q_z bins |
| `incident_angle_deg` | `float \| None` | `None` (auto-detect) | Manual incident angle override (degrees). `None` = auto from sample_name or motors |
| `theta_offset` | `float` | `-0.5` | Added to (stage_th + piezo_th) during auto-detection |

### GI Masking & Detector

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `waxs_mask_path` | `str \| Path \| None` | `None` (bundled default) | WAXS polygon-mask JSON |
| `beamstop_max_abs_arc_deg` | `float` | `15.0` | Beamstop mask active for \|arc\| ≤ this |
| `waxs_beam_col_per_arc_deg` | `float` | `0.08` | Beam-center column drift per arc degree |
| `dezinger_threshold` | `float \| None` | `30000.0` | Dezinger sigma threshold |
| `dezinger_kernel` | `int` | `5` | Dezinger kernel size |

### GI WAXS Calibration Overrides (via `waxs_cal_overrides` dict)

Same fields as WAXSCalibration above. Common ones to tweak for GI:

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `energy_kev` | `float` | `16.1` | Photon energy |
| `sample_distance_mm` | `float` | `274.0` | Distance |
| `sample_offset_z_mm` | `float` | `2.0` | Sample offset along beam |
| `theta_zero_deg` | `float` | `0.0` | Arc motor zero offset |

---

## Geometry Auto-Resolution (from Tiled metadata)

The loader (`SMISWAXSLoader`) automatically resolves geometry from the run's
metadata using a smart fallback chain. You generally don't need to set these
unless the metadata is wrong.

### SAXS Geometry Fallback Chain

| Parameter | Primary source | Fallback 1 | Fallback 2 | Final default |
|-----------|---------------|------------|------------|---------------|
| Energy | `start.energy` | baseline `energy_energy` ÷ 1000 | sample_name `_{X}keV` | 16.1 keV |
| Distance | primary `pil2M_motor_z` | baseline `pil2M_motor_z` | sample_name `_sdd{X}m` | 2000 mm |
| Beam row | baseline `pil2M_beam_center_y_px` | primary conf | — | 1165.0 px |
| Beam col | baseline `pil2M_beam_center_x_px` | primary conf | — | 746.0 px |
| Beamstop | baseline `pil2M_active_beamstop` | primary conf | — | `"rod"` |

### WAXS Geometry Fallback Chain

| Parameter | Primary source | Fallback 1 | Fallback 2 | Final default |
|-----------|---------------|------------|------------|---------------|
| Energy | `start.energy` | baseline `energy_energy` ÷ 1000 | sample_name `_{X}keV` | 16.1 keV |
| Distance | primary `pil900KW_motor_z` | baseline `pil900KW_motor_z` | — | 270 mm |
| Beam row | — | — | — | 217.0 px (calibrated) |
| Beam col | — | — | — | 319.0 px (calibrated) |

### Additive Corrections Applied by Default

| Correction | Default Value | Override parameter |
|------------|---------------|-------------------|
| SAXS distance delta | -20.0 mm | `saxs_distance_delta_mm` |
| SAXS beam row delta | +2.0 px | `saxs_beam_delta_px[0]` |
| SAXS beam col delta | +3.0 px | `saxs_beam_delta_px[1]` |
| WAXS beam row delta | 0.0 px | `waxs_beam_delta_px[0]` |
| WAXS beam col delta | -2.0 px | `waxs_beam_delta_px[1]` |

---

## How the GUI Should Package Parameters

Every parameter goes into the single `reduce_smi_combined()` call. The
challenge is knowing which go as **top-level kwargs**, which go inside
**`backend_options={}`**, which inside **`waxs_kwargs={}`**, and which
inside **`saxs_kwargs={}`**. Here's the definitive mapping:

```python
from PyHyperScattering.SMISWAXSIntegrator import reduce_smi_combined

result = reduce_smi_combined(
    # ─── TOP-LEVEL: Connection ───────────────────────────────────────
    uid="<run-uid>",                          # required
    tiled_uri="https://tiled.nsls2.bnl.gov",  # default
    catalog="smi/migration",                  # default

    # ─── TOP-LEVEL: Grid ─────────────────────────────────────────────
    n_q=1000,                                 # output q bins
    n_chi=360,                                # output chi bins

    # ─── TOP-LEVEL: Masks ────────────────────────────────────────────
    saxs_mask_path=None,                      # None = bundled default
    waxs_mask_path=None,                      # None = bundled default

    # ─── TOP-LEVEL: SAXS Geometry Corrections ────────────────────────
    saxs_beam_delta_px=(2.0, 3.0),            # (row, col) additive to metadata
    saxs_distance_delta_mm=-20.0,             # additive to motor z

    # ─── TOP-LEVEL: SAXS Q-Range ─────────────────────────────────────
    saxs_q_cutoff=None,                       # None = auto from AgBh
    saxs_agbh_ring_order=5,                   # which AgBh ring for cutoff
    saxs_q_margin_fraction=0.01,              # margin above ring

    # ─── TOP-LEVEL: WAXS Geometry Corrections ────────────────────────
    waxs_beam_delta_px=(0.0, -2.0),           # (row, col) additive
    waxs_beam_col_per_arc_deg=0.0,            # beam drift with arc motor

    # ─── TOP-LEVEL: Hot-Pixel Rejection ──────────────────────────────
    dezinger_threshold=3000.0,                # None to disable
    dezinger_kernel=5,                        # median filter kernel

    # ─── TOP-LEVEL: Intensity Correction ─────────────────────────────
    solid_angle_correction=False,

    # ─── TOP-LEVEL: Geometry Mode ────────────────────────────────────
    geometry="transmission",                  # or "grazing_incidence"
    incident_angle_deg=0.0,                   # for GI mode

    # ─── TOP-LEVEL: Caching ──────────────────────────────────────────
    cache_geometry=True,

    # ─── backend_options DICT: Display/Orientation ───────────────────
    backend_options={
        "saxs_rotate_cw_90": False,           # rotate SAXS output
        "waxs_flip_horizontal": False,        # flip WAXS left-right
        "waxs_qx_shift_nm": 0.0,             # global qx offset
        "waxs_qy_shift_nm": 0.0,             # global qy offset
    },

    # ─── waxs_kwargs DICT: WAXS Calibration & Masking ────────────────
    waxs_kwargs={
        # --- Masking ---
        "beamstop_max_abs_arc_deg": 15.0,     # mask beamstop for |arc| ≤ this
        "waxs_bsx_ref": None,                 # None = auto-derive from arc/bsx

        # --- WAXSCalibration overrides (any field accepted) ---
        "energy_kev": 16.1,
        "sample_distance_mm": 274.0,
        "pixel_size_mm": 0.172,
        "beam_center_row": 217.0,             # in rotated frame
        "beam_center_col": 319.0,             # in rotated frame
        "panel_offsets_deg": (-7.0, 0.0, 7.0),
        "panel_row_shifts": (0.0, 0.0, 0.0),
        "panel_col_shifts": (0.0, 0.0, 0.0),
        "panel_delta_deg": (0.0, 0.0, 0.0),
        "panel_col_ranges": ((0, 206), (206, 413), (413, 619)),
        "theta_zero_deg": 0.0,
        "sample_offset_x_mm": 0.0,
        "sample_offset_z_mm": 2.0,
        "beam_col_per_arc_deg": 0.0,
        "q_horizontal_sign": -1.0,
        "q_vertical_sign": -1.0,
        "rotation_k": 3,
    },

    # ─── saxs_kwargs DICT: SAXS Dynamic Masking ──────────────────────
    saxs_kwargs={
        "dynamic_saxs_mask": False,           # enable per-frame mask
        "dynamic_saxs_kwargs": {
            "waxs_shadow": {
                "enabled": True,
                "beam_visible_deg": 14.5,
                "clear_edge_deg": 18.0,
                "beam_visible_offset_px": 0.0,
                "edge_margin_px": 0.0,
            },
            "aperture": {
                "enabled": True,
                "agbh_ring_order": 5,
                "q_margin_fraction": 0.01,
                "q_cutoff": None,             # explicit override
            },
        },
    },
)
```

### Parameter Routing Summary (for GUI builders)

| Where it goes | What belongs there |
|---------------|-------------------|
| **Top-level kwargs** | Connection, grid size, mask paths, geometry corrections (beam deltas, distance delta), q-range, dezinger, solid angle, geometry mode, caching |
| **`backend_options={}`** | Display orientation: rotations, flips, global q-shifts |
| **`waxs_kwargs={}`** | WAXS detector calibration (WAXSCalibration fields), WAXS beamstop masking |
| **`saxs_kwargs={}`** | SAXS dynamic masking (WAXS shadow, aperture) |

### GUI Recommended Groupings

**Group 1: "Basics"** (most users change these)
- `n_q`, `n_chi`, `dezinger_threshold`, `saxs_q_cutoff`

**Group 2: "SAXS Calibration"** (tweak when beam center or distance is off)
- `saxs_beam_delta_px`, `saxs_distance_delta_mm`

**Group 3: "WAXS Calibration"** (tweak when WAXS pattern is misaligned)
- `waxs_beam_delta_px`, `waxs_beam_col_per_arc_deg`
- `waxs_kwargs["sample_offset_z_mm"]`, `waxs_kwargs["theta_zero_deg"]`

**Group 4: "Advanced WAXS"** (rarely changed)
- Panel geometry: `panel_offsets_deg`, `panel_row_shifts`, `panel_col_shifts`, `panel_delta_deg`
- Sign conventions: `q_horizontal_sign`, `q_vertical_sign`

**Group 5: "Masks"** (file pickers + mask behavior)
- `saxs_mask_path`, `waxs_mask_path`
- `waxs_kwargs["beamstop_max_abs_arc_deg"]`
- `saxs_kwargs["dynamic_saxs_mask"]`

**Group 6: "Display"** (cosmetic)
- All keys in `backend_options`

### Minimal Call (defaults for everything):
```python
result = reduce_smi_combined(uid="<uid>")
```

---

## Module Constants (for reference)

From `SMISWAXSLoader.py`:
```
PILATUS_PIXEL_SIZE_M     = 0.172e-3   (172 µm)
DEFAULT_ENERGY_KEV       = 16.1
SAXS_IMAGE_FIELD         = "pil2M_image"
WAXS_IMAGE_FIELD         = "pil900KW_image"
WAXS_ARC_FIELD           = "waxs_arc"
WAXS_BSX_FIELD           = "waxs_bsx"
```

From `smi_defaults.py`:
```
BSX_PER_ARC_DEG          = -4.39      (mm beamstop-x per degree of arc)
DEFAULT_SAXS_MASK_NAME   = "pil2M_mask_polygons.json"
DEFAULT_WAXS_MASK_NAME   = "900KW_mask_polygons.json"
```
