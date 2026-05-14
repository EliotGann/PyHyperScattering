# PyHyperScattering: Per-Frame I(q) Output Requirement

## Context

The SMI browser's Process tab currently receives a `result` object from
`reduce_smi_combined()` with:

- `result.merged_iq` — an **xarray Dataset** containing the all-frames-merged 1D
  intensity profile. Variables: `q`, `I`, optionally `saxs_I` and `waxs_I`.
- `result.merged_qchi` — an **xarray Dataset** with dimensions `(chi, q)` *or*
  `(frame, chi, q)` when the scan has multiple exposures. The `frame` dimension
  already enables per-frame 2D map browsing.

The problem: while the 2D q–χ maps are available per-frame, the final 1D I(q)
is only available merged. Users need per-frame I(q) curves (e.g., to track
intensity evolution over a temperature ramp or sample rotation).

---

## Requested Addition: `result.per_frame_iq`

### Structure

```
result.per_frame_iq : xarray.Dataset
    Dimensions:
        frame : int          # same length as merged_qchi["frame"]
        q     : float64      # same q grid as merged_iq["q"]

    Data variables:
        I       : (frame, q)  float64   # merged SAXS+WAXS intensity per frame
        saxs_I  : (frame, q)  float64   # (optional) SAXS-only intensity per frame
        waxs_I  : (frame, q)  float64   # (optional) WAXS-only intensity per frame

    Coordinates:
        q       : (q,)        float64   # q in nm⁻¹ (same grid as merged_iq)
        frame   : (frame,)    int       # 0-based frame index
```

### Behaviour

| Scenario | Expected |
|----------|----------|
| Single-frame scan | `per_frame_iq` may be `None` or a Dataset with `frame` size 1 |
| Multi-frame scan  | One I(q) curve per frame, azimuthally integrated identically to `merged_iq` but without frame-averaging |
| Frames with all-NaN SAXS or WAXS | Corresponding `saxs_I` / `waxs_I` row should be NaN |

### Integration with Primary Axis Labels

The browser fetches per-frame metadata (motor positions, temperature, etc.) from
Tiled's `primary` stream independently. The frame dimension in `per_frame_iq`
must correspond 1-to-1 with the rows in the primary scalar table so the browser
can label each curve (e.g., "temperature=25.3 °C").

**Critical:** The frame ordering in `per_frame_iq` must match the image
acquisition order in the Tiled run (i.e., `frame=0` corresponds to the first
image/event in the primary stream).

---

## Optional: Primary Axis Coordinate Forwarding

If convenient, PyHyper may optionally attach a primary-axis coordinate to the
dataset:

```
    Coordinates:
        primary_value : (frame,)  float64   # e.g., temperature or angle values
    Attributes:
        primary_field : str                  # field name (e.g., "sample_temperature")
```

This is **not required** — the browser can look up primary scalars itself — but
if the information is readily available during reduction it would simplify
downstream consumers.

---

## Backward Compatibility

- `merged_iq` must continue to exist as before (all frames merged).
- `per_frame_iq` should be an **additional** attribute that may be `None` when
  the backend hasn't been updated yet (the browser falls back to chi-integration
  of `merged_qchi` per frame).

---

## GI Mode (`reduce_smi_gi`)

For GI-WAXS processing, per-frame 2D maps already exist as `gi_result.frames`.
An analogous per-frame 1D output is not currently needed (no merged I(q) in GI
mode), but if added in the future it should follow the same Dataset structure
with `qxy` or `qz` as the 1D coordinate.

---

## Summary of Changes Needed in PyHyperScattering

1. In `reduce_smi_combined`, after computing `merged_iq` (the frame-averaged
   azimuthal integration), retain the **per-frame** 1D curves as well.
2. Attach them as `result.per_frame_iq` (xarray Dataset with `frame` and `q`
   dimensions).
3. Ensure frame ordering matches Tiled event order.
4. Keep `merged_iq` as-is for backward compatibility.
