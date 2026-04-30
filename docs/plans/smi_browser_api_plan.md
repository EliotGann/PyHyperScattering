# SMI-Browser ↔ PyHyperScattering API consolidation plan

**Audience:** an implementing agent (or human) working in the `PyHyperScattering` repo.
**Goal:** move beamline-/detector-specific data logic that currently lives in `smi-browser/smi_app.py` into PyHyperScattering, behind a small, stable, import-light API. Once these helpers exist, the browser can delete ~250 lines of mirrored / re-implemented code.

This plan describes the **API surface to add**, the **migration map** so the browser-side cleanup is mechanical, the **constraints** the implementation must respect, and the **test plan**. There is **no UI, no Panel, no Bokeh, no Tiled-search code in scope** — only data-side helpers.

> **⚠️ Please be careful.** Several existing `SMISWAXSIntegrator` and `SMISWAXSLoader` callers depend on current behavior (orientation conventions, calibration deltas, mask schemas). Wherever this plan adds a new function, it should *wrap* the existing primitives, not replace them. Do not change the signatures of `make_waxs_mask_callable`, `make_saxs_mask_from_spec`, `make_mask_for_angle`, `polygons_to_mask`, `shift_polygon`, `reduce_smi_combined`, or `reduce_smi_gi` without an explicit deprecation cycle.

---

## 1. Why this consolidation

The browser currently re-implements, inline, the following beamline knowledge that PyHyper *also* knows about:

1. **Detector classification** by field-name substring (`"900kw"` → WAXS, `"pil2m"` → SAXS).
2. **Display orientation** (WAXS = `rot90(k=3) + fliplr` ≡ transpose; SAXS = `flipud`).
3. **Polygon-coordinate transform** mirroring the display orientation, for both forward (file → Bokeh) and inverse (Bokeh → file) directions.
4. **Mask-file schema parsing** for two on-disk shapes: nested SAXS (`{static_regions: …, beamstops: …}`) and flat WAXS (`{name: verts, …}`). SAXS beamstops can be wrapped as `{"polygon": [...], "x_motor_key": ...}` and must be unwrapped.
5. **Per-frame dynamic mask construction**: pull `waxs_arc` and `waxs_bsx` from a run's primary stream, derive `waxs_bsx_ref` via the SMI mechanical linkage `bsx_ref = bsx − BSX_PER_ARC_DEG · arc`, and call `make_waxs_mask_callable(...)`.
6. **Loader calibration constants** (`SAXS_ROW_DELTA = 2.0`, `SAXS_COL_DELTA = 3.0`, `WAXS_ROW_DELTA = 0.0`, `WAXS_COL_DELTA = -2.0`, `SAXS_DIST_DELTA_MM = -20.0`, `BSX_PER_ARC_DEG = -4.39`) currently mirrored in the browser with a comment that says they are drift-prone.

Every one of these will silently desynchronize from upstream the next time PyHyper updates calibration. This plan eliminates that risk.

---

## 2. Hard constraints

These constraints are **non-negotiable** for the browser to keep working as it does today.

1. **Import weight.** `smi_defaults.py` must remain importable without pulling in `pyFAI`, `WPIntegrator`, `xarray`, or any other heavy dependency. Today it imports only `pathlib`, `importlib.resources`, and `typing`. Keep it that way. Anything that needs heavy imports goes in `SMISWAXSIntegrator.py` (or a new `SMISWAXSFrames.py`) — not in `smi_defaults`.

2. **No behavior change** for existing reduction calls. `reduce_smi_combined`, `reduce_smi_gi`, `make_waxs_mask_callable`, `make_saxs_mask_from_spec`, and friends keep their current signatures and current numerical output, byte-for-byte.

3. **The bundled-mask schema is owned by PyHyper.** The browser currently parses two schemas inline; after this work the browser parses none. If the on-disk format ever changes, only PyHyper updates.

4. **Orientation conventions.** WAXS display = `np.fliplr(np.rot90(raw, k=3))` (≡ transpose). SAXS display = `np.flipud(raw)`. These are the conventions the bundled mask polygons were authored against; do not change them.

5. **Detector-name substrings** (`"900kw"`, `"pil2m"`/`"pilatus2m"`/`"saxs"`, plus the bare aliases `"saxs"`/`"waxs"`) must be the source of truth. Today the browser hardcodes `{"pil2m", "pilatus2m", "saxs"}` and `{"900kw", "waxs"}`. Either reuse those exact sets, or expand them — but never narrow.

6. **SMI mechanical linkage.** `BSX_PER_ARC_DEG = -4.39` mm/deg is a beamline constant. It already lives in `SMISWAXSIntegrator`; promote/expose it but do not change its value.

---

## 3. API to add

### 3.1 In `smi_defaults.py` (light, no heavy imports)

```python
from typing import Literal, Optional
from dataclasses import dataclass

# --- Detector classification --------------------------------------------------

SAXS_DETECTOR_NAMES: frozenset[str] = frozenset({"pil2m", "pilatus2m", "saxs"})
WAXS_DETECTOR_NAMES: frozenset[str] = frozenset({"900kw", "waxs"})

DetectorKind = Literal["saxs", "waxs"]

def classify_detector_field(name: str) -> Optional[DetectorKind]:
    """Return ``'saxs'``, ``'waxs'``, or ``None`` for an unknown name.

    Matching is case-insensitive substring against
    :data:`SAXS_DETECTOR_NAMES` / :data:`WAXS_DETECTOR_NAMES`.
    """

# --- Beamline / calibration constants ----------------------------------------

#: SMI mechanical linkage: change in waxs_bsx (mm) per degree of waxs_arc.
BSX_PER_ARC_DEG: float = -4.39

@dataclass(frozen=True)
class LoaderCalibration:
    """Calibrated default deltas applied by :class:`SMISWAXSLoader`.

    These mirror ``SMISWAXSLoader._{SAXS,WAXS}_DEFAULT_*_PX`` /
    ``_MM`` so external callers can introspect "what would the loader use
    if I pass None?" without instantiating the loader.
    """
    saxs_row_delta_px:    float = 2.0
    saxs_col_delta_px:    float = 3.0
    waxs_row_delta_px:    float = 0.0
    waxs_col_delta_px:    float = -2.0
    saxs_distance_delta_mm: float = -20.0

LOADER_DEFAULTS: LoaderCalibration = LoaderCalibration()

# --- Display orientation (pure numpy / arithmetic) ---------------------------

def orient_frame_for_display(arr: "np.ndarray", detector: DetectorKind) -> "np.ndarray":
    """Apply the canonical display orientation for an SMI detector frame.

    * WAXS: ``np.fliplr(np.rot90(arr, k=3))`` (= transpose)
    * SAXS: ``np.flipud(arr)``
    """

def orient_polygon_xy(
    col_raw: float, row_raw: float,
    detector: DetectorKind,
    raw_shape: tuple[int, int],
) -> tuple[float, float]:
    """Map a *raw-detector* (col, row) vertex to display (x, y) coords.

    Mirrors :func:`orient_frame_for_display` — Bokeh's image glyph draws
    ``array[0, :]`` at the *bottom*, so the math here accounts for that
    placement.

    SAXS:  ``(col, raw_h - row)``
    WAXS:  ``(row, col)``
    """

def orient_polygon_xy_inverse(
    x: float, y: float,
    detector: DetectorKind,
    raw_shape: tuple[int, int],
) -> tuple[float, float]:
    """Inverse of :func:`orient_polygon_xy`. Use when saving edits."""

# --- Mask file I/O (schema-normalized) ---------------------------------------

NormalizedMask = dict  # see schema below

def load_mask_polygons(path) -> NormalizedMask:
    """Load a polygon-mask JSON file in either supported on-disk schema
    and return a normalized dict.

    Supported on-disk schemas:

    * **Nested** (SAXS / pil2M):
      ``{"static_regions": {name: verts}, "beamstops": {name: verts}, "image_shape": [...]}``
    * **Flat** (WAXS / 900KW):
      ``{name: verts, ...}`` with names containing ``"beamstop"`` treated
      as beamstops; everything else is static. ``"image_shape"`` may also
      appear as a top-level key.

    SAXS beamstop entries may wrap their polygon as
    ``{"polygon": [[c, r], ...], "x_motor_key": ...}``; the wrapper is
    unwrapped.

    Returns
    -------
    NormalizedMask : dict
        Always shaped as::

            {
                "image_shape": [rows, cols] | None,
                "static_regions": {name: [[col, row], ...], ...},
                "beamstops":      {name: [[col, row], ...], ...},
            }

        Coordinates are in **raw detector indexing** (col, row), unchanged
        from the source file.
    """

def save_mask_polygons(mask: NormalizedMask, path) -> None:
    """Write a normalized mask dict back out, preserving the nested schema.

    Empty buckets are written as ``{}`` rather than omitted, so a
    round-trip ``load → save`` is byte-stable for nested-schema inputs.
    """
```

### 3.2 In `SMISWAXSIntegrator.py` (or a new `SMISWAXSFrames.py`)

```python
def mask_for_frame(
    run_or_uid,                              # bluesky/tiled run, or uid str
    frame_idx: int,
    detector: DetectorKind,
    *,
    mask_path: str | Path | None = None,    # None → bundled default
    orient_for_display: bool = False,       # True applies orient_frame_for_display
    tiled_uri: str | None = None,           # used only if run_or_uid is a uid
    catalog:   str | None = None,
) -> "np.ndarray":
    """Return the boolean validity mask (True = valid) the integrator
    *would* use for a single frame of a single detector of a single run.

    Wraps the existing primitives without changing them:

    * SAXS: calls :func:`make_saxs_mask_from_spec(raw_shape, mask_path)`.
      Per-frame inputs are ignored (SAXS mask is per-scan in SMI today),
      but the function signature still takes ``frame_idx`` for symmetry
      with the WAXS path and so future per-frame variation is non-breaking.
    * WAXS: pulls ``waxs_arc`` and ``waxs_bsx`` from the run's ``primary``
      stream at ``frame_idx``, derives
      ``waxs_bsx_ref = waxs_bsx − BSX_PER_ARC_DEG · waxs_arc``,
      then calls :func:`make_waxs_mask_callable(mask_path,
      waxs_bsx_ref=…, beamstop_max_abs_arc_deg=6.0)` and invokes the
      returned callable with ``(raw_shape, theta=waxs_arc, bsx=waxs_bsx)``.

    The ``raw_shape`` is read once from the detector dataset's
    ``shape[-2:]`` so this function does **not** read the heavy image
    array.

    If ``orient_for_display`` is True the returned mask is passed through
    :func:`orient_frame_for_display` so it overlays a display-oriented
    image directly. **WAXS note:** PyHyper's WAXS mask builder already
    applies ``rot90+fliplr`` internally, so for WAXS the implementation
    must skip the second orientation pass — the public contract is just
    "the returned array overlays the display-oriented image."

    Raises
    ------
    KeyError
        If ``waxs_arc`` / ``waxs_bsx`` are not present in the primary
        stream when WAXS is requested.
    ValueError
        If ``detector`` is neither ``"saxs"`` nor ``"waxs"``.
    """
```

### 3.3 What stays exactly as-is

Do **not** change:

- `make_waxs_mask_callable`
- `make_saxs_mask_from_spec`
- `make_mask_for_angle`
- `polygons_to_mask`
- `shift_polygon`
- `reduce_smi_combined`
- `reduce_smi_gi`
- Any field of `CombinedReductionResult` / `GIReductionResult`

The new functions are *thin wrappers* over these.

---

## 4. Migration map (browser-side cleanup, after PyHyper changes land)

These are the lines in `smi-browser/smi_app.py` that become deletable. Use this as a checklist when sanity-checking the new API; if any item below cannot be deleted, the new API is incomplete.

| `smi_app.py` function / block                              | Lines (approx) | Replaced by                                       |
|------------------------------------------------------------|----------------|---------------------------------------------------|
| `_is_waxs_field`                                           | 469-472        | `smi_defaults.classify_detector_field`            |
| `_detector_kind_for_field`                                 | 657-660        | `smi_defaults.classify_detector_field`            |
| `_orient_frame`                                            | 474-486        | `smi_defaults.orient_frame_for_display`           |
| `_orient_polygon_xy`                                       | 489-507        | `smi_defaults.orient_polygon_xy`                  |
| `_orient_polygon_xy_inverse`                               | 510-520        | `smi_defaults.orient_polygon_xy_inverse`          |
| `_load_mask_dict`                                          | 528-535        | `smi_defaults.load_mask_polygons`                 |
| `_mask_dict_to_xs_ys` (the schema-parsing half)            | 537-619        | `smi_defaults.load_mask_polygons` (returns normalized dict; browser keeps a tiny "normalized → Bokeh xs/ys" projection that is pure UI) |
| `_xs_ys_to_mask_dict`                                      | 621-655        | `smi_defaults.save_mask_polygons` (browser keeps the inverse projection only) |
| Mirrored constants `DEFAULT_SAXS_ROW_DELTA` … `DEFAULT_SAXS_DIST_DELTA` | 95-104 | `smi_defaults.LOADER_DEFAULTS` |
| `_BSX_PER_ARC_DEG`                                         | 1186           | `smi_defaults.BSX_PER_ARC_DEG`                    |
| `_per_frame_scalar`                                        | 1190-1207      | absorbed into `mask_for_frame`                    |
| `_orient_mask_for_display`                                 | 1211-1221      | absorbed into `mask_for_frame(orient_for_display=True)` |
| `_build_dynamic_mask_for_frame`                            | 1224-1268      | replaced by single call to `mask_for_frame`       |

After migration the browser's dynamic-mask block collapses to roughly:

```python
from PyHyperScattering import smi_defaults as smid
from PyHyperScattering.SMISWAXSIntegrator import mask_for_frame

detector = smid.classify_detector_field(field)
if detector is None:
    return None
mask = mask_for_frame(run, idx, detector, orient_for_display=True)
```

…and the static-mask overlay becomes:

```python
normalized = smid.load_mask_polygons(path)
xs, ys, names, kinds = _normalized_to_bokeh_xs_ys(normalized, detector, raw_shape)
```

where `_normalized_to_bokeh_xs_ys` is a tiny browser-local function that only does coordinate orientation via `smid.orient_polygon_xy` — no schema parsing, no wrapper unwrapping, no SAXS/WAXS asymmetry.

---

## 5. Test plan

Add tests under `PyHyperScattering/tests/`. Use small synthetic shapes — these tests must run without network access and without any tiled catalog.

1. **`test_smi_defaults_classify.py`** — table-driven: `"900KW_image"` → `"waxs"`, `"pil2M_image"` → `"saxs"`, `"saxs"` → `"saxs"`, `"waxs"` → `"waxs"`, `"unknown"` → `None`. Case-insensitive.

2. **`test_smi_defaults_orient.py`** — round-trip:
   - `orient_frame_for_display(orient_frame_for_display(x, "saxs"), "saxs")` is `x` (since `flipud ∘ flipud = id`).
   - WAXS round-trip: applying the orient transform twice equals the identity (transpose ∘ transpose = id).
   - For each detector, build a small `(rows, cols)` array with a single `1.0` at `(r, c)`. Assert that after `orient_frame_for_display`, the `1.0` lands at `orient_polygon_xy(c, r, detector, (rows, cols))` (with appropriate floor/round to integer indices).

3. **`test_smi_defaults_polygon_inverse.py`** — for both detectors and a couple of synthetic `raw_shape`s, verify
   `orient_polygon_xy_inverse(*orient_polygon_xy(c, r, det, shp), det, shp) == (c, r)`
   for several random `(c, r)` floats.

4. **`test_smi_defaults_mask_io.py`** —
   - Load each bundled mask (`default_saxs_mask_path()`, `default_waxs_mask_path()`) and assert the returned dict has the three keys (`image_shape`, `static_regions`, `beamstops`), all coords are 2-tuples of floats, and at least one polygon exists in each.
   - Load → save → load round-trip on both bundled masks and assert structural equality.
   - Synthetic flat-schema dict with a key `"beamstop_left"` is normalized into the `beamstops` bucket.
   - Synthetic SAXS beamstop wrapped as `{"polygon": [...], "x_motor_key": "saxs_bsx"}` is unwrapped.

5. **`test_mask_for_frame.py`** —
   - Mock a "run" object with a `primary` stream exposing `waxs_arc=[0.0, 3.0]`, `waxs_bsx=[10.0, 10.0 - 3.0 * BSX_PER_ARC_DEG]`. Verify that for both frames, `waxs_bsx_ref` derived inside `mask_for_frame` is the same constant.
   - Assert the returned mask has shape `raw_shape` (or its display-oriented equivalent when `orient_for_display=True`).
   - SAXS path: `frame_idx` is ignored; mask is identical for any frame.
   - `orient_for_display=True`: returned shape matches `orient_frame_for_display(np.zeros(raw_shape), detector).shape`.

If you cannot easily mock a "run" object, accept a dict-like with `primary[<field>] -> np.ndarray` for testing and document that.

---

## 6. Performance / startup constraint

`smi_defaults` is currently importable in <50 ms. The browser relies on this — it lazily resolves PyHyper-bundled mask paths at first-use time specifically to keep startup fast.

**You must not** add a top-level `import pyFAI`, `import xarray`, `import torch`, `import WPIntegrator`, etc. to `smi_defaults.py`. The new helpers in §3.1 are pure-Python / pure-numpy and have no reason to need anything heavy.

The `mask_for_frame` helper in §3.2 *does* need the integrator primitives, so it lives in `SMISWAXSIntegrator.py` (where those primitives already are). That file is heavy and the browser already imports it lazily, so this is fine.

---

## 7. Backward compatibility

- All new symbols are **additive**.
- No existing function signature changes.
- No existing constant value changes.
- The bundled mask files on disk are untouched.

If you find that an existing internal helper (e.g. inside `SMISWAXSIntegrator`) duplicates one of the new public functions, **leave the internal helper in place** and have it call the new public one. Do not break import paths that downstream code may be using.

---

## 8. ⚠️ Required deliverable: implementation report

When the PyHyper-side work is complete, please write a short markdown file at:

```
/nsls2/users/egann/git/PyHyperScattering/docs/plans/smi_browser_api_plan_RESPONSE.md
```

(same `docs/plans/` directory as this plan), containing:

1. **Implemented symbols** — bullet list of every public name added, with its fully-qualified import path. Example:
   - `PyHyperScattering.smi_defaults.classify_detector_field`
   - `PyHyperScattering.smi_defaults.LOADER_DEFAULTS`
   - `PyHyperScattering.SMISWAXSIntegrator.mask_for_frame`
2. **Anything from §3 that was NOT implemented**, and why (e.g. "deferred — needs design discussion").
3. **Any deviations from the proposed signatures**, with the actual signature and a one-line rationale.
4. **Test files added**, with the test count per file.
5. **Migration green-light list** — for each row in the §4 migration table, mark ✅ (safe to delete browser-side) or ❌ (still needed in browser, with reason).
6. **Any new constraints discovered** that the browser-side cleanup needs to know about (e.g. "WAXS mask now returned with an extra metadata attr you may want to surface").

The smi-browser cleanup PR will read this response file directly and use it as the authoritative checklist for what to delete. Be precise; if you say a function exists, the browser's import will be done verbatim from your text.

---

## 9. Out of scope

Explicitly **not** part of this work:

- Anything in `tiled_browser.py` (tiled REST search helpers).
- Panel / Bokeh / UI code in any form.
- Changes to `reduce_smi_combined` / `reduce_smi_gi` numerics or signatures.
- Mask *editing* logic (PolyDraw / PolyEdit) — that stays in the browser.
- A new on-disk mask schema. The two existing schemas are read; only the nested schema is written.
- Async / threading concerns. All new functions are synchronous.

---

## 10. Suggested PR shape

One PR, ideally:

1. Commit 1: pure additions to `smi_defaults.py` (§3.1) + tests for §5.1-§5.4.
2. Commit 2: `mask_for_frame` in `SMISWAXSIntegrator.py` (§3.2) + test §5.5.
3. Commit 3: docs (`smi_browser_api_plan_RESPONSE.md` from §8).

Keep each commit independently reviewable and green on CI.

---

*Plan authored by the smi-browser maintainer's coding agent on 2026-04-29 in response to a refactor that has accumulated ~250 lines of mirrored beamline knowledge in the browser. The browser-side cleanup is blocked on the response document described in §8.*
