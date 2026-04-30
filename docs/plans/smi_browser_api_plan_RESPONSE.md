# SMI-Browser ↔ PyHyperScattering API consolidation — implementation response

Companion to [smi_browser_api_plan.md](smi_browser_api_plan.md). This document is the
authoritative checklist for the smi-browser cleanup PR.

## 1. Implemented symbols

All additive — no existing signatures or constants changed.

### `PyHyperScattering.smi_defaults`
- `PyHyperScattering.smi_defaults.SAXS_DETECTOR_NAMES` — `frozenset({"pil2m","pilatus2m","saxs"})`
- `PyHyperScattering.smi_defaults.WAXS_DETECTOR_NAMES` — `frozenset({"900kw","waxs"})`
- `PyHyperScattering.smi_defaults.DetectorKind` — `Literal["saxs","waxs"]`
- `PyHyperScattering.smi_defaults.classify_detector_field`
- `PyHyperScattering.smi_defaults.BSX_PER_ARC_DEG` — `-4.39`
- `PyHyperScattering.smi_defaults.LoaderCalibration` — frozen dataclass
- `PyHyperScattering.smi_defaults.LOADER_DEFAULTS` — instance with the deltas
  `(saxs_row=2.0, saxs_col=3.0, waxs_row=0.0, waxs_col=-2.0, saxs_dist_mm=-20.0)`
- `PyHyperScattering.smi_defaults.orient_frame_for_display`
- `PyHyperScattering.smi_defaults.orient_polygon_xy`
- `PyHyperScattering.smi_defaults.orient_polygon_xy_inverse`
- `PyHyperScattering.smi_defaults.load_mask_polygons`
- `PyHyperScattering.smi_defaults.save_mask_polygons`
- `PyHyperScattering.smi_defaults.SAXS_IMAGE_FIELD` — `"pil2M_image"` (re-export of the loader constant)
- `PyHyperScattering.smi_defaults.WAXS_IMAGE_FIELD` — `"pil900KW_image"`
- `PyHyperScattering.smi_defaults.WAXS_ARC_FIELD` — `"waxs_arc"`
- `PyHyperScattering.smi_defaults.WAXS_BSX_FIELD` — `"waxs_bsx"`
- `PyHyperScattering.smi_defaults.DEFAULT_DETECTOR_FIELDS` — `{"saxs": SAXS_IMAGE_FIELD, "waxs": WAXS_IMAGE_FIELD}`

### `PyHyperScattering.SMISWAXSIntegrator`
- `PyHyperScattering.SMISWAXSIntegrator.mask_for_frame`

### `PyHyperScattering.SMISWAXSLoader.TiledSMISWAXSLoader` (new methods)
- `searchCatalog(sample=None, plan=None, scan_id=None, cycle=None, proposal=None,
  user=None, institution=None, detector=None, since=None, until=None, limit=None,
  outputType="default", **kwargs)` → `pandas.DataFrame`
- `browseCatalog(**kwargs)` → `ipyaggrid.Grid` (lazy import; `ipyaggrid` is optional)
- `summarizeRun(uid)` → `dict` of headline metadata (no primary-stream read)

The new `searchCatalog` mirrors the API shape of
`PyHyperScattering.SST1RSoXSDB.searchCatalog` so JupyterHub / notebook users
get the same workflow at SMI as at SST-1.

## 2. Items from §3 NOT implemented

None. All §3.1 and §3.2 symbols are implemented.

## 3. Deviations from proposed signatures

| Plan signature | Actual signature | Rationale |
|---|---|---|
| `classify_detector_field(name) -> Optional[DetectorKind]` | Same; returns plain `str` (`"saxs"`/`"waxs"`) or `None`. WAXS classification is checked **before** SAXS to disambiguate names that contain both substrings. | Defensive; matches the plan's intent without breaking ties arbitrarily. |
| `mask_for_frame(run_or_uid, frame_idx, detector, *, mask_path=None, orient_for_display=False, tiled_uri=None, catalog=None)` | Same plus two extras: `raw_shape: tuple[int,int] | None = None` and `beamstop_max_abs_arc_deg: float | None = 6.0` | `raw_shape` lets callers (and tests) bypass the run-side shape probe. `beamstop_max_abs_arc_deg` exposes the existing `make_waxs_mask_callable` knob without requiring a re-build. Both have safe defaults that preserve the documented behavior. |
| `LoaderCalibration` field types | Same field names and values as proposed. | — |

The bundled WAXS mask currently has no `image_shape` key, so
`load_mask_polygons` returns `image_shape=None` for it. Round-tripping that
file through `save_mask_polygons` therefore omits `image_shape` from the
output (the plan said to write it as `[rows, cols]` only when present).

## 4. Test files added / updated

| File | New tests | Existing tests |
|---|---|---|
| [tests/test_smi_defaults.py](../../tests/test_smi_defaults.py) | **+27** (classify, calibration constants, orient round-trips, polygon ↔ frame placement, mask I/O nested + flat + wrapper unwrap, validation) | 8 (kept) |
| [tests/test_mask_for_frame.py](../../tests/test_mask_for_frame.py) | **9** (SAXS basic, frame-idx ignored, display-orient, WAXS bsx_ref derivation against explicit `make_waxs_mask_callable`, shape, display-orient is no-op for WAXS, missing motors → `KeyError`, invalid detector, `raw_shape` override) | n/a |

All 48 tests pass (`pytest tests/test_smi_defaults.py tests/test_mask_for_frame.py`).

The `test_loader_defaults_match_loader_module` assertion guards against
the calibration-drift risk called out in §1 of the plan: if anyone changes
`_SAXS_DEFAULT_BEAM_DELTA_*` in `SMISWAXSLoader.py` without also updating
`LoaderCalibration` in `smi_defaults.py`, this test fails on CI.

## 5. Migration green-light list (browser-side)

| `smi_app.py` block | Status | Replacement |
|---|---|---|
| `_is_waxs_field` | ✅ delete | `smi_defaults.classify_detector_field(field) == "waxs"` |
| `_detector_kind_for_field` | ✅ delete | `smi_defaults.classify_detector_field(field)` |
| `_orient_frame` | ✅ delete | `smi_defaults.orient_frame_for_display(arr, detector)` |
| `_orient_polygon_xy` | ✅ delete | `smi_defaults.orient_polygon_xy(c, r, detector, raw_shape)` |
| `_orient_polygon_xy_inverse` | ✅ delete | `smi_defaults.orient_polygon_xy_inverse(x, y, detector, raw_shape)` |
| `_load_mask_dict` | ✅ delete | `smi_defaults.load_mask_polygons(path)` |
| `_mask_dict_to_xs_ys` (schema-parsing half) | ✅ delete | `smi_defaults.load_mask_polygons` returns the normalized dict; browser keeps a tiny "normalized → Bokeh xs/ys" projection that only uses `orient_polygon_xy` |
| `_xs_ys_to_mask_dict` | ✅ delete | `smi_defaults.save_mask_polygons(mask_dict, path)`; browser keeps the inverse `xs_ys → normalized` projection |
| Mirrored constants `DEFAULT_SAXS_ROW_DELTA` … `DEFAULT_SAXS_DIST_DELTA` | ✅ delete | `smi_defaults.LOADER_DEFAULTS` (frozen dataclass) |
| `_BSX_PER_ARC_DEG` | ✅ delete | `smi_defaults.BSX_PER_ARC_DEG` |
| `_per_frame_scalar` | ✅ delete | absorbed into `mask_for_frame` |
| `_orient_mask_for_display` | ✅ delete | absorbed into `mask_for_frame(orient_for_display=True)` |
| `_build_dynamic_mask_for_frame` | ✅ delete | replaced by single call to `mask_for_frame` |

After migration the browser's dynamic-mask block collapses to:

```python
from PyHyperScattering import smi_defaults as smid
from PyHyperScattering.SMISWAXSIntegrator import mask_for_frame

detector = smid.classify_detector_field(field)
if detector is None:
    return None
mask = mask_for_frame(run, idx, detector, orient_for_display=True)
```

## 6. New constraints / things the browser should know

1. **WAXS mask returned by `mask_for_frame` is already display-oriented.**
   The plan called this out, but to be explicit: `mask_for_frame(run, i, "waxs",
   orient_for_display=True)` and `mask_for_frame(run, i, "waxs",
   orient_for_display=False)` return the *same* array. PyHyper's
   `make_waxs_mask_callable` applies `np.fliplr(np.rot90(..., k=3))` internally
   via `make_mask_for_angle`, so its output already overlays the display image.
   The `orient_for_display` flag is honored only for SAXS.

2. **`load_mask_polygons` always returns floats.** Both `(col, row)` components
   are coerced to `float`, even when the source JSON used ints. Browser
   serializers that require ints should cast back at the boundary.

3. **WAXS bundled mask has no `image_shape` key.** `load_mask_polygons` returns
   `image_shape=None` for it; the round-trip via `save_mask_polygons` will
   therefore omit that key. SAXS round-trips preserve `image_shape`.

4. **`searchCatalog` requires `tiled.queries`.** It lazy-imports
   `tiled.queries.{Key, Regex, TimeRange}` on first call. If your environment
   is `tiled[client]` only (no full tiled install), you'll see a clear
   `ImportError` from this method. `browseCatalog` additionally requires
   `ipyaggrid` (Jupyter-only).

5. **`smi_defaults` import weight is preserved.** Standalone import was
   measured at ~100 ms (numpy-dominated) and pulls in **none** of the heavy
   modules listed in plan §6 (no `pyFAI`, `xarray`, `torch`, `tiled`,
   `skimage`, `scipy`, `PFGeneralIntegrator`, `WPIntegrator`,
   `SMISWAXSIntegrator`, `SMISWAXSLoader`).

6. **No changes to existing primitives.** The signatures and numerical output
   of `make_waxs_mask_callable`, `make_saxs_mask_from_spec`,
   `make_mask_for_angle`, `polygons_to_mask`, `shift_polygon`,
   `reduce_smi_combined`, and `reduce_smi_gi` are byte-for-byte unchanged.
   The `_BSX_PER_ARC_DEG = -4.39` literal still appears inside
   `reduce_smi_combined`; it equals `smi_defaults.BSX_PER_ARC_DEG` and is
   guarded by `test_loader_defaults_match_loader_module` and
   `test_bsx_per_arc_deg`.

---

*Implementation report authored 2026-04-29 in response to
[smi_browser_api_plan.md](smi_browser_api_plan.md).*
