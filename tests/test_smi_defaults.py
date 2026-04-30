"""Tests for SMI default-asset onboarding (bundled masks, etc.)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from PyHyperScattering import smi_defaults as D  # noqa: E402


def test_default_saxs_mask_path_exists_and_parses():
    p = D.default_saxs_mask_path()
    assert p.exists(), f"bundled SAXS mask missing: {p}"
    spec = json.loads(p.read_text())
    assert "static_regions" in spec
    assert "beamstops" in spec
    assert "rod" in spec["beamstops"]


def test_default_waxs_mask_path_exists_and_parses():
    p = D.default_waxs_mask_path()
    assert p.exists(), f"bundled WAXS mask missing: {p}"
    spec = json.loads(p.read_text())
    # WAXS mask is the legacy flat polygon-dict format
    assert "beamstop" in spec
    assert isinstance(spec["beamstop"], list)


def test_resolve_mask_path_none_returns_default():
    saxs = D.resolve_mask_path(None, detector="saxs")
    waxs = D.resolve_mask_path(None, detector="waxs")
    assert saxs == D.default_saxs_mask_path()
    assert waxs == D.default_waxs_mask_path()


def test_resolve_mask_path_bare_filename_uses_bundle(tmp_path):
    """A bare legacy filename (no path) should resolve to the bundled mask.

    This preserves backward compatibility with smi-browser, which used to
    pass ``"pil2M_mask_polygons.json"`` as a string assuming a working
    directory.
    """
    cwd_before = Path.cwd()
    import os
    os.chdir(tmp_path)
    try:
        saxs = D.resolve_mask_path(D.DEFAULT_SAXS_MASK_NAME, detector="saxs")
        waxs = D.resolve_mask_path(D.DEFAULT_WAXS_MASK_NAME, detector="waxs")
        assert saxs == D.default_saxs_mask_path()
        assert waxs == D.default_waxs_mask_path()
    finally:
        os.chdir(cwd_before)


def test_resolve_mask_path_existing_file_passes_through(tmp_path):
    custom = tmp_path / "custom_mask.json"
    custom.write_text("{}")
    out = D.resolve_mask_path(custom, detector="saxs")
    assert out == custom.resolve()


def test_resolve_mask_path_invalid_detector():
    with pytest.raises(ValueError):
        D.resolve_mask_path(None, detector="xrd")


def test_make_saxs_mask_from_spec_with_bundled_default():
    """End-to-end: bundled SAXS default should drive the mask builder."""
    from PyHyperScattering.SMISWAXSIntegrator import make_saxs_mask_from_spec

    mask_path = D.default_saxs_mask_path()
    spec = json.loads(mask_path.read_text())
    image_shape = tuple(spec["image_shape"])  # (1679, 1475)
    mask = make_saxs_mask_from_spec(
        image_shape=image_shape,
        mask_path=mask_path,
        active_beamstop="rod",
    )
    assert mask.shape == image_shape
    assert mask.dtype == bool
    # Mask should mark some pixels as invalid (False) — gaps + beamstop
    assert mask.any() and not mask.all()


def test_make_waxs_mask_callable_with_bundled_default():
    from PyHyperScattering.SMISWAXSIntegrator import make_waxs_mask_callable

    fn = make_waxs_mask_callable(D.default_waxs_mask_path())
    mask = fn((619, 1475), theta_deg=0.0, waxs_bsx=0.0)
    # WAXS mask is rotated by k=3, so spatial dims swap.
    assert mask.shape == (1475, 619)
    assert mask.dtype == bool
    assert mask.any() and not mask.all()


# ---------------------------------------------------------------------------
# Detector classification
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name, expected", [
    ("pil2M_image", "saxs"),
    ("PIL2M_IMAGE", "saxs"),
    ("pilatus2M_image", "saxs"),
    ("saxs_image", "saxs"),
    ("pil900KW_image", "waxs"),
    ("900KW", "waxs"),
    ("waxs_image", "waxs"),
    ("WAXS", "waxs"),
    ("unknown_field", None),
    ("", None),
])
def test_classify_detector_field(name, expected):
    assert D.classify_detector_field(name) == expected


def test_classify_detector_field_none_input():
    assert D.classify_detector_field(None) is None


# ---------------------------------------------------------------------------
# Calibration constants
# ---------------------------------------------------------------------------

def test_loader_defaults_values():
    cal = D.LOADER_DEFAULTS
    assert cal.saxs_row_delta_px == 2.0
    assert cal.saxs_col_delta_px == 3.0
    assert cal.waxs_row_delta_px == 0.0
    assert cal.waxs_col_delta_px == -2.0
    assert cal.saxs_distance_delta_mm == -20.0


def test_bsx_per_arc_deg():
    assert D.BSX_PER_ARC_DEG == -4.39


def test_loader_defaults_match_loader_module():
    """Values in smi_defaults must mirror SMISWAXSLoader's private defaults."""
    from PyHyperScattering import SMISWAXSLoader as L
    cal = D.LOADER_DEFAULTS
    assert cal.saxs_row_delta_px == L._SAXS_DEFAULT_BEAM_DELTA_ROW_PX
    assert cal.saxs_col_delta_px == L._SAXS_DEFAULT_BEAM_DELTA_COL_PX
    assert cal.waxs_row_delta_px == L._WAXS_DEFAULT_BEAM_DELTA_ROW_PX
    assert cal.waxs_col_delta_px == L._WAXS_DEFAULT_BEAM_DELTA_COL_PX
    assert cal.saxs_distance_delta_mm == L._SAXS_DEFAULT_DISTANCE_DELTA_MM


# ---------------------------------------------------------------------------
# Display orientation
# ---------------------------------------------------------------------------

import numpy as np  # noqa: E402


def test_orient_frame_for_display_saxs_idempotent_pair():
    rng = np.random.default_rng(0)
    a = rng.integers(0, 100, size=(7, 11))
    twice = D.orient_frame_for_display(D.orient_frame_for_display(a, "saxs"), "saxs")
    np.testing.assert_array_equal(twice, a)


def test_orient_frame_for_display_waxs_idempotent_pair():
    rng = np.random.default_rng(1)
    a = rng.integers(0, 100, size=(7, 11))
    twice = D.orient_frame_for_display(D.orient_frame_for_display(a, "waxs"), "waxs")
    np.testing.assert_array_equal(twice, a)


def test_orient_frame_waxs_equals_transpose():
    a = np.arange(15).reshape(3, 5)
    np.testing.assert_array_equal(D.orient_frame_for_display(a, "waxs"), a.T)


def test_orient_frame_saxs_equals_flipud():
    a = np.arange(15).reshape(3, 5)
    np.testing.assert_array_equal(D.orient_frame_for_display(a, "saxs"), np.flipud(a))


@pytest.mark.parametrize("detector, shape, c, r", [
    ("saxs", (10, 14), 3, 4),
    ("saxs", (5, 5), 0, 0),
    ("waxs", (8, 12), 5, 2),
    ("waxs", (3, 7), 6, 1),
])
def test_orient_polygon_xy_inverse_roundtrip(detector, shape, c, r):
    x, y = D.orient_polygon_xy(c, r, detector, shape)
    c2, r2 = D.orient_polygon_xy_inverse(x, y, detector, shape)
    assert (c2, r2) == (float(c), float(r))


def test_orient_polygon_xy_matches_frame_placement_saxs():
    rows, cols = 5, 7
    arr = np.zeros((rows, cols))
    r, c = 2, 4
    arr[r, c] = 1.0
    oriented = D.orient_frame_for_display(arr, "saxs")
    x, y = D.orient_polygon_xy(c, r, "saxs", (rows, cols))
    # Bokeh-style: array index [row, col] is plotted at (x=col, y=rows-1-row)
    assert oriented[rows - 1 - r, c] == 1.0
    assert (x, y) == (float(c), float(rows - r))


def test_orient_polygon_xy_matches_frame_placement_waxs():
    rows, cols = 5, 7
    arr = np.zeros((rows, cols))
    r, c = 2, 4
    arr[r, c] = 1.0
    oriented = D.orient_frame_for_display(arr, "waxs")
    # transpose maps [r, c] -> [c, r]
    assert oriented[c, r] == 1.0
    x, y = D.orient_polygon_xy(c, r, "waxs", (rows, cols))
    assert (x, y) == (float(r), float(c))


def test_orient_validates_detector():
    with pytest.raises(ValueError):
        D.orient_frame_for_display(np.zeros((2, 2)), "xrd")
    with pytest.raises(ValueError):
        D.orient_polygon_xy(0, 0, "xrd", (2, 2))
    with pytest.raises(ValueError):
        D.orient_polygon_xy_inverse(0, 0, "xrd", (2, 2))


# ---------------------------------------------------------------------------
# Mask polygon I/O
# ---------------------------------------------------------------------------

def _normalized_keys_ok(m):
    assert set(m.keys()) == {"image_shape", "static_regions", "beamstops"}
    for poly in list(m["static_regions"].values()) + list(m["beamstops"].values()):
        for v in poly:
            assert len(v) == 2
            assert isinstance(v[0], float) and isinstance(v[1], float)


def test_load_mask_polygons_bundled_saxs():
    m = D.load_mask_polygons(D.default_saxs_mask_path())
    _normalized_keys_ok(m)
    assert m["image_shape"] == [1679, 1475]
    assert len(m["static_regions"]) >= 1
    assert "rod" in m["beamstops"]
    assert len(m["beamstops"]["rod"]) >= 3  # at least a triangle


def test_load_mask_polygons_bundled_waxs():
    m = D.load_mask_polygons(D.default_waxs_mask_path())
    _normalized_keys_ok(m)
    assert "beamstop" in m["beamstops"]
    # Static regions should include the gaps / bad module from the flat file.
    assert any("gap" in k for k in m["static_regions"])


def test_save_mask_polygons_roundtrip_bundled_saxs(tmp_path):
    m = D.load_mask_polygons(D.default_saxs_mask_path())
    out = tmp_path / "saxs_roundtrip.json"
    D.save_mask_polygons(m, out)
    m2 = D.load_mask_polygons(out)
    assert m2["image_shape"] == m["image_shape"]
    assert set(m2["static_regions"]) == set(m["static_regions"])
    assert set(m2["beamstops"]) == set(m["beamstops"])
    for k in m["static_regions"]:
        assert m2["static_regions"][k] == m["static_regions"][k]


def test_save_mask_polygons_roundtrip_bundled_waxs(tmp_path):
    m = D.load_mask_polygons(D.default_waxs_mask_path())
    out = tmp_path / "waxs_roundtrip.json"
    D.save_mask_polygons(m, out)
    m2 = D.load_mask_polygons(out)
    # WAXS bundled file has no image_shape; round-tripped file reuses nested
    # schema, so static_regions / beamstops should match.
    assert set(m2["static_regions"]) == set(m["static_regions"])
    assert set(m2["beamstops"]) == set(m["beamstops"])


def test_load_mask_polygons_flat_routes_beamstop_keys(tmp_path):
    flat = tmp_path / "flat.json"
    flat.write_text(json.dumps({
        "image_shape": [10, 12],
        "gap_left": [[0, 0], [1, 0], [1, 10], [0, 10]],
        "beamstop_main": [[2, 2], [3, 2], [3, 4], [2, 4]],
    }))
    m = D.load_mask_polygons(flat)
    assert m["image_shape"] == [10, 12]
    assert "gap_left" in m["static_regions"]
    assert "beamstop_main" in m["beamstops"]


def test_load_mask_polygons_unwraps_saxs_beamstop_wrapper(tmp_path):
    nested = tmp_path / "nested.json"
    nested.write_text(json.dumps({
        "image_shape": [100, 100],
        "static_regions": {},
        "beamstops": {
            "rod": {
                "polygon": [[1, 1], [2, 1], [2, 2], [1, 2]],
                "x_motor_key": "saxs_bsx",
            },
        },
    }))
    m = D.load_mask_polygons(nested)
    assert m["beamstops"]["rod"] == [[1.0, 1.0], [2.0, 1.0], [2.0, 2.0], [1.0, 2.0]]
