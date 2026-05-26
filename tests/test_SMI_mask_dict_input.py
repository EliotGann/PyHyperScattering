"""Tests for the dict-input variant of SMI mask builders.

The public mask functions (``make_saxs_mask_from_spec``,
``make_waxs_mask_callable``, ``mask_for_frame``, plus the ``saxs_mask`` /
``waxs_mask`` kwargs on ``reduce_smi_combined`` and ``reduce_smi_gi``)
all accept either a JSON file path *or* a pre-parsed polygon dict.  The
dict variant exists so notebook users can compose and edit masks in
memory without writing temp files.

These tests assert that the two input forms produce *identical* masks
for the same data, and that the dispatch logic correctly bypasses the
file-path resolver when given a dict.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from PyHyperScattering import smi_defaults as D  # noqa: E402
from PyHyperScattering.SMISWAXSIntegrator import (  # noqa: E402
    _resolve_mask_spec,
    make_saxs_mask_from_dict,
    make_saxs_mask_from_spec,
    make_waxs_mask_callable,
    make_waxs_mask_callable_from_dict,
)


# ---------------------------------------------------------------------------
# _resolve_mask_spec dispatcher
# ---------------------------------------------------------------------------

class TestResolveMaskSpec:
    def test_passes_dict_through(self):
        spec = {"static_regions": {}, "beamstops": {}}
        assert _resolve_mask_spec(spec) is spec

    def test_loads_json_from_path(self, tmp_path):
        spec_in = {"static_regions": {"a": [[1, 1], [2, 1], [2, 2]]}}
        p = tmp_path / "m.json"
        p.write_text(json.dumps(spec_in))
        out = _resolve_mask_spec(p)
        assert out["static_regions"]["a"] == [[1, 1], [2, 1], [2, 2]]

    def test_loads_json_from_string_path(self, tmp_path):
        spec_in = {"static_regions": {"a": [[1, 1], [2, 1], [2, 2]]}}
        p = tmp_path / "m.json"
        p.write_text(json.dumps(spec_in))
        out = _resolve_mask_spec(str(p))
        assert out["static_regions"]["a"] == [[1, 1], [2, 1], [2, 2]]


# ---------------------------------------------------------------------------
# SAXS: dict vs path equivalence (bundled default)
# ---------------------------------------------------------------------------

class TestSAXSMaskDictEquivalence:
    """The bundled SAXS mask, fed as a Path vs a dict, must produce
    bitwise-identical masks."""

    @pytest.fixture
    def bundled(self):
        path = D.default_saxs_mask_path()
        with open(path) as f:
            spec = json.load(f)
        return path, spec, tuple(spec["image_shape"])

    def test_static_regions_only_match(self, bundled):
        path, spec, shape = bundled
        # Strip the beamstop block so the comparison isn't sensitive to
        # whether beam_center_px is supplied.
        spec_no_bs = {**spec, "beamstops": {}}
        m_path = make_saxs_mask_from_spec(
            image_shape=shape,
            mask_path={**spec_no_bs},  # path-input accepts dict transparently
            active_beamstop="rod",
        )
        m_dict = make_saxs_mask_from_dict(
            image_shape=shape,
            mask_spec=spec_no_bs,
            active_beamstop="rod",
        )
        np.testing.assert_array_equal(m_path, m_dict)

    def test_with_beamstop_match(self, bundled):
        path, spec, shape = bundled
        bc = (1170.0, 750.0)
        # Build from path (using the legacy from_spec entry point) and
        # from dict (new entry point); they must agree.
        m_path = make_saxs_mask_from_spec(
            image_shape=shape,
            mask_path=path,
            active_beamstop="rod",
            beam_center_px=bc,
        )
        m_dict = make_saxs_mask_from_dict(
            image_shape=shape,
            mask_spec=spec,
            active_beamstop="rod",
            beam_center_px=bc,
        )
        np.testing.assert_array_equal(m_path, m_dict)

    def test_dict_passes_through_make_saxs_mask_from_spec(self, bundled):
        """The unified entry point should accept a dict too."""
        path, spec, shape = bundled
        bc = (1170.0, 750.0)
        m_via_dict = make_saxs_mask_from_spec(
            image_shape=shape,
            mask_path=spec,                          # dict, not Path
            active_beamstop="rod",
            beam_center_px=bc,
        )
        m_via_path = make_saxs_mask_from_spec(
            image_shape=shape,
            mask_path=path,
            active_beamstop="rod",
            beam_center_px=bc,
        )
        np.testing.assert_array_equal(m_via_dict, m_via_path)


# ---------------------------------------------------------------------------
# WAXS: dict vs path equivalence
# ---------------------------------------------------------------------------

class TestWAXSMaskDictEquivalence:
    @pytest.fixture
    def bundled(self):
        path = D.default_waxs_mask_path()
        with open(path) as f:
            spec = json.load(f)
        return path, spec

    @pytest.mark.parametrize("theta, bsx", [
        (0.0, 0.0),
        (5.0, -22.0),
        (-3.0, 13.0),
    ])
    def test_callable_outputs_match(self, bundled, theta, bsx):
        path, spec = bundled
        shape = (619, 1475)
        fn_path = make_waxs_mask_callable(
            path, waxs_bsx_ref=0.0, beamstop_max_abs_arc_deg=15.0,
        )
        fn_dict = make_waxs_mask_callable_from_dict(
            spec, waxs_bsx_ref=0.0, beamstop_max_abs_arc_deg=15.0,
        )
        m_path = fn_path(shape, theta, bsx)
        m_dict = fn_dict(shape, theta, bsx)
        np.testing.assert_array_equal(m_path, m_dict)

    def test_dict_passes_through_make_waxs_mask_callable(self, bundled):
        """The unified entry point should accept a dict too."""
        path, spec = bundled
        shape = (619, 1475)
        fn_via_dict = make_waxs_mask_callable(
            spec, waxs_bsx_ref=0.0, beamstop_max_abs_arc_deg=15.0,
        )
        fn_via_path = make_waxs_mask_callable(
            path, waxs_bsx_ref=0.0, beamstop_max_abs_arc_deg=15.0,
        )
        np.testing.assert_array_equal(
            fn_via_dict(shape, 0.0, 0.0),
            fn_via_path(shape, 0.0, 0.0),
        )


# ---------------------------------------------------------------------------
# mask_for_frame accepts dict
# ---------------------------------------------------------------------------

class _ArrayNode:
    def __init__(self, arr):
        self._arr = np.asarray(arr)

    @property
    def shape(self):
        return self._arr.shape

    def read(self):
        return self._arr


class _PrimaryData:
    def __init__(self, fields):
        self._fields = {k: _ArrayNode(v) for k, v in fields.items()}

    def __getitem__(self, key):
        return self._fields[key]

    def __contains__(self, key):
        return key in self._fields


class _Primary:
    def __init__(self, fields):
        self._data = _PrimaryData(fields)

    def __getitem__(self, key):
        if key == "data":
            return self._data
        return self._data[key]

    def __contains__(self, key):
        return key in self._data


class _FakeRun:
    def __init__(self, fields):
        self._primary = _Primary(fields)
        self.metadata = {"start": {}}

    def __getitem__(self, key):
        if key == "primary":
            return self._primary
        raise KeyError(key)


class TestMaskForFrameDictInput:
    def test_saxs_dict_matches_path(self):
        from PyHyperScattering.SMISWAXSIntegrator import mask_for_frame

        saxs_shape = (1679, 1475)
        run = _FakeRun({"pil2M_image": np.zeros((1,) + saxs_shape, dtype=np.uint8)})
        path = D.default_saxs_mask_path()
        with open(path) as f:
            spec = json.load(f)

        m_path = mask_for_frame(run, 0, "saxs", mask_path=path)
        m_dict = mask_for_frame(run, 0, "saxs", mask_path=spec)
        np.testing.assert_array_equal(m_path, m_dict)

    def test_waxs_dict_matches_path(self):
        from PyHyperScattering.SMISWAXSIntegrator import mask_for_frame

        waxs_shape = (619, 1475)
        arc = np.array([0.0])
        bsx = np.array([10.0])
        run = _FakeRun({
            "pil900KW_image": np.zeros((1,) + waxs_shape, dtype=np.uint8),
            "waxs_arc": arc,
            "waxs_bsx": bsx,
        })
        path = D.default_waxs_mask_path()
        with open(path) as f:
            spec = json.load(f)

        m_path = mask_for_frame(run, 0, "waxs", mask_path=path)
        m_dict = mask_for_frame(run, 0, "waxs", mask_path=spec)
        np.testing.assert_array_equal(m_path, m_dict)


# ---------------------------------------------------------------------------
# In-memory polygon editing (the use case the dict input was added for)
# ---------------------------------------------------------------------------

class TestInMemoryPolygonEditing:
    def test_add_static_region_in_memory(self):
        """The whole point of dict input: edit polygons without a temp file."""
        path = D.default_saxs_mask_path()
        with open(path) as f:
            spec = json.load(f)
        shape = tuple(spec["image_shape"])

        # Baseline mask
        baseline = make_saxs_mask_from_dict(
            image_shape=shape,
            mask_spec=spec,
            active_beamstop="rod",
            beam_center_px=(1170.0, 750.0),
        )
        baseline_valid = int(np.count_nonzero(baseline))

        # Edit in place: add a 100x100 box that wasn't in the bundled spec
        edited = {**spec}
        edited["static_regions"] = dict(spec["static_regions"])
        edited["static_regions"]["my_blob"] = [
            [400, 400], [500, 400], [500, 500], [400, 500],
        ]
        edited_mask = make_saxs_mask_from_dict(
            image_shape=shape,
            mask_spec=edited,
            active_beamstop="rod",
            beam_center_px=(1170.0, 750.0),
        )
        edited_valid = int(np.count_nonzero(edited_mask))
        # The new blob masks a ~100×100 region; some of it may overlap
        # gap polygons already in the bundled mask, but we should see at
        # least several thousand newly-invalidated pixels.
        assert baseline_valid - edited_valid >= 5000
        # And the original valid pixels are still valid (subset)
        assert np.all(edited_mask <= baseline)
