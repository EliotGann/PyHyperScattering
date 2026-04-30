"""Tests for ``SMISWAXSIntegrator.mask_for_frame`` (per-frame mask helper)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from PyHyperScattering import smi_defaults as D  # noqa: E402
from PyHyperScattering.SMISWAXSIntegrator import mask_for_frame  # noqa: E402


# ---------------------------------------------------------------------------
# Test fakes — no tiled / network
# ---------------------------------------------------------------------------

class _ArrayNode:
    """A minimal stand-in for a tiled ArrayClient: ``.shape`` + ``.read()``."""

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
        # Allow direct primary[<field>] access too (legacy layout)
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


# Bundled mask shapes
SAXS_RAW_SHAPE = (1679, 1475)
WAXS_RAW_SHAPE = (619, 1475)


# ---------------------------------------------------------------------------
# SAXS
# ---------------------------------------------------------------------------

def test_mask_for_frame_saxs_basic():
    fields = {
        "pil2M_image": np.zeros((3,) + SAXS_RAW_SHAPE, dtype=np.uint8),
    }
    run = _FakeRun(fields)
    mask = mask_for_frame(run, frame_idx=0, detector="saxs")
    assert mask.shape == SAXS_RAW_SHAPE
    assert mask.dtype == bool
    assert mask.any() and not mask.all()


def test_mask_for_frame_saxs_ignores_frame_idx():
    fields = {
        "pil2M_image": np.zeros((3,) + SAXS_RAW_SHAPE, dtype=np.uint8),
    }
    run = _FakeRun(fields)
    m0 = mask_for_frame(run, frame_idx=0, detector="saxs")
    m2 = mask_for_frame(run, frame_idx=2, detector="saxs")
    np.testing.assert_array_equal(m0, m2)


def test_mask_for_frame_saxs_orient_for_display():
    fields = {
        "pil2M_image": np.zeros((1,) + SAXS_RAW_SHAPE, dtype=np.uint8),
    }
    run = _FakeRun(fields)
    mask_raw = mask_for_frame(run, 0, "saxs", orient_for_display=False)
    mask_disp = mask_for_frame(run, 0, "saxs", orient_for_display=True)
    np.testing.assert_array_equal(mask_disp, np.flipud(mask_raw))


# ---------------------------------------------------------------------------
# WAXS
# ---------------------------------------------------------------------------

def test_mask_for_frame_waxs_bsx_ref_derivation():
    """When (arc, bsx) move along the SMI mechanical linkage, every frame
    must derive the same ``waxs_bsx_ref`` (= bsx − BSX_PER_ARC_DEG · arc).
    Verify equivalence with an explicit ``make_waxs_mask_callable`` build
    using the *constant* bsx_ref and matching per-frame motor positions."""
    from PyHyperScattering.SMISWAXSIntegrator import make_waxs_mask_callable

    arc = np.array([0.0, 3.0])
    bsx_ref = 10.0
    bsx = bsx_ref + D.BSX_PER_ARC_DEG * arc
    fields = {
        "pil900KW_image": np.zeros((arc.size,) + WAXS_RAW_SHAPE, dtype=np.uint8),
        "waxs_arc": arc,
        "waxs_bsx": bsx,
    }
    run = _FakeRun(fields)

    expected_fn = make_waxs_mask_callable(
        D.default_waxs_mask_path(),
        waxs_bsx_ref=bsx_ref,
        beamstop_max_abs_arc_deg=6.0,
    )
    for i in range(arc.size):
        actual = mask_for_frame(run, i, "waxs")
        expected = expected_fn(WAXS_RAW_SHAPE, theta_deg=arc[i], waxs_bsx=bsx[i])
        np.testing.assert_array_equal(actual, expected)


def test_mask_for_frame_waxs_shape():
    arc = np.array([0.0])
    bsx = np.array([10.0])
    fields = {
        "pil900KW_image": np.zeros((1,) + WAXS_RAW_SHAPE, dtype=np.uint8),
        "waxs_arc": arc,
        "waxs_bsx": bsx,
    }
    run = _FakeRun(fields)
    mask = mask_for_frame(run, 0, "waxs")
    # WAXS mask builder applies rot90+fliplr (= transpose); axes swap.
    assert mask.shape == WAXS_RAW_SHAPE[::-1]
    assert mask.dtype == bool


def test_mask_for_frame_waxs_orient_for_display_is_noop():
    """Per design, WAXS mask is already display-oriented; the
    ``orient_for_display`` flag must NOT double-orient."""
    arc = np.array([0.0])
    bsx = np.array([10.0])
    fields = {
        "pil900KW_image": np.zeros((1,) + WAXS_RAW_SHAPE, dtype=np.uint8),
        "waxs_arc": arc,
        "waxs_bsx": bsx,
    }
    run = _FakeRun(fields)
    raw = mask_for_frame(run, 0, "waxs", orient_for_display=False)
    disp = mask_for_frame(run, 0, "waxs", orient_for_display=True)
    np.testing.assert_array_equal(raw, disp)


def test_mask_for_frame_waxs_missing_motors_raises():
    fields = {
        "pil900KW_image": np.zeros((1,) + WAXS_RAW_SHAPE, dtype=np.uint8),
    }
    run = _FakeRun(fields)
    with pytest.raises(KeyError):
        mask_for_frame(run, 0, "waxs")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def test_mask_for_frame_invalid_detector():
    run = _FakeRun({"pil2M_image": np.zeros((1,) + SAXS_RAW_SHAPE, dtype=np.uint8)})
    with pytest.raises(ValueError):
        mask_for_frame(run, 0, "xrd")


def test_mask_for_frame_raw_shape_override():
    """``raw_shape`` should bypass any need to inspect the run object."""
    # Provide a run with no image field; rely on raw_shape override + bundled mask.
    run = _FakeRun({})
    mask = mask_for_frame(
        run, 0, "saxs", raw_shape=SAXS_RAW_SHAPE,
    )
    assert mask.shape == SAXS_RAW_SHAPE
