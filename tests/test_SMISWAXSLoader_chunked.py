"""Tests for the chunked-read fallback in SMISWAXSLoader.

Reproduces the scenario where ``tiled`` returns an HTTP 500 for a bulk
read of a multi-frame detector array (observed for the SMI ``pil2M_image``
field at e.g. uid ``ab058833-7f75-4179-832d-3a63e629b555``) and verifies
that the loader falls back to a frame-by-frame read and returns the
expected array.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from PyHyperScattering import SMISWAXSLoader as L  # noqa: E402


class _FakeArrayNode:
    """Minimal stand-in for a tiled ArrayClient.

    ``read()`` raises an HTTP-500-like error; per-frame slicing succeeds.
    """

    def __init__(self, data: np.ndarray, fail_bulk: bool = True):
        self._data = np.asarray(data)
        self._fail_bulk = fail_bulk
        self.shape = self._data.shape
        self.bulk_calls = 0
        self.slice_calls = 0

    def read(self):
        self.bulk_calls += 1
        if self._fail_bulk:
            raise RuntimeError(
                "Server error '500 Internal Server Error' for url "
                "'https://tiled.example/api/v1/array/full/...'"
            )
        return self._data

    def __getitem__(self, key):
        self.slice_calls += 1
        return self._data[key]


class _FakePrimaryContainer:
    """Mimics ``run['primary']`` with a nested ``data`` container."""

    def __init__(self, fields: dict[str, _FakeArrayNode]):
        self._fields = fields
        self.data = _FakeDataContainer(fields)

    def __getitem__(self, key):
        if key == "data":
            return self.data
        if key in self._fields:
            return self._fields[key]
        raise KeyError(key)

    def __iter__(self):
        return iter(self._fields)

    def read(self):  # pragma: no cover - should not be called in these tests
        raise AssertionError(
            "primary.read() must not be called when chunked fallback is used"
        )


class _FakeDataContainer:
    def __init__(self, fields: dict[str, _FakeArrayNode]):
        self._fields = fields

    def __getitem__(self, key):
        return self._fields[key]

    def __iter__(self):
        return iter(self._fields)

    def __contains__(self, key):
        return key in self._fields


class _FakeRun:
    def __init__(self, fields: dict[str, _FakeArrayNode], start: dict | None = None):
        self._primary = _FakePrimaryContainer(fields)
        self.metadata = {"start": start or {"uid": "fake-uid", "scan_id": 1}}

    def __getitem__(self, key):
        if key == "primary":
            return self._primary
        raise KeyError(key)


def _make_saxs_frames(n_frames: int = 7, h: int = 1679, w: int = 1475) -> np.ndarray:
    """Build a deterministic small fake of the failing (7, 1679, 1475) array."""
    # Use a tiny pattern that only depends on frame index so we can verify
    # the chunked read reassembled in the correct order without consuming
    # gigabytes of memory.
    out = np.zeros((n_frames, h, w), dtype=np.int32)
    for i in range(n_frames):
        out[i, 0, 0] = i + 1
        out[i, -1, -1] = (i + 1) * 10
    return out


def test_read_primary_field_falls_back_to_frame_by_frame():
    data = _make_saxs_frames()
    node = _FakeArrayNode(data, fail_bulk=True)
    run = _FakeRun({L.SAXS_IMAGE_FIELD: node})

    arr = L._read_primary_field(run, L.SAXS_IMAGE_FIELD)

    assert arr.shape == data.shape
    np.testing.assert_array_equal(arr, data)
    # Bulk attempt happened once, then n_frames per-frame requests
    assert node.bulk_calls == 1
    assert node.slice_calls == data.shape[0]


def test_read_primary_field_uses_bulk_when_available():
    data = _make_saxs_frames(n_frames=3, h=4, w=5)
    node = _FakeArrayNode(data, fail_bulk=False)
    run = _FakeRun({L.SAXS_IMAGE_FIELD: node})

    arr = L._read_primary_field(run, L.SAXS_IMAGE_FIELD)

    np.testing.assert_array_equal(arr, data)
    assert node.bulk_calls == 1
    assert node.slice_calls == 0


def test_read_primary_field_non_500_error_propagates(monkeypatch):
    # Both bulk and per-frame reads fail — the chunked fallback should
    # exhaust its retries and surface the underlying error.
    monkeypatch.setattr(L, "_PER_FRAME_RETRIES", 1)

    class _BadNode(_FakeArrayNode):
        def read(self, slice=None):
            raise RuntimeError("permission denied")

        def __getitem__(self, key):
            raise RuntimeError("permission denied")

    node = _BadNode(np.zeros((2, 3, 4)), fail_bulk=True)
    run = _FakeRun({L.SAXS_IMAGE_FIELD: node})
    with pytest.raises(RuntimeError, match="permission denied"):
        L._read_primary_field(run, L.SAXS_IMAGE_FIELD)

def test_load_saxs_raw_via_chunked_fallback():
    """End-to-end: loadSingleImage path works when bulk read 500s."""
    n_frames = 7
    # Use small spatial dims so the test stays fast; the loader doesn't
    # care about absolute pixel counts, only dimensionality.
    data = _make_saxs_frames(n_frames=n_frames, h=8, w=10)
    node = _FakeArrayNode(data, fail_bulk=True)
    run = _FakeRun(
        {L.SAXS_IMAGE_FIELD: node},
        start={"uid": "ab058833-7f75-4179-832d-3a63e629b555", "scan_id": 42},
    )

    geo = L.SAXSGeometry(
        dist_m=2.0,
        poni1_m=0.2,
        poni2_m=0.13,
    )
    da = L.load_saxs_raw(run, geo)

    assert da.ndim == 3
    assert da.shape == (n_frames, 8, 10)
    np.testing.assert_array_equal(da.values, data)
    assert da.attrs["uid"] == "ab058833-7f75-4179-832d-3a63e629b555"
    assert da.attrs["smi_detector"] == "saxs_pil2M"


def test_has_primary_field_does_not_read_bulk():
    data = _make_saxs_frames(n_frames=2, h=4, w=5)
    node = _FakeArrayNode(data, fail_bulk=True)
    run = _FakeRun({L.SAXS_IMAGE_FIELD: node})

    assert L._has_primary_field(run, L.SAXS_IMAGE_FIELD) is True
    assert L._has_primary_field(run, "nonexistent_field") is False
    # Critically: introspection must not have triggered a bulk read.
    assert node.bulk_calls == 0
