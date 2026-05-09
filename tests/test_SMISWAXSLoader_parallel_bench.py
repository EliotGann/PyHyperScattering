"""Benchmark: sequential vs. parallel (threaded) frame loading for SMI.

Compares the existing sequential per-frame tiled read against the new
ThreadPoolExecutor-based parallel path.  Because each frame is an
independent HTTP GET to the tiled server, parallelism should yield a
large speedup (limited by server concurrency, not local CPU).

Running locally::

    PYHYPER_RUN_LIVE_TILED_TESTS=1 PYTHONPATH=src \
        pytest tests/test_SMISWAXSLoader_parallel_bench.py -v -s

Set SMI_BENCH_WORKERS=<N> to override the default thread count (8).
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

RUN_UID = "ab058833-7f75-4179-832d-3a63e629b555"
EXPECTED_WAXS_SHAPE = (14, 619, 1475)
EXPECTED_SAXS_SHAPE = (14, 1679, 1475)

LIVE = os.environ.get("PYHYPER_RUN_LIVE_TILED_TESTS") == "1"
MAX_WORKERS = int(os.environ.get("SMI_BENCH_WORKERS", "8"))

pytestmark = pytest.mark.skipif(
    not LIVE,
    reason="Live tiled benchmark; set PYHYPER_RUN_LIVE_TILED_TESTS=1 to enable.",
)


@pytest.fixture(scope="module")
def loader():
    from PyHyperScattering.SMISWAXSLoader import TiledSMISWAXSLoader
    return TiledSMISWAXSLoader()


@pytest.fixture(scope="module")
def run(loader):
    try:
        return loader._get_run(RUN_UID)
    except Exception as exc:
        pytest.skip(f"Cannot reach tiled run {RUN_UID}: {exc!r}")


@pytest.fixture(scope="module")
def saxs_node(run):
    from PyHyperScattering.SMISWAXSLoader import (
        SAXS_IMAGE_FIELD,
        _get_primary_field_node,
    )
    return _get_primary_field_node(run, SAXS_IMAGE_FIELD)


@pytest.fixture(scope="module")
def waxs_node(run):
    from PyHyperScattering.SMISWAXSLoader import (
        WAXS_IMAGE_FIELD,
        _get_primary_field_node,
    )
    return _get_primary_field_node(run, WAXS_IMAGE_FIELD)


# ---------------------------------------------------------------------------
# WAXS benchmarks (~36 MB per frame × 14 frames)
# ---------------------------------------------------------------------------

def test_waxs_sequential(waxs_node):
    """WAXS: sequential frame-by-frame read (baseline)."""
    from PyHyperScattering.SMISWAXSLoader import _read_array_chunked

    t0 = time.time()
    arr = _read_array_chunked(waxs_node, parallel=False)
    elapsed = time.time() - t0
    print(f"\n[WAXS sequential] shape={arr.shape} elapsed={elapsed:.2f}s")
    assert arr.shape == EXPECTED_WAXS_SHAPE
    return elapsed


def test_waxs_parallel(waxs_node):
    """WAXS: parallel (threaded) frame read."""
    from PyHyperScattering.SMISWAXSLoader import _read_array_chunked

    t0 = time.time()
    arr = _read_array_chunked(waxs_node, parallel=True, max_workers=MAX_WORKERS)
    elapsed = time.time() - t0
    print(f"\n[WAXS parallel, {MAX_WORKERS} workers] shape={arr.shape} elapsed={elapsed:.2f}s")
    assert arr.shape == EXPECTED_WAXS_SHAPE
    return elapsed


# ---------------------------------------------------------------------------
# SAXS benchmarks (~99 MB per frame × 14 frames — triggers chunked path)
# ---------------------------------------------------------------------------

def test_saxs_sequential(saxs_node):
    """SAXS: sequential frame-by-frame read (baseline)."""
    from PyHyperScattering.SMISWAXSLoader import _read_array_chunked

    t0 = time.time()
    arr = _read_array_chunked(saxs_node, parallel=False)
    elapsed = time.time() - t0
    print(f"\n[SAXS sequential] shape={arr.shape} elapsed={elapsed:.2f}s")
    assert arr.shape == EXPECTED_SAXS_SHAPE
    return elapsed


def test_saxs_parallel(saxs_node):
    """SAXS: parallel (threaded) frame read."""
    from PyHyperScattering.SMISWAXSLoader import _read_array_chunked

    t0 = time.time()
    arr = _read_array_chunked(saxs_node, parallel=True, max_workers=MAX_WORKERS)
    elapsed = time.time() - t0
    print(f"\n[SAXS parallel, {MAX_WORKERS} workers] shape={arr.shape} elapsed={elapsed:.2f}s")
    assert arr.shape == EXPECTED_SAXS_SHAPE
    return elapsed


# ---------------------------------------------------------------------------
# End-to-end comparison via loadSingleImage (includes geometry resolution)
# ---------------------------------------------------------------------------

def test_end_to_end_comparison(loader):
    """Full loadSingleImage comparison: sequential vs parallel.

    Temporarily patches _read_array_chunked to force each mode.
    """
    import PyHyperScattering.SMISWAXSLoader as L

    original = L._read_array_chunked
    results = {}

    # Sequential
    L._read_array_chunked = lambda node: original(node, parallel=False)
    t0 = time.time()
    da_seq = loader.loadSingleImage(RUN_UID, detector="waxs")
    results["sequential"] = time.time() - t0

    # Parallel
    L._read_array_chunked = lambda node: original(node, parallel=True, max_workers=MAX_WORKERS)
    t0 = time.time()
    da_par = loader.loadSingleImage(RUN_UID, detector="waxs")
    results["parallel"] = time.time() - t0

    # Restore
    L._read_array_chunked = original

    speedup = results["sequential"] / results["parallel"] if results["parallel"] > 0 else float("inf")
    print(f"\n{'='*60}")
    print(f"End-to-end WAXS loadSingleImage:")
    print(f"  Sequential: {results['sequential']:.2f}s")
    print(f"  Parallel:   {results['parallel']:.2f}s")
    print(f"  Speedup:    {speedup:.1f}x")
    print(f"{'='*60}")

    # Verify identical results
    np.testing.assert_array_equal(da_seq.values, da_par.values)
