"""Live tiled smoke test for the failing SMI run.

Loads SAXS and WAXS images from a specific run via the default
``TiledSMISWAXSLoader`` path (``smi/migration``) so we can quickly tell
whether the loader (or the tiled server) is responsible for the
intermittent 500 errors observed in the smi-browser GUI.

Skipped automatically when network / authentication / server is
unavailable.  Set ``PYHYPER_RUN_LIVE_TILED_TESTS=1`` to opt in; tests
are skipped by default to avoid hitting the production tiled server in
CI.

Running locally::

    PYHYPER_RUN_LIVE_TILED_TESTS=1 PYTHONPATH=src \
        pytest tests/test_SMISWAXSLoader_live_run.py -v -s
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# UID and expected shapes for the run that has been intermittently 500-ing
# on the SAXS detector when accessed via the GUI.
RUN_UID = "ab058833-7f75-4179-832d-3a63e629b555"
EXPECTED_WAXS_SHAPE = (14, 619, 1475)
EXPECTED_SAXS_SHAPE = (14, 1679, 1475)

LIVE = os.environ.get("PYHYPER_RUN_LIVE_TILED_TESTS") == "1"

pytestmark = pytest.mark.skipif(
    not LIVE,
    reason="Live tiled test; set PYHYPER_RUN_LIVE_TILED_TESTS=1 to enable.",
)


@pytest.fixture(scope="module")
def loader():
    from PyHyperScattering.SMISWAXSLoader import TiledSMISWAXSLoader
    # Use defaults: tiled.nsls2.bnl.gov + smi/migration
    return TiledSMISWAXSLoader()


@pytest.fixture(scope="module")
def run(loader):
    try:
        return loader._get_run(RUN_UID)
    except Exception as exc:  # pragma: no cover - environment-dependent
        pytest.skip(f"Cannot reach tiled run {RUN_UID}: {exc!r}")


def test_waxs_loads_via_default_loader(loader):
    t0 = time.time()
    da = loader.loadSingleImage(RUN_UID, detector="waxs")
    elapsed = time.time() - t0
    print(f"\nWAXS load: shape={da.shape} dtype={da.dtype} elapsed={elapsed:.1f}s")
    assert da.shape == EXPECTED_WAXS_SHAPE


def test_saxs_loads_via_default_loader(loader):
    t0 = time.time()
    da = loader.loadSingleImage(RUN_UID, detector="saxs")
    elapsed = time.time() - t0
    print(f"\nSAXS load: shape={da.shape} dtype={da.dtype} elapsed={elapsed:.1f}s")
    assert da.shape == EXPECTED_SAXS_SHAPE


def test_saxs_low_level_read(run):
    """Exercise the same _read_primary_field path the loader uses.

    Useful to isolate whether failures originate in the read helpers vs.
    downstream geometry/xarray construction.
    """
    from PyHyperScattering import SMISWAXSLoader as L
    t0 = time.time()
    arr = L._read_primary_field(run, L.SAXS_IMAGE_FIELD)
    elapsed = time.time() - t0
    print(f"\nSAXS _read_primary_field: shape={arr.shape} elapsed={elapsed:.1f}s")
    assert arr.shape == EXPECTED_SAXS_SHAPE


def test_saxs_via_reduce_smi_combined():
    """End-to-end path that the smi-browser GUI uses."""
    from PyHyperScattering.SMISWAXSIntegrator import reduce_smi_combined
    t0 = time.time()
    result = reduce_smi_combined(uid=RUN_UID)
    elapsed = time.time() - t0
    print(f"\nreduce_smi_combined elapsed={elapsed:.1f}s")
    assert result is not None
