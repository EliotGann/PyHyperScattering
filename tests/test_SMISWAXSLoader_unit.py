"""CI-safe unit tests for the TiledSMISWAXSLoader public API.

These tests use in-process fakes that mimic the tiled run object
(``run.metadata['start']``, ``run['primary']['data'][field]``,
``run['baseline']['internal']``) so the full geometry-resolution and
image-loading code paths exercise without network or authentication.

For live-tiled smoke tests see ``test_SMISWAXSLoader_live_run.py``
(opt in with ``PYHYPER_RUN_LIVE_TILED_TESTS=1``).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from PyHyperScattering import SMISWAXSLoader as L  # noqa: E402


# ===========================================================================
# Fake tiled run infrastructure (no network)
# ===========================================================================

class _ArrayNode:
    """Stand-in for a tiled ArrayClient supporting ``.read()`` and slicing."""

    def __init__(self, arr):
        self._arr = np.asarray(arr)

    @property
    def shape(self):
        return self._arr.shape

    @property
    def dtype(self):
        return self._arr.dtype

    @property
    def chunks(self):
        # Mimic the "small chunks" case so bulk reads are preferred.
        if self._arr.ndim == 0:
            return ()
        return tuple((s,) for s in self._arr.shape)

    def read(self):
        return self._arr

    def __getitem__(self, key):
        return self._arr[key]


class _Container:
    """Dict-like container that supports both ``in`` and ``iter()``."""

    def __init__(self, fields):
        self._fields = {k: _ArrayNode(v) if not isinstance(v, _ArrayNode) else v
                        for k, v in fields.items()}

    def __getitem__(self, key):
        return self._fields[key]

    def __iter__(self):
        return iter(self._fields)

    def __contains__(self, key):
        return key in self._fields


class _PrimaryContainer:
    """Mimics ``run['primary']`` with a nested ``data`` container."""

    def __init__(self, fields, metadata=None):
        self._fields = _Container(fields)
        self.metadata = metadata or {"configuration": {}}

    def __getitem__(self, key):
        if key == "data":
            return self._fields
        return self._fields[key]

    def __iter__(self):
        return iter(self._fields)

    def __contains__(self, key):
        return key in self._fields

    def read(self):  # pragma: no cover - shouldn't be called in CI tests
        raise AssertionError("primary.read() must not be called in fake run")


class _BaselineInternal:
    """Mimics ``run['baseline']['internal']`` — a DataFrameClient-like."""

    def __init__(self, scalars):
        # scalars: dict[str, scalar] — converted to a 1-row DataFrame
        self._df = pd.DataFrame({k: [v] for k, v in scalars.items()})

    def __iter__(self):
        return iter(self._df.columns)

    def __getitem__(self, key):
        return _ArrayNode(self._df[key].to_numpy())

    def read(self):
        return self._df


class _BaselineContainer:
    """Mimics ``run['baseline']`` with the new ``['internal']`` layout."""

    def __init__(self, scalars):
        self._internal = _BaselineInternal(scalars)

    def __getitem__(self, key):
        if key == "internal":
            return self._internal
        raise KeyError(key)

    def read(self):
        # Old layout: returns an xr.Dataset directly.
        return xr.Dataset.from_dataframe(self._internal.read())


class _FakeRun:
    """Minimal tiled run mimic for CI tests."""

    def __init__(self, primary_fields=None, baseline=None, start=None):
        self._primary = _PrimaryContainer(primary_fields or {})
        self._baseline = _BaselineContainer(baseline or {})
        self.metadata = {"start": start or {"uid": "fake-uid", "scan_id": 1}}

    def __getitem__(self, key):
        if key == "primary":
            return self._primary
        if key == "baseline":
            return self._baseline
        raise KeyError(key)


@pytest.fixture(autouse=True)
def _clear_loader_caches():
    """Ensure module-level baseline/sort caches don't leak between tests."""
    L.clear_baseline_cache()
    yield
    L.clear_baseline_cache()


# ===========================================================================
# parse_sample_name_geometry — pure string parser
# ===========================================================================

class TestParseSampleNameGeometry:
    def test_waxs_arc(self):
        assert L.parse_sample_name_geometry("foo_wa20.5_bar") == {"waxs_arc_deg": 20.5}

    def test_sdd_metres(self):
        assert L.parse_sample_name_geometry("foo_sdd9.0m") == {"sdd_m": 9.0}

    def test_energy_kev(self):
        assert L.parse_sample_name_geometry("foo_16.10keV") == {"energy_kev": 16.10}

    def test_energy_ev_converts_to_kev(self):
        assert L.parse_sample_name_geometry("foo_4064.00eV") == {"energy_kev": 4.064}

    def test_incident_angle(self):
        assert L.parse_sample_name_geometry("foo_ai0.12_") == {"incident_angle_deg": 0.12}

    def test_theta(self):
        assert L.parse_sample_name_geometry("foo_th0.5_") == {"theta_deg": 0.5}

    def test_combined(self):
        result = L.parse_sample_name_geometry(
            "sample_wa20.0_sdd2.0m_16.10keV_ai0.12_"
        )
        assert result["waxs_arc_deg"] == 20.0
        assert result["sdd_m"] == 2.0
        assert result["energy_kev"] == 16.10
        assert result["incident_angle_deg"] == 0.12

    def test_no_geometry_returns_empty(self):
        assert L.parse_sample_name_geometry("plain_sample_name") == {}

    def test_empty_string(self):
        assert L.parse_sample_name_geometry("") == {}


# ===========================================================================
# resolve_saxs_geometry — fallback chain
# ===========================================================================

class TestResolveSAXSGeometry:
    def test_uses_baseline_when_primary_absent(self):
        run = _FakeRun(
            primary_fields={},
            baseline={
                "pil2M_beam_center_x_px": 750.0,
                "pil2M_beam_center_y_px": 1170.0,
                "pil2M_motor_z": 2050.0,
                "energy_energy": 16100.0,
            },
            start={"sample_name": "test"},
        )
        geo = L.resolve_saxs_geometry(run)
        # Beam centers: should pick up baseline (after default motor-x/y
        # correction of 0, since motor positions are absent → no offset).
        # Default deltas (BEAM_DELTA_ROW_PX=0, BEAM_DELTA_COL_PX=0) leave
        # the baseline values untouched.
        assert geo.beam_center_col_px == pytest.approx(750.0, abs=1.0)
        assert geo.beam_center_row_px == pytest.approx(1170.0, abs=1.0)
        # SDD: baseline 2050 mm + DISTANCE_DELTA_MM → metres.  The delta
        # is calibration-dependent and may shift when saxs_calibration.json
        # is regenerated, so check against the live value.
        expected_sdd_m = (2050.0 + L._SAXS_DEFAULT_DISTANCE_DELTA_MM) / 1000.0
        assert geo.dist_m == pytest.approx(expected_sdd_m, abs=0.01)
        # Energy: baseline 16100 eV → 16.1 keV
        assert geo.energy_ev == pytest.approx(16100.0)

    def test_override_takes_priority(self):
        run = _FakeRun(
            primary_fields={},
            baseline={
                "pil2M_beam_center_x_px": 750.0,
                "pil2M_beam_center_y_px": 1170.0,
            },
            start={"sample_name": "test"},
        )
        geo = L.resolve_saxs_geometry(
            run,
            beam_center_row_px=2000.0,
            beam_center_col_px=2500.0,
            distance_delta_mm=0.0,  # disable default correction for clarity
        )
        # The override is applied *before* delta corrections, and the absolute
        # value of the override is used (negative inputs flipped).
        assert geo.beam_center_row_px == 2000.0
        assert geo.beam_center_col_px == 2500.0

    def test_falls_back_to_default_distance(self):
        # Truly empty run — no metadata of any kind.
        run = _FakeRun(start={"sample_name": ""})
        geo = L.resolve_saxs_geometry(run)
        # Default SAXS distance (2000 mm) + delta → metres (delta is
        # calibration-dependent; check against the live value).
        expected_sdd_m = (2000.0 + L._SAXS_DEFAULT_DISTANCE_DELTA_MM) / 1000.0
        assert geo.dist_m == pytest.approx(expected_sdd_m, abs=0.01)

    def test_sample_name_energy_parsing(self):
        run = _FakeRun(start={"sample_name": "foo_4064.00eV_bar"})
        geo = L.resolve_saxs_geometry(run)
        # 4064 eV → 4.064 keV → 4064 eV
        assert geo.energy_ev == pytest.approx(4064.0)


# ===========================================================================
# resolve_waxs_geometry
# ===========================================================================

class TestResolveWAXSGeometry:
    def test_baseline_distance(self):
        run = _FakeRun(
            baseline={
                "pil900KW_motor_z": 273.0,
                "energy_energy": 16100.0,
            },
            start={},
        )
        geo = L.resolve_waxs_geometry(run)
        assert geo.dist_m == pytest.approx(0.273, abs=0.001)
        # WAXS beam center should fall back to calibrated defaults (NOT
        # baseline) per the loader's design note about coordinate-system
        # incompatibility with the rotated frame.
        assert geo.beam_center_row_px == 217.0
        assert geo.beam_center_col_px == pytest.approx(319.0 + (-4.5))

    def test_panel_geometry_defaults(self):
        run = _FakeRun(start={})
        geo = L.resolve_waxs_geometry(run)
        assert len(geo.panels) == 3
        assert geo.panels[0].offset_deg == -7.0
        assert geo.panels[1].offset_deg == 0.0
        assert geo.panels[2].offset_deg == 7.0


# ===========================================================================
# load_saxs_raw / load_waxs_raw — attrs contract
# ===========================================================================

class TestLoadSAXSRaw:
    def test_attrs_contract_minimum(self):
        images = np.zeros((3, 8, 10), dtype=np.int32)
        run = _FakeRun(
            primary_fields={L.SAXS_IMAGE_FIELD: images},
            start={"uid": "uid-x", "scan_id": 7, "sample_name": "samp"},
        )
        geo = L.SAXSGeometry(dist_m=2.0, poni1_m=0.2, poni2_m=0.13)
        da = L.load_saxs_raw(run, geo)

        # PyHyper/pyFAI geometry contract
        for key in ("dist", "poni1", "poni2", "rot1", "rot2", "rot3",
                    "pixel1", "pixel2", "energy", "wavelength"):
            assert key in da.attrs, f"missing {key} in attrs"

        # Run provenance
        assert da.attrs["uid"] == "uid-x"
        assert da.attrs["scan_id"] == 7
        assert da.attrs["sample_name"] == "samp"

        # SMI-specific
        assert da.attrs["smi_detector"] == "saxs_pil2M"

    def test_wavelength_attr_in_angstroms(self):
        """wavelength attr is in Ångstroms — that's what SMISWAXSIntegrator expects.

        SMISWAXSIntegrator.integrate_saxs reads ``attrs['wavelength']``
        and converts via ``× 1e-10`` to metres.  If we stored metres
        here, q-values would come out 10^10× too large.
        """
        images = np.zeros((1, 4, 5), dtype=np.int32)
        run = _FakeRun(primary_fields={L.SAXS_IMAGE_FIELD: images})
        geo = L.SAXSGeometry(
            dist_m=2.0, poni1_m=0.2, poni2_m=0.13,
            energy_ev=16100.0,
            wavelength_m=7.7008e-11,
        )
        da = L.load_saxs_raw(run, geo)
        # 16.1 keV ⇒ ~0.77 Å ⇒ attr should be order-unity, not 1e-11
        assert 0.1 < da.attrs["wavelength"] < 10, (
            "wavelength attr should be in Ångstroms (order ~1) "
            f"— got {da.attrs['wavelength']}"
        )
        assert da.attrs["wavelength"] == pytest.approx(0.77008, rel=1e-3)

    def test_dim_order_single_frame(self):
        run = _FakeRun(
            primary_fields={L.SAXS_IMAGE_FIELD: np.zeros((1, 5, 7), dtype=np.int32)},
        )
        geo = L.SAXSGeometry(dist_m=2.0, poni1_m=0.2, poni2_m=0.13)
        da = L.load_saxs_raw(run, geo)
        # After squeeze + reshape, single frame collapses to 2-D
        assert da.dims == ("pix_y", "pix_x")
        assert da.shape == (5, 7)

    def test_dim_order_multi_frame(self):
        run = _FakeRun(
            primary_fields={L.SAXS_IMAGE_FIELD: np.zeros((4, 5, 7), dtype=np.int32)},
        )
        geo = L.SAXSGeometry(dist_m=2.0, poni1_m=0.2, poni2_m=0.13)
        da = L.load_saxs_raw(run, geo)
        # Multi-frame -> (frame, pix_y, pix_x).
        assert "pix_y" in da.dims and "pix_x" in da.dims
        assert da.shape == (4, 5, 7)

    def test_no_none_attrs(self):
        """SAXS loader must not emit None-valued attrs (Tiled-serialization)."""
        run = _FakeRun(
            primary_fields={L.SAXS_IMAGE_FIELD: np.zeros((1, 4, 5), dtype=np.int32)},
            # Minimal start: no scan_id, no incident_angle — both default None
        )
        geo = L.SAXSGeometry(dist_m=2.0, poni1_m=0.2, poni2_m=0.13)
        da = L.load_saxs_raw(run, geo)
        nones = [k for k, v in da.attrs.items() if v is None]
        assert not nones, f"loader emitted None-valued attrs: {nones}"


# ===========================================================================
# End-to-end integration smoke test (catches q-unit regressions)
# ===========================================================================

class TestIntegrateSAXSEndToEnd:
    """Run ``integrate_saxs`` against a synthetic-but-physical fake run.

    These tests check that *units* (q in nm⁻¹) and *magnitudes* are
    sensible — the wavelength-attr unit regression (Ångstroms ↔ metres)
    silently corrupted q values by a factor of 10^10 without raising,
    producing all-NaN merged_iq.  A unit check at this layer catches
    that class of bug without requiring a tiled connection.
    """

    def test_q_grid_in_nm_inverse_units(self):
        """Resulting q grid must be in nm⁻¹ (order 0.01–100), not m⁻¹.

        Geometry: 16.1 keV at SDD=2 m on a Pilatus 2M produces a max q
        of about 5 nm⁻¹ at the detector corner.  Anything above ~1000
        means wavelength is in the wrong unit.
        """
        from PyHyperScattering.SMISWAXSIntegrator import integrate_saxs

        # 17×17 image so the integrator has enough pixels to bin.
        # The detector is small enough that the corner q is well-defined.
        ny, nx = 17, 17
        images = np.ones((1, ny, nx), dtype=np.float32)
        run = _FakeRun(primary_fields={L.SAXS_IMAGE_FIELD: images})

        # Realistic SMI SAXS at 16.1 keV, SDD=2 m, beam in middle
        # of the (tiny) detector.  Use the actual Pilatus pixel size.
        geo = L.SAXSGeometry(
            dist_m=2.0,
            poni1_m=(ny / 2) * L.PILATUS_PIXEL_SIZE_M,
            poni2_m=(nx / 2) * L.PILATUS_PIXEL_SIZE_M,
            pixel1_m=L.PILATUS_PIXEL_SIZE_M,
            pixel2_m=L.PILATUS_PIXEL_SIZE_M,
            energy_ev=16100.0,
            wavelength_m=7.7008e-11,
            beam_center_row_px=ny / 2,
            beam_center_col_px=nx / 2,
        )
        saxs_raw = L.load_saxs_raw(run, geo)

        result = integrate_saxs(
            saxs_raw=saxs_raw,
            mask=None,
            n_q=200, n_chi=90,
            beam_center_col_px=nx / 2,
            solid_angle_correction=False,
            dezinger_threshold=None,
            cache_geometry=False,
        )
        q = np.asarray(result["q_chi"]["q"].values, dtype=float)
        # For a 17×17 sub-detector at 2 m with λ ≈ 0.77 Å, corner q is
        # ~0.01 nm⁻¹.  We assert a generous physical range: q must be
        # in (0, 1000) nm⁻¹.  A unit mistake (10^10× error from a m/Å
        # mismatch) would put q at ~10^9 — instantly fails the upper
        # bound.
        assert q.max() > 0, f"q max should be positive, got {q.max()}"
        assert q.max() < 1000, (
            f"q max = {q.max():.3g} nm⁻¹ is unphysically large.  "
            f"This usually means attrs['wavelength'] is in the wrong "
            f"unit (loader stores metres but integrator expects Å, "
            f"or vice versa)."
        )

    def test_q_range_not_clipped_below_low_q(self):
        """q_min must reflect the actual minimum q in the unmasked image,
        not a percentile clip.

        USAXS at long SDD has its most important physics at the lowest q
        values; an earlier version clipped the bottom 0.5% via
        np.percentile, which silently lost the low-q range users came
        for.  q_min should be within a few percent of the true minimum
        of unmasked pixel q-values.
        """
        from PyHyperScattering.SMISWAXSIntegrator import integrate_saxs

        ny, nx = 51, 51
        # Place the beam center off-corner so the minimum pixel q
        # (1 pixel from beam) is well-defined and noticeably > 0.
        r0, c0 = 10.0, 10.0
        images = np.ones((1, ny, nx), dtype=np.float32) * 1000.0
        run = _FakeRun(primary_fields={L.SAXS_IMAGE_FIELD: images})
        geo = L.SAXSGeometry(
            dist_m=2.0,
            poni1_m=r0 * L.PILATUS_PIXEL_SIZE_M,
            poni2_m=c0 * L.PILATUS_PIXEL_SIZE_M,
            pixel1_m=L.PILATUS_PIXEL_SIZE_M,
            pixel2_m=L.PILATUS_PIXEL_SIZE_M,
            energy_ev=16100.0,
            wavelength_m=7.7008e-11,
            beam_center_row_px=r0,
            beam_center_col_px=c0,
        )
        saxs_raw = L.load_saxs_raw(run, geo)
        result = integrate_saxs(
            saxs_raw=saxs_raw, mask=None,
            n_q=200, n_chi=90,
            beam_center_col_px=c0,
            solid_angle_correction=False,
            dezinger_threshold=None,
            cache_geometry=False,
        )
        q = np.asarray(result["q_chi"]["q"].values, dtype=float)

        # Compute the *true* min q the integrator could see — the q of
        # the unmasked pixel closest to the beam.  Anything beyond ~2%
        # above this floor means a cutoff has crept back in.
        # Pixel (0, 0) is the corner; pixel (r0+1, c0) is one pixel below
        # the beam in the row direction, so its q corresponds to one
        # pixel worth of distance.
        from PyHyperScattering.SMISWAXSIntegrator import (
            wavelength_nm_from_energy_kev,
        )
        wavelength_nm = wavelength_nm_from_energy_kev(16.1)
        # Smallest non-zero pixel offset = 1 pixel
        pixel_size_mm = L.PILATUS_PIXEL_SIZE_M * 1000
        # q for 1 pixel away in small-angle limit:
        # theta ≈ pixel_size / SDD, q = (2π/λ) × sin(theta)
        sdd_mm = 2000.0
        theta = pixel_size_mm / sdd_mm
        q_one_pixel_nm = (2 * np.pi / wavelength_nm) * np.sin(theta / 2) * 2
        # When using the literal min of q_vals (which includes the beam
        # center pixel at q=0), q_grid[0] should be ~dq/2 — much smaller
        # than one pixel's q.  A percentile clip would push q_min up to
        # roughly the q of pixels a few rows out from beam.  Asserting
        # q.min() < q_one_pixel_nm cleanly separates the two regimes.
        assert q.min() < q_one_pixel_nm, (
            f"q_min = {q.min():.4g} nm⁻¹ is too high relative to "
            f"one-pixel q ({q_one_pixel_nm:.4g}) — a low-q cutoff has "
            f"likely crept back in (e.g. np.percentile)."
        )

    def test_integrate_saxs_via_attrs_consistent_with_loader(self):
        """Round-trip: integrating a freshly-loaded DataArray must yield
        finite intensities (catches attrs-vs-integrator unit mismatches)."""
        from PyHyperScattering.SMISWAXSIntegrator import integrate_saxs

        ny, nx = 17, 17
        # Use a small Gaussian-like intensity bump so binning produces
        # non-NaN bins with content.
        rr, cc = np.indices((ny, nx))
        r0, c0 = ny / 2, nx / 2
        bump = np.exp(-((rr - r0) ** 2 + (cc - c0) ** 2) / 8.0)
        images = bump[np.newaxis].astype(np.float32) * 1000.0
        run = _FakeRun(primary_fields={L.SAXS_IMAGE_FIELD: images})

        geo = L.SAXSGeometry(
            dist_m=2.0,
            poni1_m=r0 * L.PILATUS_PIXEL_SIZE_M,
            poni2_m=c0 * L.PILATUS_PIXEL_SIZE_M,
            pixel1_m=L.PILATUS_PIXEL_SIZE_M,
            pixel2_m=L.PILATUS_PIXEL_SIZE_M,
            energy_ev=16100.0,
            wavelength_m=7.7008e-11,
            beam_center_row_px=r0,
            beam_center_col_px=c0,
        )
        saxs_raw = L.load_saxs_raw(run, geo)
        result = integrate_saxs(
            saxs_raw=saxs_raw, mask=None,
            n_q=200, n_chi=90,
            beam_center_col_px=c0,
            solid_angle_correction=False,
            dezinger_threshold=None,
            cache_geometry=False,
        )
        I = np.asarray(result["iq"]["I"].values, dtype=float)
        finite_count = int(np.isfinite(I).sum())
        # Most q-bins should have data given a small detector
        assert finite_count > 0, (
            "All q-bins are NaN — likely a unit-mismatch between the "
            "loader's wavelength attr and the integrator's expectation."
        )


class TestLoadWAXSRaw:
    def test_smi_panels_is_json_string(self):
        """smi_panels attr is JSON-encoded to be netCDF/Zarr/Tiled-safe."""
        run = _FakeRun(
            primary_fields={
                L.WAXS_IMAGE_FIELD: np.zeros((2, 8, 10), dtype=np.int32),
                L.WAXS_ARC_FIELD: np.array([-7.0, 7.0]),
            },
        )
        geo = L.WAXSGeometry(
            dist_m=0.273,
            beam_center_row_px=217.0,
            beam_center_col_px=319.0,
            panels=(
                L.WAXSPanelGeometry(col_start=0, col_end=206, offset_deg=-7.0),
                L.WAXSPanelGeometry(col_start=206, col_end=413, offset_deg=0.0),
                L.WAXSPanelGeometry(col_start=413, col_end=619, offset_deg=7.0),
            ),
        )
        da = L.load_waxs_raw(run, geo)
        # Must be a string so xarray .to_netcdf / Tiled accept it.
        assert isinstance(da.attrs["smi_panels"], str)
        decoded = json.loads(da.attrs["smi_panels"])
        assert isinstance(decoded, list)
        assert len(decoded) == 3
        assert decoded[0]["offset_deg"] == -7.0
        assert decoded[1]["col_start"] == 206

    def test_smi_panels_survives_netcdf_roundtrip(self, tmp_path):
        """Bullet-proof check: the DataArray must write+read via netCDF
        unmodified (no None attrs, no nested objects).  This is the
        end-to-end contract for Tiled upload."""
        run = _FakeRun(
            primary_fields={
                L.WAXS_IMAGE_FIELD: np.zeros((1, 6, 8), dtype=np.int32),
                L.WAXS_ARC_FIELD: np.array([0.0]),
            },
        )
        geo = L.WAXSGeometry(
            dist_m=0.273,
            beam_center_row_px=217.0,
            beam_center_col_px=319.0,
            panels=(
                L.WAXSPanelGeometry(col_start=0, col_end=206, offset_deg=-7.0),
            ),
        )
        da = L.load_waxs_raw(run, geo)
        out = tmp_path / "waxs.nc"
        # No manual attr cleanup needed — load_waxs_raw drops None attrs.
        da.to_netcdf(out)
        roundtripped = xr.open_dataarray(out)
        assert isinstance(roundtripped.attrs["smi_panels"], str)
        decoded = json.loads(roundtripped.attrs["smi_panels"])
        assert decoded[0]["offset_deg"] == -7.0

    def test_no_none_attrs(self):
        """Loader must not emit None-valued attrs — netCDF/Zarr/Tiled reject them."""
        run = _FakeRun(
            primary_fields={
                L.WAXS_IMAGE_FIELD: np.zeros((1, 6, 8), dtype=np.int32),
                L.WAXS_ARC_FIELD: np.array([0.0]),
            },
        )
        geo = L.WAXSGeometry(
            dist_m=0.273,
            beam_center_row_px=217.0,
            beam_center_col_px=319.0,
            panels=(L.WAXSPanelGeometry(col_start=0, col_end=206, offset_deg=-7.0),),
        )
        da = L.load_waxs_raw(run, geo)
        nones = [k for k, v in da.attrs.items() if v is None]
        assert not nones, f"loader emitted None-valued attrs: {nones}"


# ===========================================================================
# TiledSMISWAXSLoader.loadSingleImage — public entry point
# ===========================================================================

class TestTiledLoaderPublicAPI:
    def _make_loader_with_fake_run(self, run, monkeypatch):
        loader = L.TiledSMISWAXSLoader()
        monkeypatch.setattr(loader, "_get_run", lambda uid: run)
        return loader

    def test_load_saxs_returns_dataarray(self, monkeypatch):
        run = _FakeRun(
            primary_fields={L.SAXS_IMAGE_FIELD: np.ones((2, 4, 5), dtype=np.int32)},
            start={"uid": "fake-uid", "sample_name": "t"},
        )
        loader = self._make_loader_with_fake_run(run, monkeypatch)
        da = loader.loadSingleImage("fake-uid", detector="saxs")
        assert isinstance(da, xr.DataArray)
        assert da.shape == (2, 4, 5)

    def test_load_waxs_returns_dataarray(self, monkeypatch):
        run = _FakeRun(
            primary_fields={
                L.WAXS_IMAGE_FIELD: np.ones((2, 4, 5), dtype=np.int32),
                L.WAXS_ARC_FIELD: np.array([-7.0, 7.0]),
            },
        )
        loader = self._make_loader_with_fake_run(run, monkeypatch)
        da = loader.loadSingleImage("fake-uid", detector="waxs")
        assert isinstance(da, xr.DataArray)
        # WAXS dim ordering: arc, pix_y, pix_x
        assert "pix_y" in da.dims
        assert "pix_x" in da.dims

    def test_missing_detector_returns_none(self, monkeypatch):
        run = _FakeRun(primary_fields={})  # neither saxs nor waxs present
        loader = self._make_loader_with_fake_run(run, monkeypatch)
        assert loader.loadSingleImage("fake-uid", detector="saxs") is None
        assert loader.loadSingleImage("fake-uid", detector="waxs") is None

    def test_invalid_detector_raises(self, monkeypatch):
        run = _FakeRun()
        loader = self._make_loader_with_fake_run(run, monkeypatch)
        with pytest.raises(ValueError, match="Unknown detector"):
            loader.loadSingleImage("fake-uid", detector="xrd")

    def test_load_run_returns_dict(self, monkeypatch):
        run = _FakeRun(
            primary_fields={
                L.SAXS_IMAGE_FIELD: np.ones((1, 3, 4), dtype=np.int32),
                L.WAXS_IMAGE_FIELD: np.ones((1, 3, 4), dtype=np.int32),
                L.WAXS_ARC_FIELD: np.array([0.0]),
            },
        )
        loader = self._make_loader_with_fake_run(run, monkeypatch)
        result = loader.loadRun("fake-uid")
        assert set(result.keys()) == {"saxs", "waxs"}
        assert isinstance(result["saxs"], xr.DataArray)
        assert isinstance(result["waxs"], xr.DataArray)


# ===========================================================================
# infer_detectors_and_steps
# ===========================================================================

class TestInferDetectorsAndSteps:
    def test_finds_saxs_image_field(self):
        n_frames = 5
        run = _FakeRun(
            primary_fields={
                L.SAXS_IMAGE_FIELD: np.zeros((n_frames, 10, 12), dtype=np.int32),
                "stage_th": np.array([0.0, 0.1, 0.2, 0.3, 0.4]),
            },
            start={"uid": "u", "scan_id": 1, "detectors": ["pil2M"]},
        )
        info = L.infer_detectors_and_steps(run)
        assert info["n_frames"] == n_frames
        assert L.SAXS_IMAGE_FIELD in info["detector_fields"]["saxs"]
        # stage_th varies, should appear as a step candidate
        step_names = [c["name"] for c in info["step_candidates"]]
        assert "stage_th" in step_names

    def test_skips_constant_scalars(self):
        run = _FakeRun(
            primary_fields={
                L.SAXS_IMAGE_FIELD: np.zeros((3, 4, 5), dtype=np.int32),
                "constant_motor": np.array([1.5, 1.5, 1.5]),  # not a step
            },
        )
        info = L.infer_detectors_and_steps(run)
        step_names = [c["name"] for c in info["step_candidates"]]
        assert "constant_motor" not in step_names

    def test_classifies_waxs_fields(self):
        run = _FakeRun(
            primary_fields={
                L.WAXS_IMAGE_FIELD: np.zeros((3, 4, 5), dtype=np.int32),
                L.WAXS_ARC_FIELD: np.array([-7.0, 0.0, 7.0]),
            },
        )
        info = L.infer_detectors_and_steps(run)
        assert L.WAXS_IMAGE_FIELD in info["detector_fields"]["waxs"]
        assert L.WAXS_ARC_FIELD in info["detector_fields"]["scan_axes"]


# ===========================================================================
# Module-level cache lifecycle
# ===========================================================================

class TestBaselineCache:
    def test_clear_baseline_cache_resets_state(self):
        L._BASELINE_CACHE["test-uid"] = xr.Dataset()
        L._BASELINE_COLUMNS_CACHE["test-uid"] = ["a"]
        L._TARGET_FILE_NAME_CACHE["test-uid"] = None
        L._PRIMARY_SEQ_SORT_CACHE["test-uid"] = None
        L.clear_baseline_cache()
        assert "test-uid" not in L._BASELINE_CACHE
        assert "test-uid" not in L._BASELINE_COLUMNS_CACHE
        assert "test-uid" not in L._TARGET_FILE_NAME_CACHE
        assert "test-uid" not in L._PRIMARY_SEQ_SORT_CACHE
