# SMI Tiled Geometry Parameter Handbook

> **Last verified**: 2026-04-02 using `smi/migration` catalog on `tiled.nsls2.bnl.gov`
>
> Reference scan: `ab058833-7f75-4179-832d-3a63e629b555` (scan_id 1102136)

## Overview

The SMI beamline at NSLS-II stores geometry and motor positions across three
locations in a tiled/bluesky run.  **No single location is guaranteed to
contain all parameters** — the presence depends on the scan type (count vs.
motor scan) and what was configured as a "read" device at acquisition time.

This document catalogs where each critical geometry parameter can be found and
provides a reliable fallback resolution order.

---

## Tiled Run Structure (bluesky-tiled-plugins)

```
run
├── metadata
│   └── start      ← dict of run-start metadata (always present)
├── baseline
│   └── internal   ← DataFrameClient, shape (2,) = [start_val, end_val]
│                     Contains ~564 columns (ALL motors/signals snapshot)
└── primary
    ├── internal   ← DataFrameClient, shape (N_frames,)
    │                 Only contains the scanned motor + triggered read fields
    ├── pil2M_image      ← ArrayClient (SAXS detector)
    └── pil900KW_image   ← ArrayClient (WAXS detector)
```

### Key Access Pattern (new bluesky-tiled-plugins)

```python
# Baseline (DataFrameClient — NOT xr.Dataset)
baseline_df = run["baseline"]["internal"]   # DataFrameClient
columns = list(baseline_df)                 # 564 column names
vals = baseline_df[field_name].read()       # numpy array, shape (2,)
start_val = vals[0]                         # value at scan start
end_val = vals[1]                           # value at scan end

# Primary (DataFrameClient for 1-D fields)
primary_df = run["primary"]["internal"]     # DataFrameClient
p_columns = list(primary_df)                # only fields read during scan
vals = primary_df[field_name].read()        # shape (N_frames,)

# NOTE: run["baseline"].read() raises KeyError('data') on this server.
# You must go through run["baseline"]["internal"] instead.
```

---

## Critical Geometry Parameters

### 1. WAXS Arc Angle (`waxs_arc`)

The arc motor angle of the 900KW WAXS detector.

| Location | Field Name | When Available | Shape |
|----------|-----------|----------------|-------|
| primary/internal | `waxs_arc` | Only when arc is the **scan motor** | (N,) — one per frame |
| baseline/internal | `waxs_arc` | **Always** (snapshot) | (2,) — [start, end] |
| baseline/internal | `pil900KW_motors_arc_user_setpoint` | Always | (2,) |
| start metadata | — | Never stored directly | — |
| sample_name | `_wa{value}_` pattern | When user follows naming convention | string parse |

**Fallback order**: primary → baseline → sample_name parse

**Example values**:
- Primary (arc scan): `[13.5, 14.0, 14.5, 15.0, ...]` (varying)
- Baseline (count scan): `[8.99967179, 8.99967179]` (constant)
- Sample name: `EG_abgtesting_16.10keV_wa09.0_sdd2.0m` → parse `09.0`

---

### 2. SAXS Sample-to-Detector Distance (`pil2M_motor_z`)

The SAXS detector distance in mm (Pilatus 2M position along beam).

| Location | Field Name | When Available | Shape |
|----------|-----------|----------------|-------|
| primary/internal | — | Rarely (only if z is scan motor) | — |
| baseline/internal | `pil2M_motor_z` | **Always** | (2,) — mm |
| baseline/internal | `pil2M_motor_z_user_setpoint` | Always | (2,) — mm |
| start metadata | — | Not stored | — |
| sample_name | `_sdd{value}m_` pattern | When user follows naming convention | string parse (metres) |

**Fallback order**: primary → baseline → sample_name parse

**Example values**:
- Baseline: `[2000.000056, 2000.000056]` (mm)
- Sample name: `_sdd2.0m` → parse `2.0` (metres) → `2000.0` mm

---

### 3. Incident Angle / Sample Theta (`stage_th`)

The sample incidence angle (grazing incidence geometry).

| Location | Field Name | When Available | Shape |
|----------|-----------|----------------|-------|
| primary/internal | `stage_th` | Only when theta is the **scan motor** | (N,) |
| baseline/internal | `stage_th` | **Always** | (2,) — degrees |
| baseline/internal | `stage_th_user_setpoint` | Always | (2,) |
| baseline/internal | `piezo_th` | Always (fine piezo stage) | (2,) |
| start metadata | — | Not stored directly | — |
| sample_name | `_ai{value}_` or `_th{value}_` pattern | When user follows convention | string parse |

**Fallback order**: primary → baseline(`stage_th`) → sample_name parse

**Example values**:
- Baseline (scan 898856): `stage_th: [0.0, 0.0]`, `piezo_th: [-0.236485, -0.236484]`
- Sample name: `test_test_Ni_th0_z8300` → parse `0` degrees

---

### 4. Photon Energy (`energy_energy`)

The monochromator energy in eV.

| Location | Field Name | When Available | Shape | Units |
|----------|-----------|----------------|-------|-------|
| primary/internal | `energy_energy` | Only if energy is scan motor | (N,) | eV |
| baseline/internal | `energy_energy` | **Always** | (2,) | eV |
| baseline/internal | `energy_energy_setpoint` | Always | (2,) | eV |
| start metadata | `energy` | Sometimes | scalar | **keV** |
| sample_name | `_{value}keV_` pattern | When user follows convention | string parse | keV |

**Fallback order**: primary → baseline → start metadata → sample_name parse

**Example values**:
- Baseline: `[16099.99500545, 16099.99500545]` (eV) → 16.1 keV
- Start metadata: `start["energy"]` = not present in test scans
- Sample name: `_16.10keV_` → parse `16.10` (keV)

---

### 5. SAXS Beam Center (`pil2M_beam_center_x_px`, `pil2M_beam_center_y_px`)

The SAXS detector beam center in pixels.

| Location | Field Name | When Available | Shape |
|----------|-----------|----------------|-------|
| baseline/internal | `pil2M_beam_center_x_px` | **Always** | (2,) — pixels |
| baseline/internal | `pil2M_beam_center_y_px` | **Always** | (2,) — pixels |
| primary/internal | — | Never (not a scan motor) | — |

**Example values**:
- `pil2M_beam_center_x_px: [746.575, 746.575]`
- `pil2M_beam_center_y_px: [1165.13604651, 1165.13604651]`

---

### 6. WAXS Beamstop Position (`waxs_bsx`, `waxs_bsy`)

| Location | Field Name | When Available | Shape |
|----------|-----------|----------------|-------|
| primary/internal | `waxs_bsx` | When arc is scan motor (read along with arc) | (N,) |
| primary/internal | `waxs_bsy` | Same | (N,) |
| baseline/internal | `waxs_bsx` | **Always** | (2,) |
| baseline/internal | `waxs_bsy` | **Always** | (2,) |
| baseline/internal | `pil900KW_motors_bs_x_user_setpoint` | Always | (2,) |
| baseline/internal | `pil900KW_motors_bs_y_user_setpoint` | Always | (2,) |

---

### 7. SAXS Beamstop Position

| Location | Field Name | When Available | Shape |
|----------|-----------|----------------|-------|
| baseline/internal | `saxs_beamstop_x_rod` | Always | (2,) |
| baseline/internal | `saxs_beamstop_y_rod` | Always | (2,) |
| baseline/internal | `saxs_beamstop_x_pin` | Always | (2,) |
| baseline/internal | `saxs_beamstop_y_pin` | Always | (2,) |
| baseline/internal | `pil2M_active_beamstop` | Always — `'rod'` or `'pin'` | (2,) |

---

### 8. Sample Stage Position

| Location | Field Name | When Available | Shape |
|----------|-----------|----------------|-------|
| baseline/internal | `stage_x` | Always | (2,) — mm |
| baseline/internal | `stage_y` | Always | (2,) — mm |
| baseline/internal | `stage_z` | Always | (2,) — mm |
| baseline/internal | `stage_th` | Always | (2,) — degrees |
| baseline/internal | `stage_ph` | Always | (2,) — degrees |
| baseline/internal | `stage_ch` | Always | (2,) — degrees |
| sample_name | `_z{value}_` | Sometimes | string parse |

---

## Sample Name Encoding Convention

Users at SMI commonly encode geometry in the `sample_name` field using these patterns:

| Pattern | Meaning | Units | Example |
|---------|---------|-------|---------|
| `_wa{X}_` or `_wa{X}` | WAXS arc angle | degrees | `_wa09.0_` → 9.0° |
| `_sdd{X}m_` or `_sdd{X}m` | Sample-detector distance | metres | `_sdd2.0m` → 2.0 m |
| `_{X}keV_` or `_{X}keV` | Photon energy | keV | `_16.10keV_` → 16.10 keV |
| `_ai{X}_` or `_ai{X}` | Incident angle | degrees | `_ai0.12_` → 0.12° |
| `_th{X}_` or `_th{X}` | Sample theta | degrees | `_th0_` → 0° |
| `_z{X}_` | Sample z position | µm or mm | `_z8300` → 8300 |

### Regex patterns for parsing:

```python
import re

def parse_sample_name(sample_name: str) -> dict:
    """Extract geometry parameters encoded in the sample_name string."""
    result = {}
    
    # WAXS arc angle: _wa20.0_ or _wa20.0 (at end)
    m = re.search(r'_wa([\d.]+)', sample_name)
    if m:
        result['waxs_arc_deg'] = float(m.group(1))
    
    # Sample-detector distance: _sdd2.0m or _sdd2.0m_ (value in metres)
    m = re.search(r'_sdd([\d.]+)m?', sample_name)
    if m:
        result['sdd_m'] = float(m.group(1))
    
    # Photon energy: _16.10keV_
    m = re.search(r'_([\d.]+)keV', sample_name)
    if m:
        result['energy_kev'] = float(m.group(1))
    
    # Incident angle: _ai0.12_
    m = re.search(r'_ai([\d.]+)', sample_name)
    if m:
        result['incident_angle_deg'] = float(m.group(1))
    
    # Sample theta: _th0.5_
    m = re.search(r'_th([\d.]+)', sample_name)
    if m:
        result['theta_deg'] = float(m.group(1))
    
    return result
```

---

## Recommended Fallback Resolution Order

For any geometry parameter, the resolution priority should be:

1. **User override** (passed explicitly to the loader/integrator)
2. **Primary stream** (per-frame values — only present if that motor was scanned)
3. **Baseline stream** (snapshot at scan start — always present for all motors)
4. **Start metadata** (sometimes populated, e.g. `start["energy"]`)
5. **Sample name parsing** (last resort — relies on user naming discipline)
6. **Hardcoded defaults** (calibrated fallbacks for the instrument)

### Why this order?

- **Primary** has per-frame variation (e.g., arc angles during an arc scan)
- **Baseline** always has a snapshot value but is a single scalar (start-of-scan)
- **Start metadata** is inconsistently populated across proposals
- **Sample name** is set by users and may be wrong or absent
- **Defaults** are calibration-era values that drift over time

---

## Tiled API Notes

### Authentication
```python
from tiled.client import from_uri
client = from_uri("https://tiled.nsls2.bnl.gov")
cat = client["smi"]["migration"]
# Login is interactive (username/password prompt)
# Token expires after ~5 min of inactivity
```

### Supported Query Types
- `FullText("substring")` — searches across start metadata text fields
- `Key("field") == value` — exact match on start metadata keys  
- `Eq("field", value)` — **NOT supported** on this catalog
- `Regex("field", "pattern")` — **NOT supported** on this catalog

### Common Gotchas
1. `run["baseline"].read()` **raises `KeyError('data')`** — use
   `run["baseline"]["internal"]` instead (returns `DataFrameClient`)
2. Baseline shape is always `(2,)` — `[value_at_start, value_at_end]`
3. Primary images are under `run["primary"]["pil2M_image"]` (ArrayClient),
   NOT inside `internal`
4. The `waxs_arc` field in primary has only 14 frames for a 64-point scan
   when the detector is only triggered every N steps
5. `pil2M_motor_z` is in **mm** (not metres) — the value `2000.0` = 2.0 m
6. `energy_energy` is in **eV** (not keV) — the value `16100` = 16.1 keV
7. Catalog iteration (`cat.items()`) **times out** on large catalogs —
   always use search queries with limits

---

## Verified Field Names (Baseline)

From scan `ab058833-7f75-4179-832d-3a63e629b555` (564 total baseline columns):

### Detector Geometry
| Field | Example Value | Units |
|-------|--------------|-------|
| `pil2M_motor_z` | 2000.0 | mm |
| `pil2M_motor_z_user_setpoint` | 2000.0 | mm |
| `pil2M_motor_x` | 0.0128 | mm |
| `pil2M_motor_y` | 9.9994 | mm |
| `pil2M_beam_center_x_px` | 746.575 | pixels |
| `pil2M_beam_center_y_px` | 1165.136 | pixels |
| `pil2M_active_beamstop` | 'rod' | string |
| `waxs_arc` | 8.9997 | degrees |
| `pil900KW_motors_arc_user_setpoint` | 9.0 | degrees |
| `waxs_bsx` | -97.549 | mm |
| `waxs_bsy` | -0.050 | mm |

### Monochromator / Energy
| Field | Example Value | Units |
|-------|--------------|-------|
| `energy_energy` | 16099.995 | eV |
| `energy_energy_setpoint` | 16099.995 | eV |
| `energy_bragg` | 7.068 | degrees |
| `energy_ivugap` | 6781.5 | µm |
| `energy_harmonic` | 21 | integer |
| `dcm_config_theta` | 7.068 | degrees |

### Sample Stage
| Field | Example Value | Units |
|-------|--------------|-------|
| `stage_x` | 0.0 | mm |
| `stage_y` | 0.0 | mm |
| `stage_z` | 0.0 | mm |
| `stage_th` | 0.0 | degrees |
| `stage_ph` | 0.0 | degrees |
| `stage_ch` | 0.0 | degrees |
| `piezo_th` | -0.236 | degrees |

### Beamstops
| Field | Example Value | Units |
|-------|--------------|-------|
| `saxs_beamstop_x_rod` | 6.799 | mm |
| `saxs_beamstop_y_rod` | 3.440 | mm |
| `saxs_beamstop_x_pin` | 0.0002 | mm |
| `saxs_beamstop_y_pin` | 7.400 | mm |

### Optics (less commonly needed)
| Field | Example Value | Units |
|-------|--------------|-------|
| `crl_th` | -0.016 | degrees |
| `hfm_th` | -0.170 | degrees |
| `vdm_th` | -0.210 | degrees |
| `vfm_th` | -0.216 | degrees |
