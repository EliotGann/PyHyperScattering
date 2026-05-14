"""Debug script for UID ff1bd407-5490-4327-ac63-2cd6a2cadbf8"""
import sys
sys.path.insert(0, 'src')
from PyHyperScattering.SMISWAXSLoader import (
    TiledSMISWAXSLoader, _read_scan_axis, _baseline_scalar,
    _read_baseline, _has_primary_field, _get_primary_field_node,
    _read_target_file_name_geometry,
    WAXS_ARC_FIELD, WAXS_BSX_FIELD, WAXS_IMAGE_FIELD,
    resolve_waxs_geometry, resolve_saxs_geometry, load_waxs_raw,
)
import numpy as np
from tiled.client import from_uri
from tiled.queries import FullText

uid = 'ff1bd407-5490-4327-ac63-2cd6a2cadbf8'

print("Connecting to tiled...")
client = from_uri("https://tiled.nsls2.bnl.gov")
cat = client["smi"]["migration"]

# Try direct key access first, then FullText search
try:
    run = cat[uid]
    print(f"Direct key access worked.\n")
except Exception as e1:
    print(f"Direct key access failed: {e1}")
    print("Trying FullText search...")
    results = cat.search(FullText(uid))
    if len(results) == 0:
        short_uid = uid[:8]
        results = cat.search(FullText(short_uid))
    if len(results) == 0:
        print("ERROR: Could not find run!")
        sys.exit(1)
    first_key = list(results)[0]
    run = results[first_key]
    print(f"Found via FullText search (key={first_key}).\n")

print('=== _read_scan_axis with NEW fallback order ===')
arc = _read_scan_axis(run, WAXS_ARC_FIELD)
print(f'  waxs_arc values (first 5): {arc[:5] if arc is not None and len(arc) > 5 else arc}')
print(f'  waxs_arc unique: {np.unique(arc) if arc is not None else None}')
print(f'  waxs_arc count: {len(arc) if arc is not None else 0}')
print()

energy = _read_scan_axis(run, "energy_energy")
print(f'  energy_energy (first 5 eV): {energy[:5] if energy is not None and len(energy) > 5 else energy}')
print(f'  energy range: {energy.min():.1f} - {energy.max():.1f} eV' if energy is not None and len(energy) > 1 else f'  energy: {energy}')
print(f'  energy count: {len(energy) if energy is not None else 0}')
print()

print('=== target_file_name geometry parsing ===')
tfn_geo = _read_target_file_name_geometry(run)
if tfn_geo is not None:
    print(f'  Parsed {len(tfn_geo)} frames')
    print(f'  Frame 0: {tfn_geo[0]}')
    print(f'  Frame 1: {tfn_geo[1]}')
    arcs = [g.get("waxs_arc_deg") for g in tfn_geo]
    print(f'  Unique arcs from filenames: {sorted(set(a for a in arcs if a is not None))}')
    energies = [g.get("energy_kev") for g in tfn_geo]
    print(f'  Energy range from filenames: {min(e for e in energies if e):.3f} - {max(e for e in energies if e):.3f} keV')
else:
    print('  target_file_name not available')
print()

print('=== Loading WAXS raw (with per-frame arc/energy) ===')
waxs_geo = resolve_waxs_geometry(run)
waxs_raw = load_waxs_raw(run, waxs_geo)
if waxs_raw is not None:
    print(f'  shape: {waxs_raw.shape}')
    print(f'  waxs_arc coord unique: {np.unique(waxs_raw.coords["waxs_arc"].values)}')
    print(f'  smi_energy_per_frame_ev (first 5): {waxs_raw.attrs.get("smi_energy_per_frame_ev", [])[:5]}')
    e_arr = np.array(waxs_raw.attrs.get("smi_energy_per_frame_ev", []))
    if len(e_arr) > 0:
        print(f'  energy range: {e_arr.min():.1f} - {e_arr.max():.1f} eV')
print()
print('=== DONE ===')


print('=== DONE ===')
