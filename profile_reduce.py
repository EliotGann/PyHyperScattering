#!/usr/bin/env python
"""
Deep profiling of reduce_smi_combined.

Produces:
  1. Built-in timing dict from the function itself
  2. cProfile top-N callers sorted by cumulative time
  3. A pstats dump file for later inspection (profile_reduce.prof)
"""

import cProfile
import pstats
import io
import time
import sys

sys.path.insert(0, "src")

from PyHyperScattering.SMISWAXSIntegrator import reduce_smi_combined

CALL_KWARGS = dict(
    uid="5a531047-81b8-4599-ad8c-32cc756ac39d",
    tiled_uri="https://tiled.nsls2.bnl.gov",
    catalog="smi/migration",
    solid_angle_correction=True,
    geometry="transmission",
    saxs_mask_path=None,
    waxs_mask_path=None,
    cache_geometry=True,
    n_q=2000,
    waxs_beam_delta_px=(0.0, 0.1),
)

def main():
    print("=" * 72)
    print("  reduce_smi_combined profiling")
    print("=" * 72)
    print(f"\nCall kwargs:\n  {CALL_KWARGS}\n")

    # ── Run under cProfile ──
    profiler = cProfile.Profile()
    profiler.enable()
    wall_t0 = time.perf_counter()

    result = reduce_smi_combined(**CALL_KWARGS)

    wall_elapsed = time.perf_counter() - wall_t0
    profiler.disable()

    # ── 1. Built-in timing dict ──
    print("-" * 72)
    print("  Built-in timing (seconds)")
    print("-" * 72)
    timing = result.timing
    for key in ("tiled_load", "mask_setup", "saxs_integrate",
                "waxs_integrate", "merge", "total"):
        val = timing.get(key, 0.0)
        pct = 100.0 * val / timing["total"] if timing["total"] > 0 else 0
        bar = "#" * int(pct / 2)
        print(f"  {key:20s} {val:8.3f}s  ({pct:5.1f}%)  {bar}")
    print(f"\n  Wall-clock total:   {wall_elapsed:.3f}s")

    # ── 2. Extra info about data shapes ──
    print("\n" + "-" * 72)
    print("  Data shapes")
    print("-" * 72)
    if result.saxs is not None:
        ds = result.saxs.get("ds")
        if ds is not None:
            print(f"  SAXS intensity shape: {ds['intensity'].shape}")
    if result.waxs is not None:
        ds = result.waxs.get("ds")
        if ds is not None:
            print(f"  WAXS intensity shape: {ds['intensity'].shape}")
    if result.merged_qchi is not None:
        print(f"  Merged q-chi vars:   {list(result.merged_qchi.data_vars)}")
        for v in ("intensity", "I"):
            if v in result.merged_qchi:
                print(f"  Merged q-chi shape:  {result.merged_qchi[v].shape}")
                break

    # ── 3. cProfile stats ──
    print("\n" + "-" * 72)
    print("  cProfile – top 40 by cumulative time")
    print("-" * 72)
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s)
    ps.sort_stats("cumulative")
    ps.print_stats(40)
    print(s.getvalue())

    print("-" * 72)
    print("  cProfile – top 40 by total (self) time")
    print("-" * 72)
    s2 = io.StringIO()
    ps2 = pstats.Stats(profiler, stream=s2)
    ps2.sort_stats("tottime")
    ps2.print_stats(40)
    print(s2.getvalue())

    # ── 4. Dump for later analysis ──
    profiler.dump_stats("profile_reduce.prof")
    print(f"Full profile saved to profile_reduce.prof")
    print("  View with:  python -m pstats profile_reduce.prof")
    print("  Or:         snakeviz profile_reduce.prof")


if __name__ == "__main__":
    main()
