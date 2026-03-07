#!/usr/bin/env python3
"""
Compute zooplankton R* (resource threshold) from a PlankTOM model run.

    R* = (m · K) / (GE · g - m)

Parameters read from namelist.trc.sms:
  g  = rn_grazoo   maximum grazing rate      (d⁻¹)
  K  = rn_grkzoo   half-saturation constant  (mol C L⁻¹)
  GE = rn_ggezoo   gross growth efficiency   (—)
  m  = rn_reszoo + D   total loss rate       (d⁻¹)

Usage:
  python rstar_zoo.py TOM12_RY_JRA2 [--dilution D]
  python rstar_zoo.py /full/path/to/run_dir [--dilution D]
"""

import argparse
import sys
import warnings
from pathlib import Path

try:
    import f90nml
except ImportError:
    print("Error: f90nml not found. Install with: pip install f90nml", file=sys.stderr)
    sys.exit(1)

MODELRUNS_BASE = Path('/gpfs/home/vhf24tbu/scratch/ModelRuns')
ZOO_NAMES = ['PRO', 'PTE', 'MES', 'GEL', 'CRU']


def load_namelist(filepath: Path) -> dict:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return f90nml.read(filepath)


def get_param(nml: dict, param: str):
    """Search all namelist groups for a parameter."""
    for group in nml.values():
        if isinstance(group, dict) and param in group:
            return group[param]
    return None


def to_list(val) -> list:
    return list(val) if isinstance(val, (list, tuple)) else [val]


def pad(lst: list, n: int) -> list:
    return (lst * n)[:n] if len(lst) == 1 else lst[:n]


def main():
    parser = argparse.ArgumentParser(
        description='Compute zooplankton R* from a PlankTOM model run'
    )
    parser.add_argument('run', help='Run name (e.g. TOM12_RY_JRA2) or full path to run directory')
    parser.add_argument('--dilution', '-d', type=float, default=0.0,
                        metavar='D',
                        help='Dilution rate D (d⁻¹) added to rn_reszoo (default: 0)')
    args = parser.parse_args()

    run_dir = Path(args.run)
    if not run_dir.is_dir():
        run_dir = MODELRUNS_BASE / args.run
    if not run_dir.is_dir():
        print(f"Error: '{args.run}' not found as a path or in {MODELRUNS_BASE}", file=sys.stderr)
        sys.exit(1)

    nml_path = run_dir / 'namelist.trc.sms'
    if not nml_path.exists():
        print(f"Error: namelist.trc.sms not found in {run_dir}", file=sys.stderr)
        sys.exit(1)

    nml = load_namelist(nml_path)

    params = {}
    for name in ('rn_reszoo', 'rn_grkzoo', 'rn_ggezoo', 'rn_grazoo'):
        val = get_param(nml, name)
        if val is None:
            print(f"Error: '{name}' not found in {nml_path.name}", file=sys.stderr)
            sys.exit(1)
        params[name] = to_list(val)

    n_zoo = max(len(v) for v in params.values())
    zoo_names = ZOO_NAMES[:n_zoo]

    rn_reszoo = pad(params['rn_reszoo'], n_zoo)
    rn_grkzoo = pad(params['rn_grkzoo'], n_zoo)
    rn_ggezoo = pad(params['rn_ggezoo'], n_zoo)
    rn_grazoo = pad(params['rn_grazoo'], n_zoo)

    D = args.dilution

    print(f"\nZooplankton R*  [{run_dir.name}]")
    if D:
        print(f"Dilution D = {D} d⁻¹  (m = rn_reszoo + D)")
    print()
    print(f"{'Zoo':<6}  {'g (d⁻¹)':>9}  {'K (mol C/L)':>13}  {'GE':>6}  {'m (d⁻¹)':>9}  {'R* (mol C/L)':>14}")
    print("─" * 66)

    for i, name in enumerate(zoo_names):
        g  = rn_grazoo[i]
        K  = rn_grkzoo[i]
        GE = rn_ggezoo[i]
        m  = rn_reszoo[i] + D

        denom = GE * g - m
        if denom <= 0:
            rstar_str = "∞  (GE·g ≤ m)"
        else:
            rstar_str = f"{m * K / denom:.4e}"

        print(f"{name:<6}  {g:>9.4f}  {K:>13.2e}  {GE:>6.3f}  {m:>9.4f}  {rstar_str:>14}")

    print()


if __name__ == '__main__':
    main()
