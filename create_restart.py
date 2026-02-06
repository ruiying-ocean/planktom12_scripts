#!/usr/bin/env python3
"""
Create NEMO restart files from model output.

Supports three restart types:
  - trc:   Tracer restart (restart_trc.nc) from ptrc_T output
  - oce:   Ocean restart (restart.nc) from grid_T/U/V/W output
  - ice:   Ice restart (restart_ice.nc) from icemod output

Usage:
    python create_restart.py trc <ptrc_T.nc> [template] [output]
    python create_restart.py oce <run_dir> [template] [output]
    python create_restart.py ice <icemod.nc> [template] [output]
    python create_restart.py all <run_dir> [template_dir] [output_dir]

Defaults:
    template_dir: /gpfs/data/greenocean/software/resources/ModelResources/RestartFiles/
    output_dir:   ~/Observations/input_data/

Examples:
    # Create tracer restart (uses default template and output dirs)
    python create_restart.py trc ORCA2_1m_*_ptrc_T.nc -t 11

    # Create all three restarts with defaults
    python create_restart.py all ./model_output/
"""

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

# =============================================================================
# Default paths
# =============================================================================

DEFAULT_TEMPLATE_DIR = '/gpfs/data/greenocean/software/resources/ModelResources/RestartFiles/'
DEFAULT_OUTPUT_DIR = Path.home() / 'Observations' / 'input_data'

# =============================================================================
# Variable mappings: output_var -> [(restart_var, is_3d), ...]
# =============================================================================

# Tracer mappings (ptrc_T -> restart_trc)
# Output name -> restart base name (will create TRN* and TRB*)
TRACER_MAPPING = {
    'Alkalini': 'Alkalini', 'O2': 'O2', 'DIC': 'DIC', 'PIIC': 'PIIC',
    'NO3': 'NO3', 'Si': 'Si', 'PO4': 'PO4', 'Fer': 'Fer', 'DOC': 'DOC',
    'CaCO3': 'CaCO3', 'ARA': 'ARA', 'POC': 'POC', 'GOC': 'GOC', 'HOC': 'HOC',
    'BAC': 'BAC', 'PRO': 'PRO', 'PTE': 'PTE', 'MES': 'MES', 'GEL': 'GEL',
    'CRU': 'CRU', 'DIA': 'DIA', 'MIX': 'MIX', 'COC': 'COC', 'PIC': 'PIC',
    'PHA': 'PHA', 'FIX': 'FIX', 'BSi': 'BSi', 'GON': 'GON', 'C11': 'C11',
    'B14B': 'B14B', 'C14B': 'C14B', 'D14B': 'D14B',
}

# Ocean mappings: (output_file_suffix, output_var) -> [(restart_var, ndim), ...]
# ndim: 4 = (t,z,y,x), 3 = (t,y,x)
OCEAN_MAPPING = {
    # grid_T variables
    ('grid_T', 'votemper'): [('tn', 4), ('tb', 4)],
    ('grid_T', 'vosaline'): [('sn', 4), ('sb', 4)],
    ('grid_T', 'zos'):      [('sshn', 3), ('sshb', 3)],
    ('grid_T', 'tos'):      [('sst_m', 3)],
    ('grid_T', 'sos'):      [('sss_m', 3)],
    # grid_U variables
    ('grid_U', 'vozocrtx'): [('un', 4), ('ub', 4)],
    ('grid_U', 'uos'):      [('ssu_m', 3)],
    ('grid_U', 'tauuo'):    [('utau_b', 3)],
    # grid_V variables
    ('grid_V', 'vomecrty'): [('vn', 4), ('vb', 4)],
    ('grid_V', 'vos'):      [('ssv_m', 3)],
    ('grid_V', 'tauvo'):    [('vtau_b', 3)],
    # grid_W variables
    ('grid_W', 'difvho'):   [('avt', 4)],
}

# Ice mappings: output_var -> [(restart_var, ndim), ...]
ICE_MAPPING = {
    'sit':      [('hicif', 3)],
    'snd':      [('hsnif', 3)],
    'ist_ipa':  [('sist', 3)],
    'uice_ipa': [('u_ice', 3)],
    'vice_ipa': [('v_ice', 3)],
}


def safe_output_path(new_file, verbose=True):
    """Check if file exists, backup if needed, return path to use."""
    path = Path(new_file)
    if path.exists():
        # Find a backup name that doesn't exist
        i = 1
        while True:
            backup = path.with_suffix(f'.bak{i}.nc')
            if not backup.exists():
                break
            i += 1
        shutil.move(str(path), str(backup))
        if verbose:
            print(f"  Backed up existing file to: {backup.name}")
    return str(path)


def clean_data(data, fill_threshold=1e19):
    """Replace fill values with 0 and convert to float64."""
    data = np.where(np.abs(data) > fill_threshold, 0.0, data)
    return data.astype(np.float64)


def print_stats(name, data):
    """Print min/max statistics for a variable."""
    valid = data[np.abs(data) > 1e-30]
    if len(valid) > 0:
        print(f"    {name:20s} [min: {valid.min():12.4e}, max: {valid.max():12.4e}]")
    else:
        print(f"    {name:20s} [all zeros/missing]")


def find_output_file(run_dir, pattern):
    """Find output file matching pattern in run directory."""
    run_path = Path(run_dir)
    matches = list(run_path.glob(f"*_{pattern}.nc"))
    if not matches:
        return None
    # Return most recent if multiple matches
    return str(sorted(matches)[-1])


# =============================================================================
# Tracer restart
# =============================================================================

def create_tracer_restart(output_file, template, new_file, time_index=-1, verbose=True):
    """Create tracer restart from ptrc_T output."""
    new_file = safe_output_path(new_file, verbose)

    if verbose:
        print(f"\n{'='*60}")
        print(f"Creating TRACER restart")
        print(f"{'='*60}")
        print(f"  Input:    {output_file}")
        print(f"  Template: {template}")
        print(f"  New file: {new_file}")

    shutil.copy2(template, new_file)

    with Dataset(output_file, 'r') as nc_out:
        n_times = len(nc_out.dimensions.get('time_counter', nc_out.dimensions.get('t', [1])))
        idx = time_index if time_index >= 0 else n_times + time_index
        if verbose:
            print(f"  Timesteps: {n_times}, using index: {idx}")

        with Dataset(new_file, 'r+') as nc_rst:
            restart_vars = list(nc_rst.variables.keys())
            updated = []

            for out_var, base_name in TRACER_MAPPING.items():
                if out_var not in nc_out.variables:
                    continue

                trn_name = f'TRN{base_name}'
                trb_name = f'TRB{base_name}'

                if trn_name not in restart_vars or trb_name not in restart_vars:
                    continue

                data = clean_data(nc_out.variables[out_var][idx, :, :, :])
                nc_rst.variables[trn_name][:] = data
                nc_rst.variables[trb_name][:] = data
                updated.append(out_var)

                if verbose:
                    print_stats(f"{out_var} -> TRN/TRB", data)

    if verbose:
        print(f"\n  Updated {len(updated)} tracer pairs")
        print(f"  Created: {new_file}")

    return new_file


# =============================================================================
# Ocean restart
# =============================================================================

def create_ocean_restart(run_dir, template, new_file, time_index=-1, verbose=True):
    """Create ocean restart from grid_T/U/V/W output."""
    new_file = safe_output_path(new_file, verbose)

    if verbose:
        print(f"\n{'='*60}")
        print(f"Creating OCEAN restart")
        print(f"{'='*60}")
        print(f"  Input:    {run_dir}")
        print(f"  Template: {template}")
        print(f"  New file: {new_file}")

    shutil.copy2(template, new_file)

    # Group mappings by file
    file_vars = {}
    for (file_suffix, out_var), rst_vars in OCEAN_MAPPING.items():
        if file_suffix not in file_vars:
            file_vars[file_suffix] = {}
        file_vars[file_suffix][out_var] = rst_vars

    updated = []

    with Dataset(new_file, 'r+') as nc_rst:
        restart_vars = list(nc_rst.variables.keys())

        for file_suffix, var_mapping in file_vars.items():
            out_file = find_output_file(run_dir, file_suffix)
            if not out_file:
                if verbose:
                    print(f"\n  Warning: No {file_suffix} file found, skipping")
                continue

            if verbose:
                print(f"\n  Reading {file_suffix}: {Path(out_file).name}")

            with Dataset(out_file, 'r') as nc_out:
                n_times = len(nc_out.dimensions.get('time_counter', [1]))
                idx = time_index if time_index >= 0 else n_times + time_index

                for out_var, rst_list in var_mapping.items():
                    if out_var not in nc_out.variables:
                        if verbose:
                            print(f"    {out_var:20s} not in output, skipping")
                        continue

                    # Read data
                    out_data = nc_out.variables[out_var]
                    ndim_out = len(out_data.dimensions)

                    if ndim_out == 4:
                        data = clean_data(out_data[idx, :, :, :])
                    elif ndim_out == 3:
                        data = clean_data(out_data[idx, :, :])
                    else:
                        continue

                    for rst_var, _ in rst_list:
                        if rst_var not in restart_vars:
                            continue

                        # Handle dimension differences
                        rst_shape = nc_rst.variables[rst_var].shape
                        if len(rst_shape) == 4 and ndim_out == 4:
                            # (t,z,y,x) restart
                            nc_rst.variables[rst_var][0, :, :, :] = data
                        elif len(rst_shape) == 3 and ndim_out == 3:
                            # (t,y,x) restart
                            nc_rst.variables[rst_var][0, :, :] = data
                        elif len(rst_shape) == 3 and rst_shape[0] != 1 and ndim_out == 4:
                            # (z,y,x) restart from 4D output
                            nc_rst.variables[rst_var][:] = data
                        elif len(rst_shape) == 2 and ndim_out == 3:
                            # (y,x) restart from 3D output
                            nc_rst.variables[rst_var][:] = data
                        else:
                            if verbose:
                                print(f"    {rst_var:20s} shape mismatch: out={data.shape}, rst={rst_shape}")
                            continue

                        updated.append(rst_var)
                        if verbose:
                            print_stats(f"{out_var} -> {rst_var}", data)

    if verbose:
        print(f"\n  Updated {len(updated)} variables")
        print(f"  Created: {new_file}")

    return new_file


# =============================================================================
# Ice restart
# =============================================================================

def create_ice_restart(output_file, template, new_file, time_index=-1, verbose=True):
    """Create ice restart from icemod output."""
    new_file = safe_output_path(new_file, verbose)

    if verbose:
        print(f"\n{'='*60}")
        print(f"Creating ICE restart")
        print(f"{'='*60}")
        print(f"  Input:    {output_file}")
        print(f"  Template: {template}")
        print(f"  New file: {new_file}")

    shutil.copy2(template, new_file)

    with Dataset(output_file, 'r') as nc_out:
        n_times = len(nc_out.dimensions.get('time_counter', [1]))
        idx = time_index if time_index >= 0 else n_times + time_index
        if verbose:
            print(f"  Timesteps: {n_times}, using index: {idx}")

        with Dataset(new_file, 'r+') as nc_rst:
            restart_vars = list(nc_rst.variables.keys())
            updated = []

            for out_var, rst_list in ICE_MAPPING.items():
                if out_var not in nc_out.variables:
                    if verbose:
                        print(f"    {out_var:20s} not in output, skipping")
                    continue

                data = clean_data(nc_out.variables[out_var][idx, :, :])

                for rst_var, _ in rst_list:
                    if rst_var not in restart_vars:
                        continue

                    rst_shape = nc_rst.variables[rst_var].shape
                    if len(rst_shape) == 3:
                        nc_rst.variables[rst_var][0, :, :] = data
                    else:
                        nc_rst.variables[rst_var][:] = data

                    updated.append(rst_var)
                    if verbose:
                        print_stats(f"{out_var} -> {rst_var}", data)

    if verbose:
        print(f"\n  Updated {len(updated)} variables")
        print(f"  Created: {new_file}")

    return new_file


# =============================================================================
# Create all restarts
# =============================================================================

def create_all_restarts(run_dir, template_dir, output_dir, time_index=-1, verbose=True):
    """Create all three restart files."""
    template_path = Path(template_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Find files
    ptrc_file = find_output_file(run_dir, 'ptrc_T')
    ice_file = find_output_file(run_dir, 'icemod')

    # Find templates (look for restart*trc*, restart*opa* or restart.nc, restart*ice*)
    trc_templates = list(template_path.glob('*restart*trc*.nc'))
    oce_templates = list(template_path.glob('*restart*opa*.nc'))
    if not oce_templates:
        oce_templates = [p for p in template_path.glob('restart*.nc')
                        if 'trc' not in p.name and 'ice' not in p.name]
    ice_templates = list(template_path.glob('*restart*ice*.nc'))

    results = {}

    if ptrc_file and trc_templates:
        results['trc'] = create_tracer_restart(
            ptrc_file, str(trc_templates[0]),
            str(output_path / 'restart_trc.nc'),
            time_index, verbose
        )

    if oce_templates:
        results['oce'] = create_ocean_restart(
            run_dir, str(oce_templates[0]),
            str(output_path / 'restart.nc'),
            time_index, verbose
        )

    if ice_file and ice_templates:
        results['ice'] = create_ice_restart(
            ice_file, str(ice_templates[0]),
            str(output_path / 'restart_ice.nc'),
            time_index, verbose
        )

    if verbose:
        print(f"\n{'='*60}")
        print(f"Summary: Created {len(results)} restart files")
        for rtype, rfile in results.items():
            print(f"  {rtype}: {rfile}")
        print(f"{'='*60}")

    return results


# =============================================================================
# Main
# =============================================================================

def find_template(restart_type):
    """Find default template file for given restart type."""
    template_dir = Path(DEFAULT_TEMPLATE_DIR)
    if restart_type == 'trc':
        matches = list(template_dir.glob('*restart*trc*.nc'))
    elif restart_type == 'oce':
        matches = list(template_dir.glob('*restart*opa*.nc'))
        if not matches:
            matches = [p for p in template_dir.glob('restart*.nc')
                      if 'trc' not in p.name and 'ice' not in p.name]
    elif restart_type == 'ice':
        matches = list(template_dir.glob('*restart*ice*.nc'))
    else:
        return None
    return str(matches[0]) if matches else None


def main():
    parser = argparse.ArgumentParser(
        description='Create NEMO restart files from model output',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    subparsers = parser.add_subparsers(dest='command', help='Restart type')

    # Tracer restart
    p_trc = subparsers.add_parser('trc', help='Create tracer restart from ptrc_T')
    p_trc.add_argument('input', help='Path to ptrc_T output file')
    p_trc.add_argument('template', nargs='?', default=None,
                       help=f'Template file (default: auto-find in {DEFAULT_TEMPLATE_DIR})')
    p_trc.add_argument('new_file', nargs='?', default=None,
                       help=f'Output file (default: {DEFAULT_OUTPUT_DIR}/restart_trc.nc)')
    p_trc.add_argument('-t', '--time-index', type=int, default=-1,
                       help='Timestep index (-1=last, 0=first, 11=Dec)')

    # Ocean restart
    p_oce = subparsers.add_parser('oce', help='Create ocean restart from grid_T/U/V/W')
    p_oce.add_argument('run_dir', help='Directory containing grid_*.nc files')
    p_oce.add_argument('template', nargs='?', default=None,
                       help=f'Template file (default: auto-find in {DEFAULT_TEMPLATE_DIR})')
    p_oce.add_argument('new_file', nargs='?', default=None,
                       help=f'Output file (default: {DEFAULT_OUTPUT_DIR}/restart.nc)')
    p_oce.add_argument('-t', '--time-index', type=int, default=-1,
                       help='Timestep index (-1=last, 0=first, 11=Dec)')

    # Ice restart
    p_ice = subparsers.add_parser('ice', help='Create ice restart from icemod')
    p_ice.add_argument('input', help='Path to icemod output file')
    p_ice.add_argument('template', nargs='?', default=None,
                       help=f'Template file (default: auto-find in {DEFAULT_TEMPLATE_DIR})')
    p_ice.add_argument('new_file', nargs='?', default=None,
                       help=f'Output file (default: {DEFAULT_OUTPUT_DIR}/restart_ice.nc)')
    p_ice.add_argument('-t', '--time-index', type=int, default=-1,
                       help='Timestep index (-1=last, 0=first, 11=Dec)')

    # All restarts
    p_all = subparsers.add_parser('all', help='Create all three restart files')
    p_all.add_argument('run_dir', help='Directory containing output files')
    p_all.add_argument('template_dir', nargs='?', default=DEFAULT_TEMPLATE_DIR,
                       help=f'Template directory (default: {DEFAULT_TEMPLATE_DIR})')
    p_all.add_argument('output_dir', nargs='?', default=str(DEFAULT_OUTPUT_DIR),
                       help=f'Output directory (default: {DEFAULT_OUTPUT_DIR})')
    p_all.add_argument('-t', '--time-index', type=int, default=-1,
                       help='Timestep index (-1=last, 0=first, 11=Dec)')

    parser.add_argument('-q', '--quiet', action='store_true', help='Suppress output')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    verbose = not getattr(args, 'quiet', False)
    time_idx = args.time_index

    if args.command == 'trc':
        template = args.template or find_template('trc')
        new_file = args.new_file or str(DEFAULT_OUTPUT_DIR / 'restart_trc.nc')
        if not template:
            print(f"Error: No tracer template found in {DEFAULT_TEMPLATE_DIR}")
            sys.exit(1)
        create_tracer_restart(args.input, template, new_file, time_idx, verbose)

    elif args.command == 'oce':
        template = args.template or find_template('oce')
        new_file = args.new_file or str(DEFAULT_OUTPUT_DIR / 'restart.nc')
        if not template:
            print(f"Error: No ocean template found in {DEFAULT_TEMPLATE_DIR}")
            sys.exit(1)
        create_ocean_restart(args.run_dir, template, new_file, time_idx, verbose)

    elif args.command == 'ice':
        template = args.template or find_template('ice')
        new_file = args.new_file or str(DEFAULT_OUTPUT_DIR / 'restart_ice.nc')
        if not template:
            print(f"Error: No ice template found in {DEFAULT_TEMPLATE_DIR}")
            sys.exit(1)
        create_ice_restart(args.input, template, new_file, time_idx, verbose)

    elif args.command == 'all':
        create_all_restarts(args.run_dir, args.template_dir, args.output_dir, time_idx, verbose)


if __name__ == '__main__':
    main()
