#!/usr/bin/env python3
"""
Generate difference/anomaly maps for 2-model comparisons.
Creates Model1 - Model2 difference maps for spatial diagnostics.

Usage:
    python make_difference_maps.py <model1_name> <model2_name> <year> <model1_model_dir> <model2_model_dir> <output_dir>
"""

import argparse
import sys
from pathlib import Path
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

# Import map utilities from parent directory
sys.path.insert(0, str(Path(__file__).parent.parent))
from map_utils import OceanMapPlotter, get_variable_metadata
from difference_utils import (
    calculate_difference,
    calculate_surface_difference,
    calculate_rmse,
    calculate_bias,
    plot_difference_map,
    get_symmetric_colorbar_limits
)


def load_annual_mean(model_dir, model_id, year, var_name, file_type='ptrc_T', plotter=None):
    """
    Load annual mean for a specific variable using OceanMapPlotter preprocessing.

    Args:
        model_dir: Base directory for model output
        model_id: Model run name
        year: Year to load
        var_name: Variable name (can be preprocessed like _NO3, _PO4, _Si)
        file_type: File type ('ptrc_T' or 'diad_T')
        plotter: OceanMapPlotter instance for preprocessing (required)

    Returns:
        xarray.DataArray with preprocessed annual mean data
    """
    if plotter is None:
        raise ValueError("plotter argument is required for preprocessing")

    nc_file = Path(model_dir) / model_id / f"{model_id}_{year}0101_{year}1231_{file_type}.nc"

    if not nc_file.exists():
        print(f"Warning: File not found: {nc_file}")
        return None

    try:
        # Use plotter.load_data() to get preprocessed variables with unit conversions
        # Pass volume to create integrated variables and derived variables
        ds = plotter.load_data(str(nc_file), volume=plotter.volume)

        if var_name not in ds:
            print(f"Warning: Variable {var_name} not found after preprocessing")
            return None

        # Calculate annual mean
        data = ds[var_name].mean(dim='time_counter')
        return data
    except Exception as e:
        print(f"Error loading {var_name} from {nc_file}: {e}")
        return None


def calculate_derived_variables(model_dir, model_id, year, plotter):
    """
    Load derived ecosystem variables for a model.

    These variables are now automatically calculated by map_utils.py preprocessing:
    - _SPINT: Secondary Production (integrated)
    - _RECYCLEINT: Recycled Production (integrated)
    - _eratio: Export Ratio
    - _Teff: Transfer Efficiency

    Args:
        model_dir: Base directory for model output
        model_id: Model run name
        year: Year to load
        plotter: OceanMapPlotter instance for preprocessing

    Returns:
        Dictionary of derived variables as 2D surface maps
    """
    derived = {}

    # Load preprocessed derived variables from diad file
    # These are automatically created by plotter.load_data() via map_utils.py
    derived_vars = ['_SPINT', '_RECYCLEINT', '_eratio', '_Teff']

    for var_name in derived_vars:
        data = load_annual_mean(model_dir, model_id, year, var_name, file_type='diad_T', plotter=plotter)
        if data is not None:
            # Handle depth dimension if present (should be surface for these vars)
            if 'deptht' in data.dims or 'nav_lev' in data.dims:
                depth_dim = 'deptht' if 'deptht' in data.dims else 'nav_lev'
                meta = get_variable_metadata(var_name)
                depth_index = meta.get('depth_index', 0)
                if depth_index is None:
                    depth_index = 0
                data = data.isel({depth_dim: depth_index})

            # Remove underscore prefix for compatibility with existing code
            clean_name = var_name.replace('_SPINT', 'SP').replace('_RECYCLEINT', 'recycle').replace('_', '')
            derived[clean_name] = data

    return derived


def calculate_surface_difference(data1, data2):
    """Calculate surface (top level) difference between two datasets."""
    if data1 is None or data2 is None:
        return None

    # Take surface level if 3D
    if 'deptht' in data1.dims or 'nav_lev' in data1.dims:
        depth_dim = 'deptht' if 'deptht' in data1.dims else 'nav_lev'
        surf1 = data1.isel({depth_dim: 0})
        surf2 = data2.isel({depth_dim: 0})
    else:
        surf1 = data1
        surf2 = data2

    # Calculate difference
    diff = surf1 - surf2
    return diff


def plot_difference_panel(plotter, differences, var_names, titles, output_path):
    """Plot a panel of difference maps."""
    n_vars = len(var_names)
    ncols = 2
    nrows = (n_vars + 1) // 2

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(12, 4 * nrows),
        subplot_kw={'projection': plotter.projection}
    )
    axes = axes.flatten() if n_vars > 1 else [axes]

    for idx, (var, title) in enumerate(zip(var_names, titles)):
        if var not in differences or differences[var] is None:
            axes[idx].text(0.5, 0.5, f'{var}\nData not available',
                          ha='center', va='center', transform=axes[idx].transAxes)
            axes[idx].set_title(title)
            continue

        diff_data = differences[var]

        # Plot difference with diverging colormap
        plotter.add_map_features(axes[idx])
        im = axes[idx].pcolormesh(
            plotter.nav_lon, plotter.nav_lat, diff_data,
            transform=plotter.data_crs,
            cmap='RdBu_r',  # Red for positive, Blue for negative
            vmin=-np.abs(diff_data).max(), vmax=np.abs(diff_data).max(),
            shading='auto'
        )
        plt.colorbar(im, ax=axes[idx], orientation='horizontal', pad=0.05, shrink=0.8)
        axes[idx].set_title(title)

    # Hide unused subplots
    for idx in range(n_vars, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Created difference map: {output_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Generate model difference maps')
    parser.add_argument('model1', help='First model name')
    parser.add_argument('model2', help='Second model name')
    parser.add_argument('year', help='Year to compare')
    parser.add_argument('model_dir1', help='Base directory for model 1')
    parser.add_argument('model_dir2', help='Base directory for model 2')
    parser.add_argument('output_dir', help='Output directory')

    args = parser.parse_args()
    output_dir = Path(args.output_dir)

    # Initialize plotter with mask for preprocessing
    plotter = OceanMapPlotter()

    # Generate diagnostics difference map (cflx, tchl, ppint, exp)
    print(f"Generating diagnostics difference map ({args.model1} - {args.model2})...")
    diag_vars = {
        'Cflx': ('Cflx', 'Air-Sea Carbon Flux Difference [mol C/m²/s]'),
        'TChl': ('TChl', 'Total Chlorophyll Difference [mg Chl/m³]'),
    }

    differences = {}
    for var, (var_name, title) in diag_vars.items():
        data1 = load_annual_mean(args.model_dir1, args.model1, args.year, var_name, plotter=plotter)
        data2 = load_annual_mean(args.model_dir2, args.model2, args.year, var_name, plotter=plotter)
        differences[var] = calculate_surface_difference(data1, data2)

    if any(diff is not None for diff in differences.values()):
        plot_difference_panel(
            plotter,
            differences,
            list(diag_vars.keys()),
            [title for _, title in diag_vars.values()],
            output_dir / f"difference_{args.year}_diagnostics.png"
        )

    # Generate derived variables difference map
    print(f"Generating derived variables difference map ({args.model1} - {args.model2})...")
    derived1 = calculate_derived_variables(args.model_dir1, args.model1, args.year, plotter)
    derived2 = calculate_derived_variables(args.model_dir2, args.model2, args.year, plotter)

    derived_vars = {
        'SP': 'Secondary Production Difference [gC/m³/yr]',
        'recycle': 'Recycled Production Difference [gC/m³/yr]',
        'eratio': 'Export Ratio Difference (e-ratio)',
        'Teff': 'Transfer Efficiency Difference',
    }

    derived_differences = {}
    for var, title in derived_vars.items():
        if var in derived1 and var in derived2:
            derived_differences[var] = calculate_surface_difference(derived1[var], derived2[var])
        else:
            derived_differences[var] = None

    if any(diff is not None for diff in derived_differences.values()):
        plot_difference_panel(
            plotter,
            derived_differences,
            list(derived_vars.keys()),
            list(derived_vars.values()),
            output_dir / f"difference_{args.year}_derived.png"
        )

    print(f"Difference maps complete for {args.model1} - {args.model2}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
