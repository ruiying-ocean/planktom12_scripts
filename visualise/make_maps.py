#!/usr/bin/env python3
"""
Python replacement for Ferret map generation scripts.
Creates publication-quality oceanographic maps from NEMO/PlankTom output.

Based on plotting style from ~/tompy/code/OBio_state.ipynb and warming_map.ipynb

Usage:
    python make_maps.py <run_name> <year> [--basedir BASEDIR] [--output-dir OUTPUT_DIR]
"""

import argparse
import sys
from pathlib import Path
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

# Import our plotting utilities
from map_utils import (
    OceanMapPlotter,
    PHYTOS, ZOOS, PHYTO_NAMES, ZOO_NAMES,
    BIOMASS_RANGES, ECOSYSTEM_VARS, NUTRIENT_VARS,
    get_variable_metadata, convert_units
)

# Import transect plotting functions
from make_transects import plot_basin_transects, plot_pft_transects

# Import preprocessing utilities
from preprocess_data import (
    load_and_preprocess_ptrc,
    load_and_preprocess_diad,
    load_observations,
    get_nav_coordinates
)


def plot_pft_maps(
    plotter: OceanMapPlotter,
    ptrc_ds: xr.Dataset,
    pft_list: list,
    pft_type: str,
    output_path: Path,
    cmap: str = 'turbo'
):
    """
    Create multi-panel map of plankton functional types.

    Matches style from OBio_state.ipynb cells 4-5.

    Args:
        plotter: OceanMapPlotter instance
        ptrc_ds: Dataset with tracer variables
        pft_list: List of PFT names (e.g., ['PIC', 'FIX', ...])
        pft_type: 'phyto' or 'zoo'
        output_path: Where to save the figure
        cmap: Colormap to use
    """
    # Create 2x3 subplot grid
    fig, axs = plotter.create_subplot_grid(
        nrows=2, ncols=3,
        projection=ccrs.PlateCarree(),
        figsize=(10, 5)
    )

    # Plot each PFT
    for i, pft in enumerate(pft_list):
        ax = axs.flat[i]

        # Use integrated variable for plotting (_PICINT, _FIXINT, etc.)
        var_int = f'_{pft}INT'

        if var_int not in ptrc_ds:
            print(f"Warning: {var_int} not found in dataset")
            ax.text(0.5, 0.5, f'{var_int}\nNot Available',
                   ha='center', va='center', transform=ax.transAxes)
            continue

        # Get integrated biomass (already time-averaged and vertically integrated)
        data = ptrc_ds[var_int]

        # If time dimension still exists, average it
        if 'time_counter' in data.dims:
            data = data.mean(dim='time_counter')

        # Remove any singleton dimensions
        data = data.squeeze()

        # Apply land mask
        data = plotter.apply_mask(data)

        # Calculate total global biomass (sum over grid, convert Tg to Pg)
        total_biomass = data.sum() * 1e-3

        # Get observational range if available
        obs_range_str = ''
        if var_int in BIOMASS_RANGES:
            obs_min, obs_max = BIOMASS_RANGES[var_int]
            if not np.isnan(obs_min):
                obs_range_str = f', ({obs_min:.1f}-{obs_max:.1f}) Pg C'

        # Calculate dynamic vmax from 95th percentile
        vmax = float(np.nanpercentile(data.values, 95))

        # Plot integrated biomass per grid cell (Tg C)
        im = plotter.plot_variable(
            ax=ax,
            data=data,
            cmap=cmap,
            vmin=0,
            vmax=vmax,
            add_colorbar=False
        )

        # Title shows PFT name with total biomass and obs range
        title = f'{pft}: {total_biomass.values:.1f}{obs_range_str}'
        ax.set_title(title, fontsize=10)

    # Add shared colorbar
    plotter.add_shared_colorbar(
        fig=fig,
        im=im,
        axs=axs,
        label='Tg C',
        orientation='horizontal',
        pad=0.075,
        fraction=0.05
    )

    # Save figure
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_ecosystem_diagnostics(
    plotter: OceanMapPlotter,
    diad_ds: xr.Dataset,
    output_path: Path,
    sat_chl_path: Path = None,
    variables: list = ['_TChl', '_EXP', '_PPINT']
):
    """
    Create multi-panel map of ecosystem diagnostics with satellite chlorophyll comparison.

    Creates a 2x2 grid with:
    - Top left: Model Chlorophyll
    - Top right: Satellite Chlorophyll
    - Bottom left: Export Production
    - Bottom right: Primary Production

    Args:
        plotter: OceanMapPlotter instance
        diad_ds: Dataset with diagnostic variables
        output_path: Where to save the figure
        sat_chl_path: Path to satellite chlorophyll data (optional)
        variables: List of variables to plot
    """
    # Create 2x2 subplot grid
    fig, axs = plotter.create_subplot_grid(
        nrows=2, ncols=2,
        projection=ccrs.PlateCarree(),
        figsize=(10, 6)
    )

    # Top left: Model Chlorophyll
    ax = axs[0, 0]
    if '_TChl' in diad_ds:
        meta = get_variable_metadata('_TChl')
        data = diad_ds['_TChl']

        if 'time_counter' in data.dims:
            data = data.mean(dim='time_counter')
        if 'deptht' in data.dims:
            data = data.isel(deptht=0)

        data = data.squeeze()
        data = convert_units(data, '_TChl')
        data = plotter.apply_mask(data)

        im = plotter.plot_variable(
            ax=ax, data=data, cmap=meta['cmap'],
            vmin=0, vmax=meta['vmax'], add_colorbar=False
        )

        cbar = fig.colorbar(im, ax=ax, orientation='horizontal', pad=0.05, shrink=0.8)
        cbar.set_label(meta['units'], fontsize=10)
        ax.set_title('Model Chlorophyll', fontsize=11)

    # Top right: Satellite Chlorophyll
    ax = axs[0, 1]
    if sat_chl_path and sat_chl_path.exists():
        try:
            sat_ds = xr.open_dataset(sat_chl_path, decode_times=False)
            meta = get_variable_metadata('_TChl')

            if 'chlor_a' in sat_ds:
                sat_chl = sat_ds['chlor_a']

                if 'time' in sat_chl.dims:
                    sat_chl = sat_chl.mean(dim='time')
                elif 'month' in sat_chl.dims:
                    sat_chl = sat_chl.mean(dim='month')

                sat_chl = sat_chl.squeeze()
                sat_chl = plotter.apply_mask(sat_chl)

                im_sat = plotter.plot_variable(
                    ax=ax, data=sat_chl, cmap=meta['cmap'],
                    vmin=0, vmax=meta['vmax'], add_colorbar=False
                )

                cbar = fig.colorbar(im_sat, ax=ax, orientation='horizontal', pad=0.05, shrink=0.8)
                cbar.set_label(meta['units'], fontsize=10)
                ax.set_title('Satellite Chlorophyll', fontsize=11)
            else:
                ax.text(0.5, 0.5, 'chlor_a not found', ha='center', va='center', transform=ax.transAxes)
        except Exception as e:
            ax.text(0.5, 0.5, f'Error loading\nsatellite data', ha='center', va='center', transform=ax.transAxes)
    else:
        ax.text(0.5, 0.5, 'Satellite data\nnot available', ha='center', va='center', transform=ax.transAxes)

    # Bottom row: EXP and PPINT
    for i, var in enumerate(['_EXP', '_PPINT']):
        ax = axs[1, i]

        if var not in diad_ds:
            print(f"Warning: {var} not found in dataset")
            continue

        meta = get_variable_metadata(var)
        data = diad_ds[var]

        if 'time_counter' in data.dims:
            data = data.mean(dim='time_counter')
        if 'deptht' in data.dims:
            depth_idx = meta.get('depth_index', 0)
            data = data.isel(deptht=depth_idx)

        data = data.squeeze()
        data = convert_units(data, var)
        data = plotter.apply_mask(data)

        im = plotter.plot_variable(
            ax=ax, data=data, cmap=meta['cmap'],
            vmin=0, vmax=meta['vmax'], add_colorbar=False
        )

        cbar = fig.colorbar(im, ax=ax, orientation='horizontal', pad=0.05, shrink=0.8)
        cbar.set_label(meta['units'], fontsize=10)
        ax.set_title(meta['long_name'], fontsize=11)

    # Save figure
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_nutrient_comparison(
    plotter: OceanMapPlotter,
    ptrc_ds: xr.Dataset,
    obs_datasets: dict,
    output_path: Path,
    nutrients: list = ['_NO3', '_PO4', '_Si', '_Fer']
):
    """
    Create model vs observations comparison for nutrients with difference maps.

    Creates a 3-column layout: Model | Observations | Difference
    Each nutrient is a row.

    Args:
        plotter: OceanMapPlotter instance
        ptrc_ds: Model dataset with tracer variables
        obs_datasets: Dict mapping nutrient names to observational datasets
        output_path: Where to save the figure
        nutrients: List of nutrients to plot
    """
    nnuts = len(nutrients)

    # Create NxS subplot grid (nutrients in rows, 3 columns: model, obs, diff)
    fig, axs = plotter.create_subplot_grid(
        nrows=nnuts, ncols=3,
        projection=ccrs.PlateCarree(),
        figsize=(12, 3 * nnuts)
    )

    for i, nut in enumerate(nutrients):
        # Column 0: Model data
        ax_model = axs[i, 0]
        # Column 1: Observations
        ax_obs = axs[i, 1]
        # Column 2: Difference
        ax_diff = axs[i, 2]

        if nut not in ptrc_ds:
            print(f"Warning: {nut} not found in model dataset")
            continue

        # Get metadata
        meta = get_variable_metadata(nut)

        # Get surface model data (already time-averaged from preprocessing if available)
        model_data = ptrc_ds[nut]

        # If time dimension still exists, average it
        if 'time_counter' in model_data.dims:
            model_data = model_data.mean(dim='time_counter')

        if 'deptht' in model_data.dims:
            model_data = model_data.isel(deptht=0)

        # Remove any singleton dimensions
        model_data = model_data.squeeze()

        # Convert units (if not already converted)
        model_data = convert_units(model_data, nut)

        # Apply mask
        model_data = plotter.apply_mask(model_data)

        # Calculate dynamic vmax from 95th percentile of model and obs data
        vmax_model = float(np.nanpercentile(model_data.values, 95))

        # Get obs data first to calculate combined vmax
        obs_data = None
        if nut in obs_datasets and obs_datasets[nut] is not None:
            obs_data = obs_datasets[nut]
            # Get surface level
            if 'depth' in obs_data.dims:
                obs_data = obs_data.isel(depth=0)
            elif 'deptht' in obs_data.dims:
                obs_data = obs_data.isel(deptht=0)
            # Remove any singleton dimensions
            obs_data = obs_data.squeeze()
            # Apply mask
            obs_data = plotter.apply_mask(obs_data)
            vmax_obs = float(np.nanpercentile(obs_data.values, 95))
            vmax = max(vmax_model, vmax_obs)
        else:
            vmax = vmax_model

        # Plot model (Column 0)
        im = plotter.plot_variable(
            ax=ax_model,
            data=model_data,
            cmap=meta['cmap'],
            vmin=0,
            vmax=vmax,
            add_colorbar=False
        )

        ax_model.set_title(f"{meta['long_name']} - Model", fontsize=12)
        cbar = fig.colorbar(im, ax=ax_model, orientation='horizontal', pad=0.05, shrink=0.8)
        cbar.set_label(meta['units'], fontsize=10)

        # Plot observations (Column 1)
        if obs_data is not None:
            plotter.plot_variable(
                ax=ax_obs,
                data=obs_data,
                cmap=meta['cmap'],
                vmin=0,
                vmax=vmax,
                add_colorbar=False
            )
            ax_obs.set_title(f"{meta['long_name']} - Observations", fontsize=12)
            cbar_obs = fig.colorbar(im, ax=ax_obs, orientation='horizontal', pad=0.05, shrink=0.8)
            cbar_obs.set_label(meta['units'], fontsize=10)
        else:
            ax_obs.text(0.5, 0.5, 'No observations',
                       ha='center', va='center', transform=ax_obs.transAxes)

        # Plot difference (Column 2)
        if obs_data is not None:
            # Import difference utilities
            from difference_utils import calculate_difference, get_symmetric_colorbar_limits

            diff = calculate_difference(model_data, obs_data)

            # Ensure nav_lon and nav_lat are preserved in difference
            if 'nav_lon' in model_data.coords and 'nav_lat' in model_data.coords:
                diff = diff.assign_coords({
                    'nav_lon': model_data.coords['nav_lon'],
                    'nav_lat': model_data.coords['nav_lat']
                })

            diff = plotter.apply_mask(diff)

            # Get symmetric colorbar limits
            vmin_diff, vmax_diff = get_symmetric_colorbar_limits(diff)

            im_diff = plotter.plot_variable(
                ax=ax_diff,
                data=diff,
                cmap='RdBu_r',
                vmin=vmin_diff,
                vmax=vmax_diff,
                add_colorbar=False
            )

            ax_diff.set_title(f"Difference (Model - Obs)", fontsize=12)
            cbar_diff = fig.colorbar(im_diff, ax=ax_diff, orientation='horizontal', pad=0.05, shrink=0.8)
            cbar_diff.set_label(f'Δ {meta["units"]}', fontsize=10)
        else:
            ax_diff.text(0.5, 0.5, 'Cannot compute\ndifference',
                        ha='center', va='center', transform=ax_diff.transAxes)

    # Save figure
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_carbon_chemistry(
    plotter: OceanMapPlotter,
    ptrc_ds: xr.Dataset,
    output_path: Path,
    variables: list = ['_ALK', '_DIC']
):
    """
    Create multi-panel map of carbon chemistry variables.

    Args:
        plotter: OceanMapPlotter instance
        ptrc_ds: Dataset with tracer variables
        output_path: Where to save the figure
        variables: List of variables to plot
    """
    # Create 1x2 subplot grid for ALK and DIC
    fig, axs = plotter.create_subplot_grid(
        nrows=1, ncols=2,
        projection=ccrs.PlateCarree(),
        figsize=(12, 4)
    )

    for idx, var_name in enumerate(variables):
        ax = axs.flat[idx]

        if var_name in ptrc_ds:
            meta = get_variable_metadata(var_name)
            data = ptrc_ds[var_name]

            # Time average if needed
            if 'time_counter' in data.dims:
                data = data.mean(dim='time_counter')

            # Get surface level
            if 'deptht' in data.dims:
                data = data.isel(deptht=0)

            data = data.squeeze()
            data = convert_units(data, var_name)
            data = plotter.apply_mask(data)

            # Plot
            vmin = meta.get('vmin', None)
            vmax = meta.get('vmax', None)

            im = plotter.plot_variable(
                ax=ax, data=data, cmap=meta['cmap'],
                vmin=vmin, vmax=vmax, add_colorbar=False
            )

            ax.set_title(meta['long_name'], fontsize=12, fontweight='bold')

            # Add colorbar
            cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.05, shrink=0.8)
            cbar.set_label(meta['units'], fontsize=10)
            cbar.ax.tick_params(labelsize=8)
        else:
            ax.text(0.5, 0.5, f'{var_name}\nnot available',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(var_name, fontsize=12)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_derived_variables(
    plotter: OceanMapPlotter,
    diad_ds: xr.Dataset,
    output_path: Path,
    variables: list = ['_SPINT', '_RECYCLEINT', '_eratio', '_Teff']
):
    """
    Create multi-panel map of derived ecosystem variables.

    Args:
        plotter: OceanMapPlotter instance
        diad_ds: Dataset with diagnostic variables
        output_path: Where to save the figure
        variables: List of variables to plot
    """
    # Create 2x2 subplot grid
    fig, axs = plotter.create_subplot_grid(
        nrows=2, ncols=2,
        projection=ccrs.PlateCarree(),
        figsize=(10, 6)
    )

    # Flatten axes for easier indexing
    axs_flat = axs.flatten()

    for idx, var_name in enumerate(variables):
        if idx >= 4:  # Only plot first 4 variables
            break

        ax = axs_flat[idx]

        if var_name in diad_ds:
            meta = get_variable_metadata(var_name)
            data = diad_ds[var_name]

            # Time average if needed
            if 'time_counter' in data.dims:
                data = data.mean(dim='time_counter')

            # Handle depth dimension if present
            if 'deptht' in data.dims or 'nav_lev' in data.dims:
                depth_dim = 'deptht' if 'deptht' in data.dims else 'nav_lev'
                depth_index = meta.get('depth_index', 0)
                if depth_index is not None:
                    data = data.isel({depth_dim: depth_index})
                else:
                    # If depth_index is None, use surface
                    data = data.isel({depth_dim: 0})

            data = data.squeeze()
            data = convert_units(data, var_name)
            data = plotter.apply_mask(data)

            # Plot
            vmin = meta.get('vmin', 0)
            vmax = meta.get('vmax', None)

            im = plotter.plot_variable(
                ax=ax, data=data, cmap=meta['cmap'],
                vmin=vmin, vmax=vmax, add_colorbar=False
            )

            ax.set_title(meta['long_name'], fontsize=12, fontweight='bold')

            # Add colorbar
            cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.05, shrink=0.8)
            cbar.set_label(meta['units'], fontsize=10)
            cbar.ax.tick_params(labelsize=8)
        else:
            ax.text(0.5, 0.5, f'{var_name}\nnot available',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(var_name, fontsize=12)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_path}")


def main():
    """Main entry point for map generation."""
    parser = argparse.ArgumentParser(
        description='Generate oceanographic maps from NEMO/PlankTom output'
    )
    parser.add_argument('run_name', help='Model run name (e.g., ORCA2_test)')
    parser.add_argument('year', help='Year to process (YYYY)')
    parser.add_argument('--model-run-dir', default='~/scratch/ModelRuns',
                       help='Directory containing model runs (default: %(default)s)')
    parser.add_argument('--output-dir', default=None,
                       help='Output directory for maps (default: <model-run-dir>/monitor/<run_name>)')
    parser.add_argument('--mask-path',
                       default='/gpfs/data/greenocean/software/resources/breakdown/basin_mask.nc',
                       help='Path to basin mask file')
    parser.add_argument('--obs-dir',
                       default='/gpfs/home/vhf24tbu/Observations',
                       help='Directory containing observational data files')
    parser.add_argument('--no-nutrient-comparison', action='store_true',
                       help='Disable model vs observations nutrient comparison plots (generate model-only)')

    args = parser.parse_args()

    # Setup paths
    model_run_dir = Path(args.model_run_dir).expanduser()
    run_dir = model_run_dir / args.run_name

    # Set default output directory to <model-run-dir>/monitor/<run_name> if not specified
    if args.output_dir is None:
        output_dir = model_run_dir / "monitor" / args.run_name
    else:
        output_dir = Path(args.output_dir).expanduser()

    output_dir.mkdir(parents=True, exist_ok=True)

    # Construct file paths
    date_str = f"{args.year}0101_{args.year}1231"
    ptrc_file = run_dir / f"ORCA2_1m_{date_str}_ptrc_T.nc"
    diad_file = run_dir / f"ORCA2_1m_{date_str}_diad_T.nc"

    # Check files exist
    if not ptrc_file.exists():
        print(f"Error: {ptrc_file} not found")
        sys.exit(1)
    if not diad_file.exists():
        print(f"Error: {diad_file} not found")
        sys.exit(1)

    print(f"Loading data from {run_dir}")
    print(f"Processing year: {args.year}")

    # Initialize plotter
    plotter = OceanMapPlotter(mask_path=args.mask_path)

    # Load and preprocess datasets using preprocessing module
    ptrc_ds = load_and_preprocess_ptrc(
        ptrc_file=ptrc_file,
        plotter=plotter,
        compute_integrated=True,  # Need integrated vars for PFT maps
        compute_concentrations=True  # Need concentration vars for transects
    )

    diad_ds = load_and_preprocess_diad(
        diad_file=diad_file,
        plotter=plotter
    )

    print("Data processing complete.")

    # Generate maps
    print("\n=== Generating Maps ===\n")

    # 1. Ecosystem diagnostics (TChl, EXP, PPINT) with satellite chlorophyll comparison
    print("1. Ecosystem diagnostics with satellite chlorophyll...")
    obs_dir = Path(args.obs_dir)
    # Try OC-CCI first, then modis, then merged climatology
    chl_obs_file = obs_dir / 'occi_chla_monthly_climatology.nc'
    if not chl_obs_file.exists():
        chl_obs_file = obs_dir / 'modis_chla_climatology_orca.nc'
    if not chl_obs_file.exists():
        chl_obs_file = obs_dir / 'merged_chla_climatology_orca.nc'

    plot_ecosystem_diagnostics(
        plotter=plotter,
        diad_ds=diad_ds,
        sat_chl_path=chl_obs_file,
        output_path=output_dir / f"{args.run_name}_{args.year}_diagnostics.png"
    )

    # 2. Phytoplankton PFTs
    print("2. Phytoplankton functional types...")
    plot_pft_maps(
        plotter=plotter,
        ptrc_ds=ptrc_ds,
        pft_list=PHYTOS,
        pft_type='phyto',
        output_path=output_dir / f"{args.run_name}_{args.year}_phytos.png",
        cmap='turbo'
    )

    # 3. Zooplankton PFTs
    print("3. Zooplankton functional types...")
    plot_pft_maps(
        plotter=plotter,
        ptrc_ds=ptrc_ds,
        pft_list=ZOOS,
        pft_type='zoo',
        output_path=output_dir / f"{args.run_name}_{args.year}_zoos.png",
        cmap='turbo'
    )

    # 4. Nutrient maps
    if not args.no_nutrient_comparison:
        print("4. Nutrients (model vs observations)...")

        # Load observational datasets using preprocessing module (including O2)
        obs_dir = Path(args.obs_dir)
        nutrients = ['_NO3', '_PO4', '_Si', '_Fer', '_O2']
        obs_datasets = load_observations(obs_dir, nutrients=nutrients)

        # Generate comparison plot
        plot_nutrient_comparison(
            plotter=plotter,
            ptrc_ds=ptrc_ds,
            obs_datasets=obs_datasets,
            output_path=output_dir / f"{args.run_name}_{args.year}_nutrients.png",
            nutrients=nutrients
        )

        # Generate basin transects
        print("  Generating Atlantic and Pacific transects...")
        try:
            nav_lon, nav_lat = get_nav_coordinates(ptrc_file)

            plot_basin_transects(
                plotter=plotter,
                ptrc_ds=ptrc_ds,
                obs_datasets=obs_datasets,
                nav_lon=nav_lon,
                nav_lat=nav_lat,
                output_dir=output_dir,
                run_name=args.run_name,
                year=args.year,
                nutrients=nutrients
            )

            # Generate PFT transects
            print("  Generating PFT transects for Atlantic and Pacific...")
            plot_pft_transects(
                plotter=plotter,
                ptrc_ds=ptrc_ds,
                nav_lon=nav_lon,
                nav_lat=nav_lat,
                output_dir=output_dir,
                run_name=args.run_name,
                year=args.year,
                pfts=PHYTOS + ZOOS
            )
        except ValueError as e:
            print(f"  Warning: {e}, skipping transects")
    else:
        print("4. Nutrients (model only)...")

        # Create a simple model-only nutrient plot (including O2)
        nutrients = ['_NO3', '_PO4', '_Si', '_Fer', '_O2']
        # Use 3 columns, 2 rows for 5 nutrients
        fig, axs = plotter.create_subplot_grid(
            nrows=2, ncols=3,
            projection=ccrs.PlateCarree(),
            figsize=(15, 8)
        )

        for i, nut in enumerate(nutrients):
            ax = axs.flat[i]

            if nut not in ptrc_ds:
                print(f"Warning: {nut} not found in dataset")
                continue

            # Get metadata
            meta = get_variable_metadata(nut)

            # Get surface data
            data = ptrc_ds[nut]

            # Time average if needed
            if 'time_counter' in data.dims:
                data = data.mean(dim='time_counter')

            # Get surface level
            depth_dims = [d for d in data.dims if d in ['deptht', 'z', 'depth', 'depthu', 'depthv']]
            if depth_dims:
                data = data.isel({depth_dims[0]: 0})

            # Squeeze and mask
            data = data.squeeze()
            data = plotter.apply_mask(data)

            # Plot
            im = plotter.plot_variable(
                ax=ax,
                data=data,
                cmap=meta['cmap'],
                vmin=0,
                vmax=meta['vmax'],
                add_colorbar=False
            )

            # Add colorbar
            cbar = fig.colorbar(im, ax=ax, orientation='horizontal', pad=0.05, shrink=0.8)
            cbar.set_label(meta['units'], fontsize=10)

            ax.set_title(meta['long_name'], fontsize=13)

        # Hide unused subplot (6th position)
        if len(nutrients) < len(axs.flat):
            for idx in range(len(nutrients), len(axs.flat)):
                axs.flat[idx].set_visible(False)

        # Save
        output_path = output_dir / f"{args.run_name}_{args.year}_nutrients.png"
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {output_path}")

    # 5. Derived variables (SP, RECYCLE, e-ratio, Teff)
    print("5. Derived ecosystem variables...")
    plot_derived_variables(
        plotter=plotter,
        diad_ds=diad_ds,
        output_path=output_dir / f"{args.run_name}_{args.year}_derived.png"
    )

    # 6. Carbon chemistry (ALK, DIC)
    print("6. Carbon chemistry variables...")
    plot_carbon_chemistry(
        plotter=plotter,
        ptrc_ds=ptrc_ds,
        output_path=output_dir / f"{args.run_name}_{args.year}_carbon_chemistry.png"
    )

    print("\n=== All maps generated successfully ===")
    print(f"Output directory: {output_dir}")


if __name__ == '__main__':
    main()
