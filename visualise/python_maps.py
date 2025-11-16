#!/usr/bin/env python3
"""
Python replacement for Ferret map generation scripts.
Creates publication-quality oceanographic maps from NEMO/PlankTom output.

Based on plotting style from ~/tompy/code/OBio_state.ipynb and warming_map.ipynb

Usage:
    python python_maps.py <run_name> <year_start> <year_end> [--output-dir OUTPUT_DIR]
"""

import argparse
import sys
from pathlib import Path
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

# Import our plotting utilities
from ocean_maps import (
    OceanMapPlotter,
    PHYTOS, ZOOS, PHYTO_NAMES, ZOO_NAMES,
    BIOMASS_RANGES, ECOSYSTEM_VARS, NUTRIENT_VARS,
    get_variable_metadata, convert_units
)


def plot_pft_maps(
    plotter: OceanMapPlotter,
    ptrc_ds: xr.Dataset,
    pft_list: list,
    pft_type: str,
    output_path: Path,
    cmap: str = 'NCV_jet'
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
        figsize=(10, 4)
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

        # Get time-averaged integrated biomass (Tg C per grid cell)
        data = ptrc_ds[var_int].mean(dim='time_counter')

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

        # Plot integrated biomass per grid cell (Tg C)
        im = plotter.plot_variable(
            ax=ax,
            data=data,
            cmap=cmap,
            vmin=0,
            vmax=0.05,  # Tg C per grid cell
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
    variables: list = ['_TChl', '_EXP', '_PPINT']
):
    """
    Create multi-panel map of ecosystem diagnostics.

    Matches style from OBio_state.ipynb cell 3.

    Args:
        plotter: OceanMapPlotter instance
        diad_ds: Dataset with diagnostic variables
        output_path: Where to save the figure
        variables: List of variables to plot
    """
    nvars = len(variables)

    # Create subplot grid
    fig, axs = plotter.create_subplot_grid(
        nrows=1, ncols=nvars,
        projection=ccrs.PlateCarree(),
        figsize=(10, 3)
    )

    # Ensure axs is iterable
    if nvars == 1:
        axs = [axs.flat[0]]
    else:
        axs = axs.flat

    for i, var in enumerate(variables):
        ax = axs[i]

        if var not in diad_ds:
            print(f"Warning: {var} not found in dataset")
            continue

        # Get metadata
        meta = get_variable_metadata(var)

        # Get data
        data = diad_ds[var].mean(dim='time_counter')

        # Apply depth indexing if needed
        if 'deptht' in data.dims:
            depth_idx = meta.get('depth_index', 0)
            data = data.isel(deptht=depth_idx)

        # Convert units
        data = convert_units(data, var)

        # Apply land mask
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

        # Add individual colorbar
        cbar = fig.colorbar(
            im, ax=ax,
            orientation='horizontal',
            pad=0.05,
            shrink=0.8
        )
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
    Create model vs observations comparison for nutrients.

    Matches style from OBio_state.ipynb cell 6.

    Args:
        plotter: OceanMapPlotter instance
        ptrc_ds: Model dataset with tracer variables
        obs_datasets: Dict mapping nutrient names to observational datasets
        output_path: Where to save the figure
        nutrients: List of nutrients to plot
    """
    nnuts = len(nutrients)

    # Create 2xN subplot grid (model top, obs bottom)
    fig, axs = plotter.create_subplot_grid(
        nrows=2, ncols=nnuts,
        projection=ccrs.PlateCarree(),
        figsize=(10, 4)
    )

    for i, nut in enumerate(nutrients):
        # Model data (top row)
        ax_model = axs[0, i]

        if nut not in ptrc_ds:
            print(f"Warning: {nut} not found in model dataset")
            continue

        # Get metadata
        meta = get_variable_metadata(nut)

        # Get surface model data
        model_data = ptrc_ds[nut].mean(dim='time_counter')
        if 'deptht' in model_data.dims:
            model_data = model_data.isel(deptht=0)

        # Convert units
        model_data = convert_units(model_data, nut)

        # Apply mask
        model_data = plotter.apply_mask(model_data)

        # Plot model
        im = plotter.plot_variable(
            ax=ax_model,
            data=model_data,
            cmap=meta['cmap'],
            vmin=0,
            vmax=meta['vmax'],
            add_colorbar=False
        )

        ax_model.set_title(f"{meta['long_name']} - Model", fontsize=10)

        # Observational data (bottom row)
        ax_obs = axs[1, i]

        if nut in obs_datasets and obs_datasets[nut] is not None:
            obs_data = obs_datasets[nut]

            # Get surface level
            if 'depth' in obs_data.dims:
                obs_data = obs_data.isel(depth=0)
            elif 'deptht' in obs_data.dims:
                obs_data = obs_data.isel(deptht=0)

            # Apply mask
            obs_data = plotter.apply_mask(obs_data)

            # Plot observations
            plotter.plot_variable(
                ax=ax_obs,
                data=obs_data,
                cmap=meta['cmap'],
                vmin=0,
                vmax=meta['vmax'],
                add_colorbar=False
            )

            ax_obs.set_title(f"{meta['long_name']} - Observations", fontsize=10)
        else:
            ax_obs.text(0.5, 0.5, 'No observations',
                       ha='center', va='center', transform=ax_obs.transAxes)

        # Add shared colorbar for this column
        cbar = fig.colorbar(im, ax=[ax_model, ax_obs],
                          orientation='horizontal',
                          shrink=0.8)
        cbar.set_label(meta['units'], fontsize=10)

    # Save figure
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
    parser.add_argument('year_start', help='Start year (YYYY)')
    parser.add_argument('year_end', help='End year (YYYY)')
    parser.add_argument('--basedir', default='..',
                       help='Base directory for model output')
    parser.add_argument('--output-dir', default='.',
                       help='Output directory for maps')
    parser.add_argument('--mask-path',
                       default='/gpfs/data/greenocean/software/resources/breakdown/basin_mask.nc',
                       help='Path to basin mask file')

    args = parser.parse_args()

    # Setup paths
    basedir = Path(args.basedir)
    run_dir = basedir / args.run_name
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Construct file paths
    date_str = f"{args.year_start}0101_{args.year_end}1231"
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
    print(f"Date range: {args.year_start}-{args.year_end}")

    # Initialize plotter
    plotter = OceanMapPlotter(mask_path=args.mask_path)

    # Load datasets with volume integration
    print("Loading ptrc_T dataset...")
    ptrc_ds = plotter.load_data(str(ptrc_file), volume=plotter.volume)

    print("Loading diad_T dataset...")
    diad_ds = plotter.load_data(str(diad_file), volume=plotter.volume)

    # Generate maps
    print("\n=== Generating Maps ===\n")

    # 1. Ecosystem diagnostics (TChl, EXP, PPINT)
    print("1. Ecosystem diagnostics...")
    plot_ecosystem_diagnostics(
        plotter=plotter,
        diad_ds=diad_ds,
        output_path=output_dir / f"{args.run_name}_{args.year_start}_diagnostics.png"
    )

    # 2. Phytoplankton PFTs
    print("2. Phytoplankton functional types...")
    plot_pft_maps(
        plotter=plotter,
        ptrc_ds=ptrc_ds,
        pft_list=PHYTOS,
        pft_type='phyto',
        output_path=output_dir / f"{args.run_name}_{args.year_start}_phytos.png",
        cmap='NCV_jet'
    )

    # 3. Zooplankton PFTs
    print("3. Zooplankton functional types...")
    plot_pft_maps(
        plotter=plotter,
        ptrc_ds=ptrc_ds,
        pft_list=ZOOS,
        pft_type='zoo',
        output_path=output_dir / f"{args.run_name}_{args.year_start}_zoos.png",
        cmap='NCV_jet'
    )

    # 4. Nutrient maps (model only - observations require separate loading)
    print("4. Nutrients (model only)...")
    # For now, skip obs comparison - can be added when obs files are available
    # plot_nutrient_comparison would be called here

    print("\n=== All maps generated successfully ===")
    print(f"Output directory: {output_dir}")


if __name__ == '__main__':
    main()
