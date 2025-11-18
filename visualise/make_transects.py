#!/usr/bin/env python3
"""
Create vertical transect plots for oceanographic data.
Generates Atlantic and Pacific transects for nutrients and plankton functional types.

Usage:
    python make_transects.py <run_name> <year> [--model-dir MODEL_DIR] [--output-dir OUTPUT_DIR]
"""

import argparse
import sys
from pathlib import Path
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

# Import our plotting utilities
from map_utils import (
    OceanMapPlotter,
    PHYTOS, ZOOS, PHYTO_NAMES, ZOO_NAMES,
    get_variable_metadata
)

# Import preprocessing utilities
from preprocess_data import (
    load_and_preprocess_ptrc,
    load_observations,
    get_nav_coordinates
)

# Import shared transect utilities
from transect_utils import get_longitude_transect, get_central_latitude


def plot_basin_transects(
    plotter: OceanMapPlotter,
    ptrc_ds: xr.Dataset,
    obs_datasets: dict,
    nav_lon: xr.DataArray,
    nav_lat: xr.DataArray,
    output_dir: Path,
    run_name: str,
    year: str,
    nutrients: list = ['_NO3', '_PO4', '_Si', '_Fer']
):
    """
    Create Atlantic and Pacific nutrient transect plots with observations.

    Args:
        plotter: OceanMapPlotter instance
        ptrc_ds: Model dataset with tracer variables
        obs_datasets: Dict mapping nutrient names to observational datasets
        nav_lon: 2D longitude array
        nav_lat: 2D latitude array
        output_dir: Output directory for plots
        run_name: Model run name
        year: Year string
        nutrients: List of nutrients to plot
    """
    # Get latitude values for transect
    lat_values = get_central_latitude(nav_lat)

    # Define transect longitudes
    atlantic_lon = -35.0  # 35W
    pacific_lon = -170.0  # 170W

    transects = [
        ('Atlantic', atlantic_lon, '35W'),
        ('Pacific', pacific_lon, '170W')
    ]

    for basin_name, target_lon, lon_label in transects:
        # Create 2x4 grid: model top, obs bottom
        fig = plt.figure(figsize=(16, 8))
        gs = fig.add_gridspec(3, 4, height_ratios=[1, 1, 0.1], hspace=0.3, wspace=0.3)

        # Store mappables and vmin/vmax for colorbars
        mappables = []
        vranges = []

        for i, nut in enumerate(nutrients):
            ax_model = fig.add_subplot(gs[0, i])
            ax_obs = fig.add_subplot(gs[1, i])

            meta = get_variable_metadata(nut)

            # Initialize vmin/vmax
            vmin = 0
            vmax = None
            im_model = None
            im_obs = None

            # Model transect
            if nut in ptrc_ds:
                model_data = ptrc_ds[nut]

                # Time average if needed
                if 'time_counter' in model_data.dims:
                    model_data = model_data.mean(dim='time_counter')

                # Remove bottom level if needed
                if 'deptht' in model_data.dims:
                    model_data = model_data.isel(deptht=slice(None, -1))

                model_data = model_data.squeeze()

                # Extract transect
                model_transect = get_longitude_transect(model_data, nav_lon, target_lon, lat_values)

                # Mask land values (0 or very close to 0)
                model_transect_masked = model_transect.where(model_transect > 1e-10)

                # Calculate dynamic vmax from 95th percentile of model data
                vmax = float(np.nanpercentile(model_transect_masked.values, 95))

                # Plot model
                im_model = model_transect_masked.plot(
                    ax=ax_model,
                    cmap=meta['cmap'],
                    vmin=vmin,
                    vmax=vmax,
                    add_colorbar=False
                )

                ax_model.set_title(f"{meta['long_name']}\nModel", fontsize=10)
                ax_model.invert_yaxis()

            # Observation transect
            if nut in obs_datasets and obs_datasets[nut] is not None:
                obs_data = obs_datasets[nut]

                # Get surface and depth if present
                if 'depth' in obs_data.dims:
                    obs_data = obs_data.isel(depth=slice(None, -1))

                obs_data = obs_data.squeeze()

                # Get obs nav_lon (should be in the dataset)
                if 'nav_lon' in obs_data.coords:
                    obs_nav_lon = obs_data.coords['nav_lon']
                else:
                    obs_nav_lon = nav_lon  # Use model nav_lon

                # Extract transect
                obs_transect = get_longitude_transect(obs_data, obs_nav_lon, target_lon, lat_values)

                # If vmax not set from model, calculate from obs
                if vmax is None:
                    vmax = float(np.nanpercentile(obs_transect.values, 95))

                # Plot observations
                im_obs = obs_transect.plot(
                    ax=ax_obs,
                    cmap=meta['cmap'],
                    vmin=vmin,
                    vmax=vmax,
                    add_colorbar=False
                )

                ax_obs.set_title(f"Observations", fontsize=10)
                ax_obs.invert_yaxis()
            else:
                ax_obs.text(0.5, 0.5, 'No observations',
                           ha='center', va='center', transform=ax_obs.transAxes)

            # Store mappable and vrange for colorbar
            mappables.append(im_model if im_model is not None else im_obs)
            vranges.append((vmin, vmax if vmax is not None else meta['vmax']))

            # Set labels only on edge subplots
            if i == 0:
                ax_model.set_ylabel('Depth (m)', fontsize=10)
                ax_obs.set_ylabel('Depth (m)', fontsize=10)
            else:
                ax_model.set_ylabel('')
                ax_obs.set_ylabel('')

            ax_obs.set_xlabel('Latitude (°N)', fontsize=10)
            ax_model.set_xlabel('')

        # Add colorbars at bottom
        for i, nut in enumerate(nutrients):
            if mappables[i] is not None:
                cax = fig.add_subplot(gs[2, i])
                meta = get_variable_metadata(nut)
                cbar = fig.colorbar(mappables[i], cax=cax, orientation='horizontal')
                cbar.set_label(meta['units'], fontsize=9)
                cbar.ax.tick_params(labelsize=8)

        # Overall title
        fig.suptitle(f'{basin_name} Transect ({lon_label})', fontsize=12)

        # Save
        output_path = output_dir / f"{run_name}_{year}_transect_{basin_name.lower()}.png"
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {output_path}")


def plot_pft_transects(
    plotter: OceanMapPlotter,
    ptrc_ds: xr.Dataset,
    nav_lon: xr.DataArray,
    nav_lat: xr.DataArray,
    output_dir: Path,
    run_name: str,
    year: str,
    pfts: list = None,
    max_depth: float = 500.0
):
    """
    Create Atlantic and Pacific PFT transect plots (model only).

    Args:
        plotter: OceanMapPlotter instance
        ptrc_ds: Model dataset with tracer variables
        nav_lon: 2D longitude array
        nav_lat: 2D latitude array
        output_dir: Output directory for plots
        run_name: Model run name
        year: Year string
        pfts: List of PFTs to plot (default: all 12 PFTs)
        max_depth: Maximum depth to plot in meters (default: 500m)
    """
    if pfts is None:
        pfts = PHYTOS + ZOOS

    # Get latitude values for transect
    lat_values = get_central_latitude(nav_lat)

    # Define transect longitudes
    atlantic_lon = -35.0  # 35W
    pacific_lon = -170.0  # 170W

    transects = [
        ('Atlantic', atlantic_lon, '35W'),
        ('Pacific', pacific_lon, '170W')
    ]

    for basin_name, target_lon, lon_label in transects:
        # Create 4x3 subplots for 12 PFTs
        fig, axs = plt.subplots(4, 3, figsize=(12, 13), sharex=True, sharey=True)
        fig.subplots_adjust(hspace=0.3, wspace=0.3)

        for i, pft in enumerate(pfts):
            # Calculate subplot position (4 rows x 3 columns)
            row = i // 3
            col = i % 3
            ax = axs[row, col]

            # Initialize
            vmin = 0
            vmax = None

            # Model transect
            if pft in ptrc_ds:
                model_data = ptrc_ds[pft]

                # Time average if needed
                if 'time_counter' in model_data.dims:
                    model_data = model_data.mean(dim='time_counter')

                # Remove bottom level if needed
                if 'deptht' in model_data.dims:
                    model_data = model_data.isel(deptht=slice(None, -1))

                model_data = model_data.squeeze()

                # Extract transect
                model_transect = get_longitude_transect(model_data, nav_lon, target_lon, lat_values)

                # Convert from mmol C/m³ to µmol C/L (multiply by 1e6)
                model_transect = model_transect * 1e6

                # Limit depth to max_depth
                if 'deptht' in model_transect.coords:
                    depth_mask = model_transect.coords['deptht'] <= max_depth
                    model_transect = model_transect.where(depth_mask, drop=True)

                # Mask land values (0 or very close to 0)
                model_transect_masked = model_transect.where(model_transect > 1e-10)

                # Calculate dynamic vmax from 95th percentile of model data
                vmax = float(np.nanpercentile(model_transect_masked.values, 95))

                # Plot model
                im = model_transect_masked.plot(
                    ax=ax,
                    cmap='turbo',
                    vmin=vmin,
                    vmax=vmax,
                    add_colorbar=True,
                    cbar_kwargs={'label': 'µmol C L⁻¹', 'shrink': 0.8}
                )

                # Get PFT name for title
                if pft in PHYTO_NAMES:
                    pft_name = PHYTO_NAMES[pft]
                elif pft in ZOO_NAMES:
                    pft_name = ZOO_NAMES[pft]
                else:
                    pft_name = pft

                ax.set_title(f"{pft_name}", fontsize=10)
                ax.invert_yaxis()
                ax.set_ylim(max_depth, 0)

            # Set labels only on edge subplots
            if col == 0:
                ax.set_ylabel('Depth (m)', fontsize=10)
            else:
                ax.set_ylabel('')

            if row == 3:  # Bottom row (4 rows, so row 3 is the last)
                ax.set_xlabel('Latitude (°N)', fontsize=10)
            else:
                ax.set_xlabel('')

        # Save
        output_path = output_dir / f"{run_name}_{year}_transect_{basin_name.lower()}_pfts.png"
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {output_path}")


def main():
    """Main entry point for transect generation."""
    parser = argparse.ArgumentParser(
        description='Generate vertical transect plots from NEMO/PlankTom output'
    )
    parser.add_argument('run_name', help='Model run name (e.g., ORCA2_test)')
    parser.add_argument('year', help='Year to process (YYYY)')
    parser.add_argument('--model-dir', default='~/scratch/ModelRuns',
                       help='Directory containing model runs (default: %(default)s)')
    parser.add_argument('--output-dir', default=None,
                       help='Output directory for transects (default: <model-dir>/monitor/<run_name>)')
    parser.add_argument('--mask-path',
                       default='/gpfs/data/greenocean/software/resources/breakdown/basin_mask.nc',
                       help='Path to basin mask file')
    parser.add_argument('--obs-dir',
                       default='/gpfs/home/vhf24tbu/Observations',
                       help='Directory containing observational data files')
    parser.add_argument('--max-depth', type=float, default=500.0,
                       help='Maximum depth for PFT transects in meters (default: 500)')

    args = parser.parse_args()

    # Setup paths
    model_dir = Path(args.model_dir).expanduser()
    run_dir = model_dir / args.run_name

    # Set default output directory to <model-dir>/monitor/<run_name> if not specified
    if args.output_dir is None:
        output_dir = model_dir / "monitor" / args.run_name
    else:
        output_dir = Path(args.output_dir).expanduser()

    output_dir.mkdir(parents=True, exist_ok=True)

    # Construct file paths
    date_str = f"{args.year}0101_{args.year}1231"
    ptrc_file = run_dir / f"ORCA2_1m_{date_str}_ptrc_T.nc"

    # Check files exist
    if not ptrc_file.exists():
        print(f"Error: {ptrc_file} not found")
        sys.exit(1)

    print(f"Loading data from {run_dir}")
    print(f"Processing year: {args.year}")

    # Initialize plotter
    plotter = OceanMapPlotter(mask_path=args.mask_path)

    # Get navigation coordinates
    try:
        nav_lon, nav_lat = get_nav_coordinates(ptrc_file)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Load and preprocess data
    nutrients = ['_NO3', '_PO4', '_Si', '_Fer']
    ptrc_ds = load_and_preprocess_ptrc(
        ptrc_file=ptrc_file,
        plotter=plotter,
        compute_integrated=False,  # Don't need integrated vars for transects
        compute_concentrations=True  # Need concentration vars for transects
    )

    print("Data processing complete.")

    # Load observational datasets
    obs_dir = Path(args.obs_dir)
    obs_datasets = load_observations(obs_dir, nutrients=nutrients)

    # Generate transects
    print("\n=== Generating Transects ===\n")

    print("1. Nutrient transects (Atlantic and Pacific)...")
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

    print("2. PFT transects (Atlantic and Pacific)...")
    plot_pft_transects(
        plotter=plotter,
        ptrc_ds=ptrc_ds,
        nav_lon=nav_lon,
        nav_lat=nav_lat,
        output_dir=output_dir,
        run_name=args.run_name,
        year=args.year,
        pfts=PHYTOS + ZOOS,
        max_depth=args.max_depth
    )

    print("\n=== All transects generated successfully ===")
    print(f"Output directory: {output_dir}")


if __name__ == '__main__':
    main()
