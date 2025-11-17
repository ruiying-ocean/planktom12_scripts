#!/usr/bin/env python3
"""
Python replacement for Ferret map generation scripts.
Creates publication-quality oceanographic maps from NEMO/PlankTom output.

Based on plotting style from ~/tompy/code/OBio_state.ipynb and warming_map.ipynb

Usage:
    python make_maps.py <run_name> <year_start> <year_end> [--output-dir OUTPUT_DIR]
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


def get_longitude_transect(data, nav_lon, target_lon, lat_values):
    """
    Extract data along a specific longitude transect.

    Args:
        data: xarray.DataArray to extract from
        nav_lon: 2D longitude array
        target_lon: Target longitude in degrees (-180 to 180, negative for W, positive for E)
        lat_values: Latitude values for the y dimension

    Returns:
        Data along the longitude transect
    """
    # Find the x-index closest to the target longitude for each y
    lon_diff = np.abs(nav_lon - target_lon)
    x_indices = lon_diff.argmin(dim='x')

    # Extract data along this transect
    transect_data = data.isel(x=x_indices)

    # Assign latitude coordinates
    transect_data['y'] = lat_values

    # Sort by latitude
    transect_data = transect_data.sortby('y')

    return transect_data


def get_central_latitude(nav_lat):
    """Extract latitude from the central meridian of a 2D nav_lat array"""
    mid_x = nav_lat.shape[1] // 2
    central_lat = nav_lat[:, mid_x]
    return central_lat


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

        # Plot model
        im = plotter.plot_variable(
            ax=ax_model,
            data=model_data,
            cmap=meta['cmap'],
            vmin=0,
            vmax=vmax,
            add_colorbar=False
        )

        ax_model.set_title(f"{meta['long_name']} - Model", fontsize=10)

        # Observational data (bottom row)
        ax_obs = axs[1, i]

        if obs_data is not None:
            # Plot observations
            plotter.plot_variable(
                ax=ax_obs,
                data=obs_data,
                cmap=meta['cmap'],
                vmin=0,
                vmax=vmax,
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
    parser.add_argument('--obs-dir',
                       default='/gpfs/home/vhf24tbu/Observations',
                       help='Directory containing observational data files')
    parser.add_argument('--no-nutrient-comparison', action='store_true',
                       help='Disable model vs observations nutrient comparison plots (generate model-only)')

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

    # Load datasets with volume integration using chunked/lazy loading
    # This is memory efficient - data is only loaded when needed
    print("Loading ptrc_T dataset (lazy loading with dask)...")
    ptrc_ds = plotter.load_data(
        str(ptrc_file),
        volume=plotter.volume,
        chunks={'time_counter': 1}  # Load one time step at a time
    )

    print("Loading diad_T dataset (lazy loading with dask)...")
    diad_ds = plotter.load_data(
        str(diad_file),
        volume=plotter.volume,
        chunks={'time_counter': 1}  # Load one time step at a time
    )

    # Compute only the variables we need for plotting (triggers dask computation)
    # This processes data in chunks and releases memory as it goes
    print("Computing time averages (this may take a moment)...")

    # For PFT maps, we only need the integrated variables
    # Note: Vertical integration should already be done by _vint() in ocean_maps.py
    pft_vars = [f'_{pft}INT' for pft in PHYTOS + ZOOS]
    for var in pft_vars:
        if var in ptrc_ds:
            # Compute time average immediately to reduce memory
            # Squeeze to remove any singleton dimensions
            if 'time_counter' in ptrc_ds[var].dims:
                ptrc_ds[var] = ptrc_ds[var].mean(dim='time_counter').squeeze().compute()
            else:
                ptrc_ds[var] = ptrc_ds[var].squeeze().compute()

    # For PFT transects, we also need the raw concentration variables (not integrated)
    # These are stored without the underscore prefix in the raw file
    pft_concentration_vars = PHYTOS + ZOOS
    for var in pft_concentration_vars:
        if var in ptrc_ds:
            # Time average the concentration data for transects
            if 'time_counter' in ptrc_ds[var].dims:
                ptrc_ds[var] = ptrc_ds[var].mean(dim='time_counter').squeeze().compute()
            else:
                ptrc_ds[var] = ptrc_ds[var].squeeze().compute()

    # For diagnostics, we only need these variables
    diag_vars = ['_TChl', '_EXP', '_PPINT']
    for var in diag_vars:
        if var in diad_ds:
            diad_ds[var] = diad_ds[var].mean(dim='time_counter').squeeze().compute()

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
        cmap='turbo'
    )

    # 3. Zooplankton PFTs
    print("3. Zooplankton functional types...")
    plot_pft_maps(
        plotter=plotter,
        ptrc_ds=ptrc_ds,
        pft_list=ZOOS,
        pft_type='zoo',
        output_path=output_dir / f"{args.run_name}_{args.year_start}_zoos.png",
        cmap='turbo'
    )

    # 4. Nutrient maps
    if not args.no_nutrient_comparison:
        print("4. Nutrients (model vs observations)...")

        # Load observational datasets
        obs_dir = Path(args.obs_dir)
        obs_datasets = {}

        # Try to load WOA data for NO3, PO4, Si
        woa_file = obs_dir / 'woa_orca_bil.nc'
        if woa_file.exists():
            print(f"  Loading WOA data from {woa_file}")
            woa_ds = xr.open_dataset(woa_file, decode_times=False)
            print(f"  WOA variables: {list(woa_ds.data_vars)}")
            # Map WOA variables to our naming convention
            if 'no3' in woa_ds:
                obs_datasets['_NO3'] = woa_ds['no3']
            if 'po4' in woa_ds:
                obs_datasets['_PO4'] = woa_ds['po4']
            if 'si' in woa_ds:
                obs_datasets['_Si'] = woa_ds['si']
        else:
            print(f"  Warning: WOA file not found at {woa_file}")

        # Try to load Fe data from Huang2022
        fe_file = obs_dir / 'Huang2022_orca.nc'
        if fe_file.exists():
            print(f"  Loading Fe data from {fe_file}")
            fe_ds = xr.open_dataset(fe_file, decode_times=False)
            print(f"  Fe variables: {list(fe_ds.data_vars)}")
            if 'fe' in fe_ds:
                obs_datasets['_Fer'] = fe_ds['fe']
        else:
            print(f"  Warning: Fe file not found at {fe_file}")

        # Generate comparison plot
        plot_nutrient_comparison(
            plotter=plotter,
            ptrc_ds=ptrc_ds,
            obs_datasets=obs_datasets,
            output_path=output_dir / f"{args.run_name}_{args.year_start}_nutrients.png",
            nutrients=['_NO3', '_PO4', '_Si', '_Fer']
        )

        # Generate basin transects
        print("  Generating Atlantic and Pacific transects...")
        # Get nav_lon/nav_lat from the model output file (ptrc_ds should have it)
        ptrc_file_full = xr.open_dataset(str(ptrc_file), decode_times=False)

        if 'nav_lon' in ptrc_file_full.coords and 'nav_lat' in ptrc_file_full.coords:
            plot_basin_transects(
                plotter=plotter,
                ptrc_ds=ptrc_ds,
                obs_datasets=obs_datasets,
                nav_lon=ptrc_file_full['nav_lon'],
                nav_lat=ptrc_file_full['nav_lat'],
                output_dir=output_dir,
                run_name=args.run_name,
                year=args.year_start,
                nutrients=['_NO3', '_PO4', '_Si', '_Fer']
            )

            # Generate PFT transects
            print("  Generating PFT transects for Atlantic and Pacific...")
            plot_pft_transects(
                plotter=plotter,
                ptrc_ds=ptrc_ds,
                nav_lon=ptrc_file_full['nav_lon'],
                nav_lat=ptrc_file_full['nav_lat'],
                output_dir=output_dir,
                run_name=args.run_name,
                year=args.year_start,
                pfts=PHYTOS + ZOOS
            )

            ptrc_file_full.close()
        else:
            print("  Warning: nav_lon/nav_lat not found in model file, skipping transects")
            ptrc_file_full.close()
    else:
        print("4. Nutrients (model only)...")

        # Create a simple model-only nutrient plot
        nutrients = ['_NO3', '_PO4', '_Si', '_Fer']
        fig, axs = plotter.create_subplot_grid(
            nrows=2, ncols=2,
            projection=ccrs.PlateCarree(),
            figsize=(10, 6)
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

            ax.set_title(meta['long_name'], fontsize=11)

        # Save
        output_path = output_dir / f"{args.run_name}_{args.year_start}_nutrients.png"
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {output_path}")

    print("\n=== All maps generated successfully ===")
    print(f"Output directory: {output_dir}")


if __name__ == '__main__':
    main()
