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

            # Remove any singleton dimensions
            obs_data = obs_data.squeeze()

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

    # For PFTs, we only need the integrated variables
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
