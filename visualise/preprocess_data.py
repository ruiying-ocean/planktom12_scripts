#!/usr/bin/env python3
"""
Data preprocessing utilities for NEMO/PlankTom visualization.
Handles loading, time-averaging, and preparing datasets for plotting.
"""

from pathlib import Path
import xarray as xr
from typing import Optional, List

from map_utils import OceanMapPlotter, PHYTOS, ZOOS


def load_and_preprocess_ptrc(
    ptrc_file: Path,
    plotter: OceanMapPlotter,
    variables: Optional[List[str]] = None,
    compute_integrated: bool = True,
    compute_concentrations: bool = True
) -> xr.Dataset:
    """
    Load and preprocess tracer (ptrc_T) dataset.

    Args:
        ptrc_file: Path to ptrc_T NetCDF file
        plotter: OceanMapPlotter instance with volume data
        variables: Specific variables to load (None = all)
        compute_integrated: Whether to compute integrated PFT variables (_PICINT, etc.)
        compute_concentrations: Whether to compute PFT concentration variables (PIC, FIX, etc.)

    Returns:
        Preprocessed dataset with time-averaged variables
    """
    print(f"Loading ptrc_T dataset from {ptrc_file}...")

    # Load with lazy loading
    ptrc_ds = plotter.load_data(
        str(ptrc_file),
        volume=plotter.volume,
        chunks={'time_counter': 1}
    )

    print("Computing time averages...")

    # Process integrated PFT variables for maps
    if compute_integrated:
        pft_vars = [f'_{pft}INT' for pft in PHYTOS + ZOOS]
        for var in pft_vars:
            if var in ptrc_ds and (variables is None or var in variables):
                if 'time_counter' in ptrc_ds[var].dims:
                    ptrc_ds[var] = ptrc_ds[var].mean(dim='time_counter').squeeze().compute()
                else:
                    ptrc_ds[var] = ptrc_ds[var].squeeze().compute()
                print(f"  Processed {var}")

    # Process concentration variables for transects
    if compute_concentrations:
        pft_concentration_vars = PHYTOS + ZOOS
        for var in pft_concentration_vars:
            if var in ptrc_ds and (variables is None or var in variables):
                if 'time_counter' in ptrc_ds[var].dims:
                    ptrc_ds[var] = ptrc_ds[var].mean(dim='time_counter').squeeze().compute()
                else:
                    ptrc_ds[var] = ptrc_ds[var].squeeze().compute()
                print(f"  Processed {var}")

    # Process nutrient variables
    nutrients = ['_NO3', '_PO4', '_Si', '_Fer']
    for var in nutrients:
        if var in ptrc_ds and (variables is None or var in variables):
            if 'time_counter' in ptrc_ds[var].dims:
                ptrc_ds[var] = ptrc_ds[var].mean(dim='time_counter').squeeze().compute()
            else:
                ptrc_ds[var] = ptrc_ds[var].squeeze().compute()
            print(f"  Processed {var}")

    return ptrc_ds


def load_and_preprocess_diad(
    diad_file: Path,
    plotter: OceanMapPlotter,
    variables: Optional[List[str]] = None
) -> xr.Dataset:
    """
    Load and preprocess diagnostic (diad_T) dataset.

    Args:
        diad_file: Path to diad_T NetCDF file
        plotter: OceanMapPlotter instance with volume data
        variables: Specific variables to load (None = standard diagnostics)

    Returns:
        Preprocessed dataset with time-averaged variables
    """
    print(f"Loading diad_T dataset from {diad_file}...")

    # Load with lazy loading
    diad_ds = plotter.load_data(
        str(diad_file),
        volume=plotter.volume,
        chunks={'time_counter': 1}
    )

    print("Computing time averages...")

    # Standard diagnostic variables
    if variables is None:
        variables = ['_TChl', '_EXP', '_PPINT']

    for var in variables:
        if var in diad_ds:
            if 'time_counter' in diad_ds[var].dims:
                diad_ds[var] = diad_ds[var].mean(dim='time_counter').squeeze().compute()
            else:
                diad_ds[var] = diad_ds[var].squeeze().compute()
            print(f"  Processed {var}")

    return diad_ds


def load_observations(
    obs_dir: Path,
    nutrients: List[str] = ['_NO3', '_PO4', '_Si', '_Fer']
) -> dict:
    """
    Load observational datasets for nutrients.

    Args:
        obs_dir: Directory containing observational data files
        nutrients: List of nutrients to load observations for

    Returns:
        Dictionary mapping nutrient names to observational DataArrays
    """
    obs_datasets = {}

    # Try to load WOA data for NO3, PO4, Si
    woa_file = obs_dir / 'woa_orca_bil.nc'
    if woa_file.exists():
        print(f"Loading WOA data from {woa_file}")
        woa_ds = xr.open_dataset(woa_file, decode_times=False)
        print(f"  WOA variables: {list(woa_ds.data_vars)}")

        # Map WOA variables to our naming convention
        if 'no3' in woa_ds and '_NO3' in nutrients:
            obs_datasets['_NO3'] = woa_ds['no3']
        if 'po4' in woa_ds and '_PO4' in nutrients:
            obs_datasets['_PO4'] = woa_ds['po4']
        if 'si' in woa_ds and '_Si' in nutrients:
            obs_datasets['_Si'] = woa_ds['si']
    else:
        print(f"Warning: WOA file not found at {woa_file}")

    # Try to load Fe data from Huang2022
    fe_file = obs_dir / 'Huang2022_orca.nc'
    if fe_file.exists() and '_Fer' in nutrients:
        print(f"Loading Fe data from {fe_file}")
        fe_ds = xr.open_dataset(fe_file, decode_times=False)
        print(f"  Fe variables: {list(fe_ds.data_vars)}")
        if 'fe' in fe_ds:
            obs_datasets['_Fer'] = fe_ds['fe']
    else:
        print(f"Warning: Fe file not found at {fe_file}")

    return obs_datasets


def get_nav_coordinates(ptrc_file: Path) -> tuple:
    """
    Extract navigation coordinates (nav_lon, nav_lat) from a NetCDF file.

    Args:
        ptrc_file: Path to NetCDF file containing nav_lon and nav_lat

    Returns:
        Tuple of (nav_lon, nav_lat) DataArrays

    Raises:
        ValueError: If nav_lon or nav_lat not found in file
    """
    print(f"Loading navigation coordinates from {ptrc_file}...")
    ds = xr.open_dataset(str(ptrc_file), decode_times=False)

    if 'nav_lon' not in ds.coords or 'nav_lat' not in ds.coords:
        ds.close()
        raise ValueError(f"nav_lon/nav_lat not found in {ptrc_file}")

    nav_lon = ds['nav_lon']
    nav_lat = ds['nav_lat']

    # Keep file open or extract values - let's extract to avoid file handle issues
    nav_lon = nav_lon.copy()
    nav_lat = nav_lat.copy()
    ds.close()

    return nav_lon, nav_lat
