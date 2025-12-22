#!/usr/bin/env python3
"""
Data preprocessing utilities for NEMO/PlankTom visualization.
Handles loading, time-averaging, and preparing datasets for plotting.
"""

from pathlib import Path
import xarray as xr
from typing import Optional, List, Tuple

from map_utils import OceanMapPlotter, PHYTOS, ZOOS, calculate_3d_aou


def load_grid_t_for_aou(
    grid_t_file: Path
) -> Tuple[xr.DataArray, xr.DataArray]:
    """
    Load temperature and salinity from grid_T file for AOU calculation.

    Args:
        grid_t_file: Path to grid_T NetCDF file

    Returns:
        Tuple of (temperature, salinity) DataArrays (time-averaged)
    """
    print(f"Loading grid_T dataset for AOU from {grid_t_file}...")

    grid_ds = xr.open_dataset(str(grid_t_file), decode_times=False)

    # Extract temperature and salinity
    temp = grid_ds['votemper']
    sal = grid_ds['vosaline']

    # Time average if needed
    if 'time_counter' in temp.dims:
        temp = temp.mean(dim='time_counter').squeeze()
        sal = sal.mean(dim='time_counter').squeeze()

    print("  Loaded votemper and vosaline")
    return temp, sal


def load_and_preprocess_ptrc(
    ptrc_file: Path,
    plotter: OceanMapPlotter,
    variables: Optional[List[str]] = None,
    compute_integrated: bool = True,
    compute_concentrations: bool = True,
    compute_aou: bool = False,
    grid_t_file: Optional[Path] = None
) -> xr.Dataset:
    """
    Load and preprocess tracer (ptrc_T) dataset.

    Args:
        ptrc_file: Path to ptrc_T NetCDF file
        plotter: OceanMapPlotter instance with volume data
        variables: Specific variables to load (None = all)
        compute_integrated: Whether to compute integrated PFT variables (_PICINT, etc.)
        compute_concentrations: Whether to compute PFT concentration variables (PIC, FIX, etc.)
        compute_aou: Whether to compute AOU (requires grid_t_file)
        grid_t_file: Path to grid_T file for temperature/salinity (required if compute_aou=True)

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
    nutrients = ['_NO3', '_PO4', '_Si', '_Fer', '_O2']
    for var in nutrients:
        if var in ptrc_ds and (variables is None or var in variables):
            if 'time_counter' in ptrc_ds[var].dims:
                ptrc_ds[var] = ptrc_ds[var].mean(dim='time_counter').squeeze().compute()
            else:
                ptrc_ds[var] = ptrc_ds[var].squeeze().compute()
            print(f"  Processed {var}")

    # Process carbon chemistry variables
    carbon_vars = ['_ALK', '_DIC']
    for var in carbon_vars:
        if var in ptrc_ds and (variables is None or var in variables):
            if 'time_counter' in ptrc_ds[var].dims:
                ptrc_ds[var] = ptrc_ds[var].mean(dim='time_counter').squeeze().compute()
            else:
                ptrc_ds[var] = ptrc_ds[var].squeeze().compute()
            print(f"  Processed {var}")

    # Compute AOU if requested and grid_T file provided
    if compute_aou and grid_t_file is not None and grid_t_file.exists():
        if 'O2' in ptrc_ds:
            temp, sal = load_grid_t_for_aou(grid_t_file)
            # Get raw O2 (before unit conversion) for AOU calculation
            o2_raw = ptrc_ds['O2']
            if 'time_counter' in o2_raw.dims:
                o2_raw = o2_raw.mean(dim='time_counter').squeeze()
            # Calculate AOU at 300m (depth level 17) using plotter method
            ptrc_ds['_AOU'] = plotter.calculate_aou(o2_raw, temp, sal, depth_index=17).compute()
            print(f"  Processed _AOU")
            # Also calculate 3D AOU for transects if needed
            ptrc_ds['_AOU_3D'] = calculate_3d_aou(o2_raw, temp, sal).compute()
            print(f"  Processed _AOU_3D (for transects), shape: {ptrc_ds['_AOU_3D'].shape}")
        else:
            print("  Warning: O2 not found in ptrc_ds, skipping AOU calculation")
    elif compute_aou:
        print("  Warning: grid_t_file not provided or doesn't exist, skipping AOU calculation")

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

    # Standard diagnostic variables including derived variables
    if variables is None:
        variables = ['_TChl', '_EXP', '_PPINT', '_SPINT', '_RESIDUALINT', '_eratio', '_Teff']

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
    nutrients: List[str] = ['_NO3', '_PO4', '_Si', '_Fer'],
    carbon_chemistry: List[str] = None
) -> dict:
    """
    Load observational datasets for nutrients and carbon chemistry.

    Args:
        obs_dir: Directory containing observational data files
        nutrients: List of nutrients to load observations for
        carbon_chemistry: List of carbon chemistry variables to load (e.g., ['_ALK', '_DIC'])

    Returns:
        Dictionary mapping variable names to observational DataArrays
    """
    obs_datasets = {}

    # Try to load WOA data for NO3, PO4, Si, O2
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
        if 'o2' in woa_ds and '_O2' in nutrients:
            obs_datasets['_O2'] = woa_ds['o2']
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

    # Try to load GLODAP data for carbon chemistry (ALK, DIC)
    if carbon_chemistry:
        glodap_file = obs_dir / 'glodap_orca_bil.nc'
        if glodap_file.exists():
            print(f"Loading GLODAP data from {glodap_file}")
            glodap_ds = xr.open_dataset(glodap_file, decode_times=False)
            print(f"  GLODAP variables: {list(glodap_ds.data_vars)}")

            # Map GLODAP variables to our naming convention
            if 'alk' in glodap_ds and '_ALK' in carbon_chemistry:
                obs_datasets['_ALK'] = glodap_ds['alk']
            if 'dic' in glodap_ds and '_DIC' in carbon_chemistry:
                obs_datasets['_DIC'] = glodap_ds['dic']
        else:
            print(f"Warning: GLODAP file not found at {glodap_file}")

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
