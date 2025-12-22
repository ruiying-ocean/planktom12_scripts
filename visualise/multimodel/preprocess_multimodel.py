#!/usr/bin/env python3
"""
Data preprocessing utilities for multi-model NEMO/PlankTom comparisons.
Handles loading and preprocessing data from multiple model runs.
"""

import sys
from pathlib import Path
import xarray as xr
from typing import List, Dict, Optional

# Import from parent visualise directory
sys.path.insert(0, str(Path(__file__).parent.parent))
from map_utils import OceanMapPlotter


def load_multimodel_data(
    models: List[Dict],
    variable: str,
    file_type: str = 'ptrc_T',
    plotter: Optional[OceanMapPlotter] = None,
    use_preprocessing: bool = True
) -> Dict[str, xr.DataArray]:
    """
    Load a single variable from multiple model runs.

    Args:
        models: List of dicts with 'name', 'year', 'model_dir' keys
        variable: Variable name to load (e.g., '_NO3', '_PO4', 'PIC')
        file_type: NetCDF file type ('ptrc_T' or 'diad_T')
        plotter: OceanMapPlotter instance (required if use_preprocessing=True)
        use_preprocessing: Whether to use plotter.load_data() for preprocessing

    Returns:
        Dict mapping model names to time-averaged DataArrays
    """
    data = {}

    for model in models:
        model_name = model['name']
        model_dir = model['model_dir']
        year = model['year']

        run_dir = Path(model_dir) / model_name
        nc_file = run_dir / f"ORCA2_1m_{year}0101_{year}1231_{file_type}.nc"

        if not nc_file.exists():
            print(f"Warning: File not found: {nc_file}")
            data[model_name] = None
            continue

        try:
            if use_preprocessing and plotter:
                # Use plotter for preprocessing (unit conversions, integrated variables)
                ds = plotter.load_data(str(nc_file), volume=plotter.volume)
            else:
                # Direct load without preprocessing
                ds = xr.open_dataset(str(nc_file), decode_times=False)

            if variable not in ds:
                print(f"Warning: Variable {variable} not found in {model_name}")
                data[model_name] = None
                continue

            # Time average
            var_data = ds[variable].mean(dim='time_counter')

            # Store as copy to avoid file handle issues
            data[model_name] = var_data.copy()
            ds.close()

        except Exception as e:
            print(f"Error loading {variable} from {model_name}: {e}")
            data[model_name] = None

    return data


def load_transect_variable(
    models: List[Dict],
    variable: str,
    plotter: Optional[OceanMapPlotter] = None
) -> Dict[str, xr.DataArray]:
    """
    Load 3D variable data for transect plotting from multiple models.

    Args:
        models: List of dicts with 'name', 'year', 'model_dir' keys
        variable: Variable name (e.g., 'PIC', '_NO3')
        plotter: OceanMapPlotter instance

    Returns:
        Dict mapping model names to depth-resolved DataArrays
    """
    data = {}

    # Variable mapping for derived variables
    variable_map = {
        '_NO3': ('NO3', 1e6),
        '_PO4': ('PO4', 1e6 / 122),
        '_Si': ('Si', 1e6),
        '_Fer': ('Fer', 1e9),
        '_O2': ('O2', 1e6),
    }

    # Determine base variable and conversion
    if variable in variable_map:
        base_var, conversion = variable_map[variable]
    else:
        base_var = variable
        conversion = 1.0

    for model in models:
        model_name = model['name']
        model_dir = model['model_dir']
        year = model['year']

        run_dir = Path(model_dir) / model_name
        ptrc_file = run_dir / f"ORCA2_1m_{year}0101_{year}1231_ptrc_T.nc"

        if not ptrc_file.exists():
            print(f"Warning: File not found: {ptrc_file}")
            data[model_name] = None
            continue

        try:
            ds = xr.open_dataset(str(ptrc_file), decode_times=False)

            if base_var not in ds:
                print(f"Warning: Variable {base_var} not found in {model_name}")
                data[model_name] = None
                ds.close()
                continue

            # Time average
            var_data = ds[base_var].mean(dim='time_counter')

            # Apply unit conversion
            var_data = var_data * conversion

            # Remove bottom level if needed
            if 'deptht' in var_data.dims:
                var_data = var_data.isel(deptht=slice(None, -1))

            var_data = var_data.squeeze()

            # Store as copy
            data[model_name] = var_data.copy()
            ds.close()

        except Exception as e:
            print(f"Error loading {variable} from {model_name}: {e}")
            data[model_name] = None

    return data


def get_nav_coordinates(models: List[Dict]) -> tuple:
    """
    Get navigation coordinates from first available model.

    Args:
        models: List of dicts with 'name', 'year', 'model_dir' keys

    Returns:
        Tuple of (nav_lon, nav_lat) DataArrays

    Raises:
        ValueError: If no valid navigation coordinates found
    """
    for model in models:
        model_name = model['name']
        model_dir = model['model_dir']
        year = model['year']

        run_dir = Path(model_dir) / model_name
        ptrc_file = run_dir / f"ORCA2_1m_{year}0101_{year}1231_ptrc_T.nc"

        if ptrc_file.exists():
            try:
                ds = xr.open_dataset(str(ptrc_file), decode_times=False)

                # Try different navigation coordinate names
                if 'nav_lon' in ds and 'nav_lat' in ds:
                    nav_lon = ds['nav_lon'].copy()
                    nav_lat = ds['nav_lat'].copy()
                elif 'lon' in ds and 'lat' in ds:
                    nav_lon = ds['lon'].copy()
                    nav_lat = ds['lat'].copy()
                else:
                    ds.close()
                    continue

                ds.close()
                return nav_lon, nav_lat

            except Exception as e:
                print(f"Warning: Error loading navigation from {model_name}: {e}")
                continue

    raise ValueError("No valid navigation coordinates found in any model")


def load_surface_data(
    models: List[Dict],
    variable: str,
    plotter: OceanMapPlotter
) -> Dict[str, xr.DataArray]:
    """
    Load surface (2D) data for a variable from multiple models.

    Args:
        models: List of dicts with 'name', 'year', 'model_dir' keys
        variable: Variable name
        plotter: OceanMapPlotter instance

    Returns:
        Dict mapping model names to 2D surface DataArrays
    """
    # Determine file type based on variable
    diagnostic_vars = ['Cflx', 'TChl', '_TChl', 'PPT', 'EXP', '_EXP', '_PPINT',
                       '_SP', '_RESIDUALINT', '_eratio', '_Teff']

    file_type = 'diad_T' if variable in diagnostic_vars else 'ptrc_T'

    # Load with preprocessing
    data = load_multimodel_data(
        models=models,
        variable=variable,
        file_type=file_type,
        plotter=plotter,
        use_preprocessing=True
    )

    # Extract surface level for 3D variables
    from map_utils import get_variable_metadata

    for model_name, var_data in data.items():
        if var_data is None:
            continue

        # Take appropriate depth level if 3D
        if 'deptht' in var_data.dims or 'nav_lev' in var_data.dims:
            depth_dim = 'deptht' if 'deptht' in var_data.dims else 'nav_lev'

            # Get metadata for depth index
            meta = get_variable_metadata(variable)
            depth_index = meta.get('depth_index', 0)

            # Extract surface or specified depth
            data[model_name] = var_data.isel({depth_dim: depth_index})

        # Apply land mask
        data[model_name] = plotter.apply_mask(data[model_name])

    return data
