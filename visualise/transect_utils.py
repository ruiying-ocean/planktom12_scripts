#!/usr/bin/env python3
"""
Shared utilities for creating vertical transect plots.
Used by both single-model and multi-model visualization scripts.
"""

import numpy as np
import xarray as xr


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
    """
    Extract latitude from the central meridian of a 2D nav_lat array.

    Args:
        nav_lat: 2D latitude array

    Returns:
        1D array of latitude values along central meridian
    """
    mid_x = nav_lat.shape[1] // 2
    central_lat = nav_lat[:, mid_x]
    return central_lat


# Common transect definitions
TRANSECTS = {
    'atlantic': {
        'name': 'Atlantic',
        'lon': -35.0,  # 35°W
        'label': '35°W'
    },
    'pacific': {
        'name': 'Pacific',
        'lon': -170.0,  # 170°W
        'label': '170°W'
    }
}


def get_transect_config(name='atlantic'):
    """
    Get configuration for a standard transect.

    Args:
        name: Transect name ('atlantic' or 'pacific')

    Returns:
        Dict with 'name', 'lon', and 'label' keys
    """
    return TRANSECTS.get(name.lower(), TRANSECTS['atlantic'])
