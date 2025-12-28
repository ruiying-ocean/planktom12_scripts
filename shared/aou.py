"""
Apparent Oxygen Utilization (AOU) calculations.

AOU = O2_saturation - O2_measured

Uses full TEOS-10 standard following Garcia & Gordon (1992, 1993).
"""

import numpy as np
import xarray as xr
import gsw


def calculate_aou(
    o2: xr.DataArray,
    temp: xr.DataArray,
    sal: xr.DataArray,
    depth_index: int = 17,
    lon: xr.DataArray = None,
    lat: xr.DataArray = None
) -> xr.DataArray:
    """
    Calculate Apparent Oxygen Utilization (AOU) at a specific depth.

    Uses full TEOS-10 standard:
    1. Convert practical salinity to absolute salinity
    2. Convert in-situ temperature to conservative temperature
    3. Calculate O2 saturation using gsw.O2sol
    4. Use actual in-situ density for unit conversion

    Args:
        o2: Oxygen concentration (mol/L) with depth coordinate
        temp: Temperature (in-situ C)
        sal: Salinity (practical salinity)
        depth_index: Depth level index (default 17 for ~300m)
        lon: Longitude array (optional, uses nav_lon from data or 0)
        lat: Latitude array (optional, uses nav_lat from data or 0)

    Returns:
        AOU in umol/L
    """
    # Get depth dimension and values from data
    depth_dim = 'deptht' if 'deptht' in o2.dims else 'nav_lev'
    depth_values = o2[depth_dim].values
    pressure = depth_values[depth_index]

    # Extract data at specified depth
    o2_at_depth = o2.isel({depth_dim: depth_index})
    temp_at_depth = temp.isel({depth_dim: depth_index})
    sal_at_depth = sal.isel({depth_dim: depth_index})

    # Get coordinates
    if lon is None:
        lon = o2.coords.get('nav_lon', xr.zeros_like(sal_at_depth))
    if lat is None:
        lat = o2.coords.get('nav_lat', xr.zeros_like(sal_at_depth))

    # TEOS-10 calculation
    SA = gsw.SA_from_SP(sal_at_depth, pressure, lon, lat)
    CT = gsw.CT_from_t(SA, temp_at_depth, pressure)
    o2_sat = gsw.O2sol(SA, CT, pressure, lon, lat)  # umol/kg
    rho = gsw.rho(SA, CT, pressure)  # kg/m3

    # Convert units
    o2_measured = o2_at_depth * 1e6  # mol/L to umol/L
    o2_sat_umol_L = o2_sat * (rho / 1000.0)  # umol/kg to umol/L

    aou = o2_sat_umol_L - o2_measured
    return aou


def calculate_aou_3d(
    o2: xr.DataArray,
    temp: xr.DataArray,
    sal: xr.DataArray,
    lon: xr.DataArray = None,
    lat: xr.DataArray = None
) -> xr.DataArray:
    """
    Calculate 3D AOU field for all depths (used for transects).

    Uses full TEOS-10 standard following Garcia & Gordon (1992, 1993).

    Args:
        o2: 3D oxygen concentration (mol/L) with depth coordinate
        temp: 3D temperature (in-situ C)
        sal: 3D salinity (practical salinity)
        lon: Longitude array (optional)
        lat: Latitude array (optional)

    Returns:
        3D AOU field in umol/L
    """
    # Get depth dimension and values from data
    depth_dim = 'deptht' if 'deptht' in o2.dims else 'nav_lev'
    depth_values = o2[depth_dim].values
    n_depths = len(depth_values)

    if lon is None:
        lon = o2.coords.get('nav_lon', None)
    if lat is None:
        lat = o2.coords.get('nav_lat', None)

    aou_list = []
    for k in range(n_depths):
        pressure = depth_values[k]
        o2_k = o2.isel({depth_dim: k})
        temp_k = temp.isel({depth_dim: k})
        sal_k = sal.isel({depth_dim: k})

        lon_k = lon if lon is not None else xr.zeros_like(sal_k)
        lat_k = lat if lat is not None else xr.zeros_like(sal_k)

        # TEOS-10 calculation
        SA = gsw.SA_from_SP(sal_k, pressure, lon_k, lat_k)
        CT = gsw.CT_from_t(SA, temp_k, pressure)
        o2_sat = gsw.O2sol(SA, CT, pressure, lon_k, lat_k)
        rho = gsw.rho(SA, CT, pressure)

        o2_sat_umol_L = o2_sat * (rho / 1000.0)
        o2_measured = o2_k * 1e6

        aou_k = o2_sat_umol_L - o2_measured
        aou_list.append(aou_k)

    aou = xr.concat(aou_list, dim=depth_dim)
    aou = aou.assign_coords({depth_dim: o2[depth_dim]})
    aou.name = '_AOU'

    return aou
