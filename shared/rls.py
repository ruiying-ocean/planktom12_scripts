"""
Remineralization length scale (RLS) calculation for carbon export flux.

RLS (z*) is the depth scale over which export flux decreases by a factor of e,
measured from a reference depth z0 = MLD + 10m.
"""

import numpy as np
import xarray as xr
from numba import njit


@njit
def _calculate_rls_core(exp_flux, depth_vals, mld_vals):
    """
    Numba-optimized core RLS calculation for 3D data (depth, y, x).

    Args:
        exp_flux: 3D numpy array (depth, y, x)
        depth_vals: 1D numpy array of depth values
        mld_vals: 2D numpy array (y, x) of MLD values

    Returns:
        2D numpy array (y, x) of z* values
    """
    nz, ny, nx = exp_flux.shape
    z_star = np.full((ny, nx), np.nan)

    for i in range(ny):
        for j in range(nx):
            mld_val = mld_vals[i, j]
            if np.isnan(mld_val):
                continue

            z0_depth = mld_val + 10
            z0_idx = np.argmin(np.abs(depth_vals - z0_depth))

            flux_z0 = exp_flux[z0_idx, i, j]
            if np.isnan(flux_z0) or flux_z0 <= 0:
                continue

            target_flux = flux_z0 / np.e

            for k in range(z0_idx, nz):
                flux_k = exp_flux[k, i, j]
                if not np.isnan(flux_k) and flux_k <= target_flux:
                    if k == z0_idx:
                        z_star[i, j] = depth_vals[k] - z0_depth
                    else:
                        flux_before = exp_flux[k - 1, i, j]
                        if not np.isnan(flux_before):
                            frac = (flux_before - target_flux) / (flux_before - flux_k)
                            depth_at_target = depth_vals[k - 1] + frac * (depth_vals[k] - depth_vals[k - 1])
                            z_star[i, j] = depth_at_target - z0_depth
                    break

    return z_star


def calculate_rls_numba(poc_flux, depth, mld_field):
    """
    Calculate remineralization length scale (RLS/z*).

    Handles both numpy arrays and xarray DataArrays, with or without time dimension.
    Returns xarray DataArray when inputs are xarray, numpy array otherwise.

    Args:
        poc_flux: Export flux array - either (time, depth, y, x) or (depth, y, x)
        depth: 1D array of depth values in meters
        mld_field: Mixed layer depth array - either (time, y, x) or (y, x)

    Returns:
        z* values - xarray DataArray if inputs are xarray, numpy array otherwise
    """
    # Check if inputs are xarray
    is_xarray = hasattr(poc_flux, 'dims')

    # Check for time dimension (only possible with xarray)
    has_time = is_xarray and 'time_counter' in poc_flux.dims and 'time_counter' in mld_field.dims

    # Extract numpy arrays from xarray if needed
    if is_xarray:
        poc_flux_vals = poc_flux.values
        depth_vals = depth.values if hasattr(depth, 'values') else depth
        mld_vals = mld_field.values
    else:
        poc_flux_vals = poc_flux
        depth_vals = depth
        mld_vals = mld_field

    # Ensure depth_vals is contiguous float64 for numba
    depth_vals = np.ascontiguousarray(depth_vals, dtype=np.float64)

    if has_time:
        nt = poc_flux_vals.shape[0]
        ny, nx = mld_vals.shape[1], mld_vals.shape[2]
        z_star = np.full((nt, ny, nx), np.nan)

        for t in range(nt):
            z_star[t] = _calculate_rls_core(
                np.ascontiguousarray(poc_flux_vals[t]),
                depth_vals,
                np.ascontiguousarray(mld_vals[t])
            )

        if is_xarray:
            return xr.DataArray(
                z_star,
                dims=mld_field.dims,
                coords=mld_field.coords,
                name='z_star',
                attrs={'units': 'm', 'long_name': 'Remineralization length scale'}
            )
        return z_star
    else:
        # No time dimension case - call core function directly
        z_star = _calculate_rls_core(
            np.ascontiguousarray(poc_flux_vals),
            depth_vals,
            np.ascontiguousarray(mld_vals)
        )

        if is_xarray:
            return xr.DataArray(
                z_star,
                dims=mld_field.dims,
                coords=mld_field.coords,
                name='z_star',
                attrs={'units': 'm', 'long_name': 'Remineralization length scale'}
            )
        return z_star
