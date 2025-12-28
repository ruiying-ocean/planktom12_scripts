"""
Remineralization length scale (RLS) calculation for carbon export flux.

RLS (z*) is the depth scale over which export flux decreases by a factor of e,
measured from a reference depth z0 = MLD + 10m.
"""

import numpy as np
import xarray as xr


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

    if has_time:
        z_star = np.zeros_like(mld_vals) * np.nan

        for t in range(poc_flux_vals.shape[0]):
            for i in range(poc_flux_vals.shape[2]):
                for j in range(poc_flux_vals.shape[3]):
                    mld_val = mld_vals[t, i, j]
                    if np.isnan(mld_val):
                        continue

                    z0_depth = mld_val + 10
                    z0_idx = np.argmin(np.abs(depth_vals - z0_depth))

                    flux_column = poc_flux_vals[t, :, i, j]
                    flux_z0 = flux_column[z0_idx]

                    if np.isnan(flux_z0) or flux_z0 <= 0:
                        continue

                    target_flux = flux_z0 / np.e

                    flux_below = flux_column[z0_idx:]
                    depth_below = depth_vals[z0_idx:]

                    valid = ~np.isnan(flux_below) & (flux_below > 0)
                    if valid.sum() < 2:
                        continue

                    flux_valid = flux_below[valid]
                    depth_valid = depth_below[valid]

                    if flux_valid[-1] > target_flux:
                        continue

                    idx_crossing = np.where(flux_valid <= target_flux)[0]
                    if len(idx_crossing) == 0:
                        continue

                    idx = idx_crossing[0]
                    if idx == 0:
                        z_star[t, i, j] = depth_valid[0] - z0_depth
                    else:
                        flux_before = flux_valid[idx - 1]
                        flux_after = flux_valid[idx]
                        depth_before = depth_valid[idx - 1]
                        depth_after = depth_valid[idx]

                        frac = (flux_before - target_flux) / (flux_before - flux_after)
                        depth_at_target = depth_before + frac * (depth_after - depth_before)
                        z_star[t, i, j] = depth_at_target - z0_depth

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
        # No time dimension case
        z_star = np.zeros_like(mld_vals) * np.nan

        for i in range(poc_flux_vals.shape[1]):
            for j in range(poc_flux_vals.shape[2]):
                mld_val = mld_vals[i, j]
                if np.isnan(mld_val):
                    continue

                z0_depth = mld_val + 10
                z0_idx = np.argmin(np.abs(depth_vals - z0_depth))

                flux_column = poc_flux_vals[:, i, j]
                flux_z0 = flux_column[z0_idx]

                if np.isnan(flux_z0) or flux_z0 <= 0:
                    continue

                target_flux = flux_z0 / np.e

                flux_below = flux_column[z0_idx:]
                depth_below = depth_vals[z0_idx:]

                valid = ~np.isnan(flux_below) & (flux_below > 0)
                if valid.sum() < 2:
                    continue

                flux_valid = flux_below[valid]
                depth_valid = depth_below[valid]

                if flux_valid[-1] > target_flux:
                    continue

                idx_crossing = np.where(flux_valid <= target_flux)[0]
                if len(idx_crossing) == 0:
                    continue

                idx = idx_crossing[0]
                if idx == 0:
                    z_star[i, j] = depth_valid[0] - z0_depth
                else:
                    flux_before = flux_valid[idx - 1]
                    flux_after = flux_valid[idx]
                    depth_before = depth_valid[idx - 1]
                    depth_after = depth_valid[idx]

                    frac = (flux_before - target_flux) / (flux_before - flux_after)
                    depth_at_target = depth_before + frac * (depth_after - depth_before)
                    z_star[i, j] = depth_at_target - z0_depth

        if is_xarray:
            return xr.DataArray(
                z_star,
                dims=mld_field.dims,
                coords=mld_field.coords,
                name='z_star',
                attrs={'units': 'm', 'long_name': 'Remineralization length scale'}
            )
        return z_star


def calculate_dic_inv(ptrc, volume, land_mask_3d):
    """
    Calculate dissolved inorganic carbon inventory from a time-mean DIC field.

    Args:
        ptrc: Dataset containing 'DIC' variable (time-averaged)
        volume: 3D volume array
        land_mask_3d: 3D land mask array

    Returns:
        xr.DataArray: DIC inventory in PgC
    """
    dic_inv = ptrc['DIC'] * volume * land_mask_3d * 1e3
    dic_inv_c = dic_inv.sum(dim=['x', 'y', 'deptht']) / 1E15 * 12.01
    return dic_inv_c
