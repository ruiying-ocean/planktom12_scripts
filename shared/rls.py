"""
Remineralization length scale (RLS) calculation for carbon export flux.

RLS (z*) is the depth scale over which export flux decreases by a factor of e,
measured from a reference depth z0 = MLD + 10m.
"""

import numpy as np


def calculate_rls_numba(poc_flux, depth, mld_field):
    """
    Calculate remineralization length scale (RLS/z*).

    Args:
        poc_flux: Export flux array (depth, y, x) - time-averaged, as numpy array
        depth: 1D array of depth values in meters
        mld_field: Mixed layer depth array (y, x) in meters, as numpy array

    Returns:
        z_star: 2D array (y, x) of remineralization length scale in meters
    """
    z_star = np.zeros_like(mld_field) * np.nan

    for i in range(poc_flux.shape[1]):
        for j in range(poc_flux.shape[2]):
            mld_val = mld_field[i, j]
            if np.isnan(mld_val):
                continue

            z0_depth = mld_val + 10
            z0_idx = np.argmin(np.abs(depth - z0_depth))

            flux_column = poc_flux[:, i, j]
            flux_z0 = flux_column[z0_idx]

            if np.isnan(flux_z0) or flux_z0 <= 0:
                continue

            target_flux = flux_z0 / np.e

            flux_below = flux_column[z0_idx:]
            depth_below = depth[z0_idx:]

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

    return z_star
