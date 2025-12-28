"""
Numba-optimized remineralization length scale (RLS) calculation for carbon export flux.

RLS (z*) is the depth scale over which export flux decreases by a factor of e,
measured from a reference depth z0 = MLD + 10m.
"""

import numpy as np
from numba import njit


@njit
def calculate_rls_numba(exp_flux, depth_vals, mld_vals):
    """
    Calculate remineralization length scale (RLS/z*) using Numba-optimized loops.

    Args:
        exp_flux: Export flux array (depth, y, x) - time-averaged
        depth_vals: 1D array of depth values in meters
        mld_vals: Mixed layer depth array (y, x) in meters

    Returns:
        rls: 2D array (y, x) of remineralization length scale in meters
    """
    nz, ny, nx = exp_flux.shape
    rls = np.full((ny, nx), np.nan)

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
                if np.isnan(flux_k):
                    continue  # Skip NaN values, keep searching deeper
                if flux_k <= target_flux:
                    if k == z0_idx:
                        rls[i, j] = depth_vals[k] - z0_depth
                    else:
                        flux_before = exp_flux[k - 1, i, j]
                        # Only interpolate if flux is decreasing (flux_before > flux_k)
                        if not np.isnan(flux_before) and flux_before > flux_k:
                            frac = (flux_before - target_flux) / (flux_before - flux_k)
                            depth_at_target = depth_vals[k - 1] + frac * (depth_vals[k] - depth_vals[k - 1])
                            rls[i, j] = depth_at_target - z0_depth
                    break

    return rls
