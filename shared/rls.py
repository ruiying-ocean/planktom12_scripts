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

            # Extract flux and depth below z0, filter to valid (non-NaN, positive) values
            flux_below = exp_flux[z0_idx:, i, j]
            depth_below = depth_vals[z0_idx:]

            # Count valid points and build filtered arrays
            valid_count = 0
            for k in range(len(flux_below)):
                if not np.isnan(flux_below[k]) and flux_below[k] > 0:
                    valid_count += 1

            if valid_count < 2:
                continue

            # Build filtered arrays
            flux_valid = np.empty(valid_count)
            depth_valid = np.empty(valid_count)
            idx = 0
            for k in range(len(flux_below)):
                if not np.isnan(flux_below[k]) and flux_below[k] > 0:
                    flux_valid[idx] = flux_below[k]
                    depth_valid[idx] = depth_below[k]
                    idx += 1

            # Check if flux ever drops below target
            if flux_valid[-1] > target_flux:
                continue

            # Find first crossing point in valid array
            crossing_idx = -1
            for k in range(valid_count):
                if flux_valid[k] <= target_flux:
                    crossing_idx = k
                    break

            if crossing_idx == -1:
                continue

            if crossing_idx == 0:
                rls[i, j] = depth_valid[0] - z0_depth
            else:
                flux_before = flux_valid[crossing_idx - 1]
                flux_after = flux_valid[crossing_idx]
                depth_before = depth_valid[crossing_idx - 1]
                depth_after = depth_valid[crossing_idx]

                frac = (flux_before - target_flux) / (flux_before - flux_after)
                depth_at_target = depth_before + frac * (depth_after - depth_before)
                rls[i, j] = depth_at_target - z0_depth

    return rls
