"""
Carbon inventory calculations.
"""

import xarray as xr


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
