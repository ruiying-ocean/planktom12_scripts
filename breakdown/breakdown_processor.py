#!/usr/bin/env python
"""
Processing module for breakdown system.

This module provides unified processing functions that work across all variable types,
eliminating code duplication in the original breakdown.py
"""

import logging
import numpy as np
from typing import List, Tuple, Callable, Any
from breakdown_io import find_variable_in_files
from breakdown_functions import surfaceData, volumeData, intergrateData, volumeDataAverage, levelData

log = logging.getLogger("Processor")


# ---------- REGION MASK CACHE ----------

def precompute_region_masks(landMask: np.ndarray, volMask: np.ndarray, regions: List) -> dict:
    """
    Pre-compute all region mask combinations to avoid repeated calculations.

    This computes 2D land masks and 3D volume masks for all 54 regions once,
    then stores them in a dictionary for instant lookup during processing.

    Args:
        landMask: 2D land mask array
        volMask: 3D volume mask array
        regions: List of region masks

    Returns:
        Dictionary with keys like 'land_-1', 'land_0', 'vol_-1', 'vol_0', etc.
    """
    cache = {}

    # Pre-compute for global (-1) and all defined regions
    region_indices = [-1] + list(range(len(regions)))

    for reg in region_indices:
        if reg == -1:
            # Global: no regional masking
            cache[f'land_{reg}'] = landMask.copy()
            cache[f'vol_{reg}'] = volMask.copy()
        else:
            # Regional: apply region mask
            cache[f'land_{reg}'] = landMask * regions[reg]

            # For 3D volume mask, broadcast region mask to all depth levels
            region_vol = volMask.copy()
            for z in range(region_vol.shape[0]):
                region_vol[z, :, :] = region_vol[z, :, :] * regions[reg]
            cache[f'vol_{reg}'] = region_vol

    log.info(f"Pre-computed {len(cache)} region masks ({len(region_indices)} regions × 2 mask types)")
    return cache


# ---------- UNIFIED PROCESSING ----------

def process_variables(
    variables: List,
    nc_run_ids: List,
    nc_filenames: List,
    units_lookup: dict,
    regions: List,
    landMask: np.ndarray,
    volMask: np.ndarray,
    mask_area: np.ndarray,
    mask_vol: np.ndarray,
    missingVal: float,
    processor_func: Callable,
    processor_type: str,
    region_mask_cache: dict = None
):
    """
    Unified function to process any type of variable.

    This replaces the 5 nearly-identical processing loops in the original code.

    Args:
        variables: List of variable configuration objects
        nc_run_ids: List of lists of NetCDF file handles per year
        nc_filenames: List of lists of filenames per year
        units_lookup: Dictionary mapping unit names to conversion factors
        regions: List of region masks
        landMask: 2D land mask array
        volMask: 3D volume mask array
        mask_area: 2D area mask
        mask_vol: 3D volume mask
        missingVal: Missing value indicator
        processor_func: Function to call for processing (surfaceData, volumeData, etc.)
        processor_type: Type of processing ('surface', 'level', 'volume', 'integration', 'average')
        region_mask_cache: Pre-computed region masks (recommended for performance)
    """
    null_annual = -1
    null_monthly = np.array([np.zeros((6)) - 1 for r in range(12)])

    for var in variables:
        for n in range(len(nc_run_ids)):
            # Extract variable configuration
            var_name, units, reg, lon_lim, lat_lim, extra_params = _extract_var_config(var, processor_type)

            # Set default lat/lon limits
            if reg == -1:
                # lon_lim and lat_lim are already tuples from parsing
                lon_limit = [float(lon_lim[0]), float(lon_lim[1])]
                lat_limit = [float(lat_lim[0]), float(lat_lim[1])]
            else:
                lon_limit = [-180, 180]
                lat_limit = [-90, 90]

            # Find variable in NetCDF files
            found, data, val_lats, val_lons, filename = find_variable_in_files(
                nc_run_ids[n], nc_filenames[n], var_name
            )

            if not found:
                log.info(f"{var_name} not found")
                if hasattr(var, 'results'):
                    var.results.append((null_annual, null_monthly))
                else:
                    var[-1].append((null_annual, null_monthly))
                continue

            # Handle dimension checks
            if processor_type == 'surface':
                if len(data.shape) == 4:
                    log.info(f"{var_name} is volume data, taking surface values")
                    data = data[:, 0, :, :]

            elif processor_type in ['level', 'volume', 'integration', 'average']:
                if len(data.shape) == 3:
                    log.info(f"{var_name} in {filename} is 2D data, try using Surface")
                    if hasattr(var, 'results'):
                        var.results.append((null_annual, null_monthly))
                    else:
                        var[-1].append((null_annual, null_monthly))
                    continue

            # Get unit conversion factor
            try:
                units_to_use = units_lookup[units]
            except KeyError:
                log.info(f"Unit: {units} not found, using raw data")
                units_to_use = 1

            # Prepare region masks (use cache if available)
            if region_mask_cache is not None:
                region_land_mask = region_mask_cache[f'land_{reg}']
                region_vol_mask = region_mask_cache[f'vol_{reg}']
            else:
                region_land_mask, region_vol_mask = _prepare_region_masks(
                    landMask, volMask, regions, reg, processor_type
                )

            # Call appropriate processing function
            output_total = _call_processor(
                processor_type,
                processor_func,
                data,
                val_lons,
                val_lats,
                units_to_use,
                mask_area,
                mask_vol,
                region_land_mask,
                region_vol_mask,
                missingVal,
                lon_limit,
                lat_limit,
                extra_params
            )

            # Store results
            if hasattr(var, 'results'):
                var.results.append(output_total)
            else:
                var[-1].append(output_total)


def _extract_var_config(var, processor_type: str) -> Tuple:
    """
    Extract variable configuration based on processor type.

    Returns:
        Tuple of (var_name, units, region, lon_lim, lat_lim, extra_params)
    """
    if hasattr(var, 'name'):
        # Using dataclass objects
        var_name = var.name
        units = var.units
        reg = var.region
        lon_lim = var.lon_limit
        lat_lim = var.lat_limit

        if processor_type == 'level':
            extra_params = {'level': var.level}
        elif processor_type == 'integration':
            extra_params = {'depth_from': var.depth_from, 'depth_to': var.depth_to}
        elif processor_type == 'average':
            extra_params = {'depth_from': var.depth_from, 'depth_to': var.depth_to}
        else:
            extra_params = {}

    else:
        # Using legacy list format
        if processor_type == 'surface':
            var_name = var[0]
            units = var[1]
            lon_lim = var[2]
            lat_lim = var[3]
            reg = var[5]
            extra_params = {}
        elif processor_type == 'level':
            var_name = var[0]
            units = var[2]
            lon_lim = var[3]
            lat_lim = var[4]
            reg = var[6]
            extra_params = {'level': var[1]}
        elif processor_type == 'volume':
            var_name = var[0]
            units = var[1]
            lon_lim = var[2]
            lat_lim = var[3]
            reg = var[5]
            extra_params = {}
        elif processor_type == 'integration':
            var_name = var[0]
            units = var[3]
            lon_lim = var[4]
            lat_lim = var[5]
            reg = var[7]
            extra_params = {'depth_from': var[1], 'depth_to': var[2]}
        elif processor_type == 'average':
            var_name = var[0]
            units = var[3]
            lon_lim = var[4]
            lat_lim = var[5]
            reg = var[7]
            extra_params = {'depth_from': var[1], 'depth_to': var[2]}
        else:
            var_name = var[0]
            units = var[1]
            lon_lim = var[2]
            lat_lim = var[3]
            reg = var[5]
            extra_params = {}

    return var_name, units, reg, lon_lim, lat_lim, extra_params


def _prepare_region_masks(
    landMask: np.ndarray,
    volMask: np.ndarray,
    regions: List,
    reg: int,
    processor_type: str
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare region-specific masks.

    Returns:
        Tuple of (region_land_mask, region_vol_mask)
    """
    region_land_mask = np.copy(landMask)
    region_vol_mask = np.copy(volMask)

    if reg != -1:
        # Apply region mask to land mask
        region_land_mask = region_land_mask * regions[reg]

        # Apply region mask to volume mask for 3D processors
        if processor_type in ['level', 'volume', 'integration', 'average']:
            for z in range(region_vol_mask.shape[0]):
                region_vol_mask[z, :, :] = region_vol_mask[z, :, :] * regions[reg]

    return region_land_mask, region_vol_mask


def _call_processor(
    processor_type: str,
    processor_func: Callable,
    data: np.ndarray,
    val_lons: np.ndarray,
    val_lats: np.ndarray,
    units_to_use: float,
    mask_area: np.ndarray,
    mask_vol: np.ndarray,
    region_land_mask: np.ndarray,
    region_vol_mask: np.ndarray,
    missingVal: float,
    lon_limit: List,
    lat_limit: List,
    extra_params: dict
) -> Any:
    """
    Call the appropriate processing function with correct parameters.
    """
    if processor_type == 'surface':
        return processor_func(
            data, val_lons, val_lats, units_to_use,
            mask_area, region_land_mask, region_vol_mask, missingVal,
            lon_limit, lat_limit
        )

    elif processor_type == 'level':
        return processor_func(
            data, val_lons, val_lats, units_to_use,
            mask_area, region_land_mask, region_vol_mask, missingVal,
            lon_limit, lat_limit, extra_params['level']
        )

    elif processor_type == 'volume':
        if len(data.shape) == 4:
            return processor_func(
                data, val_lons, val_lats, units_to_use,
                mask_vol, region_land_mask, region_vol_mask, missingVal,
                lon_limit, lat_limit
            )

    elif processor_type == 'integration':
        return processor_func(
            data, val_lons, val_lats,
            extra_params['depth_from'], extra_params['depth_to'],
            units_to_use, mask_vol, region_land_mask, region_vol_mask,
            missingVal, lon_limit, lat_limit
        )

    elif processor_type == 'average':
        return processor_func(
            data, val_lons, val_lats,
            extra_params['depth_from'], extra_params['depth_to'],
            units_to_use, mask_vol, region_land_mask, region_vol_mask,
            missingVal, lon_limit, lat_limit
        )


def process_average_variables_special(
    variables: List,
    nc_run_ids: List,
    nc_filenames: List,
    units_lookup: dict,
    regions: List,
    landMask: np.ndarray,
    volMask: np.ndarray,
    mask_vol: np.ndarray,
    missingVal: float,
    region_mask_cache: dict = None
):
    """
    Special processing for average variables that can sum multiple variables.

    This handles the unique case where varTotalAve can have multiple variables
    specified with '+' separator (e.g., "CHL1+CHL2+CHL3").
    """
    null_annual = -1
    null_monthly = np.array([np.zeros((6)) - 1 for r in range(12)])

    for var in variables:
        for n in range(len(nc_run_ids)):
            # Extract configuration
            if hasattr(var, 'name'):
                var_names = var.name.split('+')
                depth_from = var.depth_from
                depth_to = var.depth_to
                units = var.units
                reg = var.region
                lon_lim = var.lon_limit
                lat_lim = var.lat_limit
            else:
                var_names = var[0].split('+')
                depth_from = var[1]
                depth_to = var[2]
                units = var[3]
                reg = var[7]
                lon_lim = var[4]
                lat_lim = var[5]

            # Set lat/lon limits
            if reg == -1:
                lon_limit = [float(lon_lim[0]), float(lon_lim[1])]
                lat_limit = [float(lat_lim[0]), float(lat_lim[1])]
            else:
                lon_limit = [-180, 180]
                lat_limit = [-90, 90]

            # Find all variables
            all_data = []
            found = False
            for var_name in var_names:
                found_var, data, val_lats, val_lons, filename = find_variable_in_files(
                    nc_run_ids[n], nc_filenames[n], var_name
                )

                if found_var:
                    # Handle 3D data - expand to 4D
                    if len(data.shape) == 3:
                        new_data = np.zeros((data.shape[0], volMask.shape[0], data.shape[1], data.shape[2]))
                        for z in range(volMask.shape[0]):
                            new_data[:, z, :, :] = data
                        data = new_data

                    if len(data.shape) == 4:
                        all_data.append(data)
                        found = True
                        log.info(f"{var_name} found in {filename}")
                else:
                    log.info(f"Not all of {var_name} are found in files")

            if not found:
                log.info(f"{var_names} not found")
                if hasattr(var, 'results'):
                    var.results.append((null_annual, null_monthly))
                else:
                    var[-1].append((null_annual, null_monthly))
                continue

            # Get unit conversion
            try:
                units_to_use = units_lookup[units]
            except KeyError:
                log.info(f"Unit: {units} not found, using raw data")
                units_to_use = 1

            # Prepare region masks (use cache if available)
            if region_mask_cache is not None:
                region_vol_mask = region_mask_cache[f'vol_{reg}']
            else:
                region_vol_mask = np.copy(volMask)
                if reg != -1:
                    for z in range(region_vol_mask.shape[0]):
                        region_vol_mask[z, :, :] = region_vol_mask[z, :, :] * regions[reg]

            # Process each variable and sum results
            output_all_total = 0
            monthly_all_output = np.array([np.zeros((6)) for r in range(12)])

            for data in all_data:
                output = volumeDataAverage(
                    data, val_lons, val_lats, depth_from, depth_to,
                    units_to_use, mask_vol, landMask, region_vol_mask,
                    missingVal, lon_limit, lat_limit
                )
                output_all_total = output_all_total + output[0]
                for m in range(12):
                    for f in range(6):
                        if len(output[1]) > m:
                            monthly_all_output[m][f] = monthly_all_output[m][f] + output[1][m][f]

            output_total = (output_all_total, monthly_all_output)

            # Store results
            if hasattr(var, 'results'):
                var.results.append(output_total)
            else:
                var[-1].append(output_total)
