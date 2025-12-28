#!/usr/bin/env python
"""
Processing module for analyser system.

This module provides unified processing functions that work across all variable types,
eliminating code duplication in the original analyser.py
"""

import logging
import numpy as np
from typing import List, Tuple, Callable, Any, Union
from analyser_io import find_variable_in_files, get_depth_coordinate
from analyser_functions import surfaceData, volumeData, integrateData, volumeDataAverage, levelData

log = logging.getLogger("Processor")


# ---------- REGION MASK CACHE ----------

class LazyRegionMaskCache:
    """
    Lazy region mask cache that computes masks on-demand and caches results.

    This provides the same interface as the precomputed dictionary but only
    computes masks when they are first accessed, significantly reducing startup
    time when only a few regions are used.

    Uses NumPy broadcasting for efficient 3D mask computation.
    """

    def __init__(self, landMask: np.ndarray, volMask: np.ndarray, regions: List):
        """
        Initialize the lazy cache.

        Args:
            landMask: 2D land mask array
            volMask: 3D volume mask array
            regions: List of region masks
        """
        self._landMask = landMask
        self._volMask = volMask
        self._regions = regions
        self._cache = {}
        self._access_count = 0

    def __getitem__(self, key: str) -> np.ndarray:
        """Get a mask, computing it lazily if not cached."""
        if key not in self._cache:
            self._cache[key] = self._compute_mask(key)
            log.debug(f"Computed and cached mask: {key}")
        self._access_count += 1
        return self._cache[key]

    def __contains__(self, key: str) -> bool:
        """Check if a key could be valid (not whether it's cached)."""
        parts = key.split('_')
        if len(parts) != 2:
            return False
        mask_type, reg_str = parts
        try:
            reg = int(reg_str)
            return mask_type in ('land', 'vol') and -1 <= reg < len(self._regions)
        except ValueError:
            return False

    def _compute_mask(self, key: str) -> np.ndarray:
        """Compute a single mask on demand."""
        parts = key.split('_')
        mask_type = parts[0]
        reg = int(parts[1])

        if mask_type == 'land':
            if reg == -1:
                return self._landMask.copy()
            return self._landMask * self._regions[reg]

        elif mask_type == 'vol':
            if reg == -1:
                return self._volMask.copy()
            # Use NumPy broadcasting instead of loop: region[np.newaxis, :, :]
            # This broadcasts the 2D region mask across all depth levels
            return self._volMask * self._regions[reg][np.newaxis, :, :]

        raise KeyError(f"Unknown mask type: {mask_type}")

    def get_stats(self) -> dict:
        """Get cache statistics."""
        return {
            'cached_masks': len(self._cache),
            'total_accesses': self._access_count,
            'possible_masks': (len(self._regions) + 1) * 2  # +1 for global, *2 for land/vol
        }


def precompute_region_masks(landMask: np.ndarray, volMask: np.ndarray, regions: List) -> dict:
    """
    Pre-compute all region mask combinations to avoid repeated calculations.

    This computes 2D land masks and 3D volume masks for all 54 regions once,
    then stores them in a dictionary for instant lookup during processing.

    NOTE: For better performance with few regions, use LazyRegionMaskCache instead.

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

            # Use NumPy broadcasting for 3D mask (faster than loop)
            cache[f'vol_{reg}'] = volMask * regions[reg][np.newaxis, :, :]

    log.info(f"Pre-computed {len(cache)} region masks ({len(region_indices)} regions × 2 mask types)")
    return cache


def create_region_mask_cache(
    landMask: np.ndarray,
    volMask: np.ndarray,
    regions: List,
    lazy: bool = True
) -> Union[LazyRegionMaskCache, dict]:
    """
    Create a region mask cache.

    Args:
        landMask: 2D land mask array
        volMask: 3D volume mask array
        regions: List of region masks
        lazy: If True, use lazy cache (recommended). If False, precompute all.

    Returns:
        LazyRegionMaskCache or dict depending on lazy parameter
    """
    if lazy:
        log.info("Using lazy region mask cache (masks computed on demand)")
        return LazyRegionMaskCache(landMask, volMask, regions)
    else:
        return precompute_region_masks(landMask, volMask, regions)


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
    null_monthly = np.full((12, 6), -1.0)

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

            # For level/integration/average processing, convert depth_m to level index
            if processor_type == 'level' and 'depth_m' in extra_params:
                depth_vals = get_depth_coordinate(nc_run_ids[n], nc_filenames[n])
                if depth_vals is not None:
                    target_depth = extra_params['depth_m']
                    level_idx = np.argmin(np.abs(depth_vals - target_depth))
                    actual_depth = depth_vals[level_idx]
                    log.info(f"{var_name}: using level {level_idx} = {actual_depth:.1f}m (target: {target_depth}m)")
                    extra_params['level'] = level_idx
                else:
                    log.warning(f"Cannot find depth coordinate, using default level 0")
                    extra_params['level'] = 0

            elif processor_type in ['integration', 'average'] and 'depth_from_m' in extra_params:
                depth_vals = get_depth_coordinate(nc_run_ids[n], nc_filenames[n])
                if depth_vals is not None:
                    from_m = extra_params['depth_from_m']
                    to_m = extra_params['depth_to_m']
                    from_idx = np.argmin(np.abs(depth_vals - from_m))
                    to_idx = np.argmin(np.abs(depth_vals - to_m))
                    log.info(f"{var_name}: depth {from_m}-{to_m}m -> levels {from_idx}-{to_idx} ({depth_vals[from_idx]:.1f}-{depth_vals[to_idx]:.1f}m)")
                    extra_params['depth_from'] = from_idx
                    extra_params['depth_to'] = to_idx
                else:
                    log.warning(f"Cannot find depth coordinate, using default levels 0-0")
                    extra_params['depth_from'] = 0
                    extra_params['depth_to'] = 0

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
            extra_params = {'depth_m': var.depth_m}
        elif processor_type == 'integration':
            extra_params = {'depth_from_m': var.depth_from_m, 'depth_to_m': var.depth_to_m}
        elif processor_type == 'average':
            extra_params = {'depth_from_m': var.depth_from_m, 'depth_to_m': var.depth_to_m}
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
            extra_params = {'depth_m': var[1]}
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
            extra_params = {'depth_from_m': var[1], 'depth_to_m': var[2]}
        elif processor_type == 'average':
            var_name = var[0]
            units = var[3]
            lon_lim = var[4]
            lat_lim = var[5]
            reg = var[7]
            extra_params = {'depth_from_m': var[1], 'depth_to_m': var[2]}
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
            region_vol_mask *= regions[reg][np.newaxis, :, :]

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
        # extra_params['level'] is computed from depth_m in process_variables
        level_idx = extra_params.get('level', 0)
        return processor_func(
            data, val_lons, val_lats, units_to_use,
            mask_area, region_land_mask, region_vol_mask, missingVal,
            lon_limit, lat_limit, level_idx
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
    null_monthly = np.full((12, 6), -1.0)

    for var in variables:
        for n in range(len(nc_run_ids)):
            # Extract configuration
            if hasattr(var, 'name'):
                var_names = var.name.split('+')
                depth_from_m = var.depth_from_m
                depth_to_m = var.depth_to_m
                units = var.units
                reg = var.region
                lon_lim = var.lon_limit
                lat_lim = var.lat_limit
            else:
                var_names = var[0].split('+')
                depth_from_m = var[1]
                depth_to_m = var[2]
                units = var[3]
                reg = var[7]
                lon_lim = var[4]
                lat_lim = var[5]

            # Convert depth_m to level indices
            depth_vals = get_depth_coordinate(nc_run_ids[n], nc_filenames[n])
            if depth_vals is not None:
                depth_from = np.argmin(np.abs(depth_vals - depth_from_m))
                depth_to = np.argmin(np.abs(depth_vals - depth_to_m))
                log.info(f"{var_names}: depth {depth_from_m}-{depth_to_m}m -> levels {depth_from}-{depth_to}")
            else:
                log.warning(f"Cannot find depth coordinate, using default levels 0-0")
                depth_from = 0
                depth_to = 0

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
                        data = np.broadcast_to(
                            data[:, np.newaxis, :, :],
                            (data.shape[0], volMask.shape[0], data.shape[1], data.shape[2])
                        ).copy()

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
                    region_vol_mask *= regions[reg][np.newaxis, :, :]

            # Process each variable and sum results
            output_all_total = 0
            monthly_all_output = np.zeros((12, 6))

            for data in all_data:
                output = volumeDataAverage(
                    data, val_lons, val_lats, depth_from, depth_to,
                    units_to_use, mask_vol, landMask, region_vol_mask,
                    missingVal, lon_limit, lat_limit
                )
                output_all_total = output_all_total + output[0]
                monthly_all_output += np.array(output[1])

            output_total = (output_all_total, monthly_all_output)

            # Store results
            if hasattr(var, 'results'):
                var.results.append(output_total)
            else:
                var[-1].append(output_total)
