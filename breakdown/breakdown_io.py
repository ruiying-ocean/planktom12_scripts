#!/usr/bin/env python
"""
I/O module for breakdown system.

This module handles:
- NetCDF file loading and searching
- Output file writing with unified formatting
"""

import logging
import glob
import os
from pathlib import Path
from netCDF4 import Dataset
import numpy as np
from typing import List, Tuple, Dict, Any, Union, Optional

log = logging.getLogger("IO")


# ---------- NETCDF LOADING ----------

def load_netcdf_files(year_from: int, year_to: int) -> Tuple[List, List, List, List]:
    """
    Load NetCDF files for the specified year range.

    Args:
        year_from: Starting year
        year_to: Ending year

    Returns:
        Tuple of (nc_run_ids, nc_runFileNames, years, failed_files)
        - nc_run_ids: List of lists of NetCDF Dataset objects per year
        - nc_runFileNames: List of lists of file names per year
        - years: List of years processed
        - failed_files: List of tuples (file_path, error_message) for files that failed to load
    """
    nc_run_ids = []
    nc_runFileNames = []
    years = []
    failed_files = []

    for year in range(year_from, year_to + 1):
        # Try to find all possible output files for this year
        file_patterns = [
            f"ORCA2_1m_{year}0101_{year}1231_grid_T.nc",
            f"ORCA2_1m_{year}0101_{year}1231_ptrc_T.nc",
            f"ORCA2_1m_{year}0101_{year}1231_diad_T.nc",
            f"ORCA2_1m_{year}0101_{year}1231_dia2d_T.nc",
            f"ORCA2_1m_{year}0101_{year}1231_dia3d_T.nc"
        ]

        nc_avail = []
        nc_names_avail = []

        for pattern in file_patterns:
            files = glob.glob(pattern)
            if len(files) > 0:
                file_path = files[0]
                try:
                    nc_id = Dataset(file_path, 'r')
                    nc_avail.append(nc_id)
                    nc_names_avail.append(file_path)
                    log.info(f"Loaded: {file_path}")
                except (OSError, RuntimeError, Exception) as e:
                    # File exists but is corrupted or unreadable
                    error_msg = f"{type(e).__name__}: {str(e)}"
                    failed_files.append((file_path, error_msg))
                    log.warning(f"SKIPPED corrupted file: {file_path}")
                    log.warning(f"  Error: {error_msg}")

        if len(nc_avail) > 0:
            nc_run_ids.append(nc_avail)
            nc_runFileNames.append(nc_names_avail)
            years.append(year)
            log.info(f"Run data for year {year}: {len(nc_avail)} files loaded")

    return nc_run_ids, nc_runFileNames, years, failed_files


def find_variable_in_files(nc_files: List, nc_filenames: List, var_name: str) -> Tuple[bool, Any, Any, Any, str]:
    """
    Search for a variable across multiple NetCDF files.

    Args:
        nc_files: List of NetCDF Dataset objects to search
        nc_filenames: List of corresponding filenames
        var_name: Variable name to find

    Returns:
        Tuple of (found, data, lats, lons, filename)
    """
    for i, nc_file in enumerate(nc_files):
        try:
            data = nc_file.variables[var_name][:].data
            lats = nc_file.variables['nav_lat'][:].data
            lons = nc_file.variables['nav_lon'][:].data
            log.info(f"{var_name} found in {nc_filenames[i]}")
            return True, data, lats, lons, nc_filenames[i]
        except KeyError:
            # Variable not in this file, try next one
            continue
        except (OSError, RuntimeError, ValueError, Exception) as e:
            # File corruption or data read error during variable access
            log.warning(f"Error reading variable '{var_name}' from {nc_filenames[i]}: {type(e).__name__}: {str(e)}")
            continue

    return False, None, None, None, None


# ---------- OUTPUT FILE WRITING ----------

class OutputWriter:
    """Unified output file writer for all breakdown output types."""

    @staticmethod
    def write_annual_csv(
        filename: Union[str, Path],
        variables: List,
        year_from: int,
        year_to: int,
        var_index: int = 0,
        include_units: bool = False,
        include_keys: bool = False
    ):
        """
        Write annual output file in clean CSV format.

        Args:
            filename: Output filename
            variables: List of variable configuration objects
            year_from: Starting year
            year_to: Ending year
            var_index: Index of variable name in config
            include_units: Whether to include units in column names
            include_keys: Whether to include keys in column names
        """
        filename = Path(filename)
        write_headers = not filename.exists()

        if write_headers:
            with filename.open('w') as f:
                # Write single header row with variable names (optionally with units/keys)
                f.write("year,")
                headers = []
                for var in variables:
                    # Use custom column name if available, otherwise build from variable name
                    if hasattr(var, 'column_name') and var.column_name:
                        col_name = var.column_name
                    elif hasattr(var, 'name'):
                        name = var.name
                        units = var.units if include_units else None
                        key = var.key if include_keys else None
                        col_name = name
                        if include_units and units:
                            col_name += f" ({units})"
                        if include_keys and key:
                            col_name += f" [{key}]"
                    else:
                        name = var[var_index]
                        units = var[1] if include_units and len(var) > 1 else None
                        key = var[-3] if include_keys else None
                        col_name = name
                        if include_units and units:
                            col_name += f" ({units})"
                        if include_keys and key:
                            col_name += f" [{key}]"

                    headers.append(col_name)

                f.write(",".join(headers))
                f.write("\n")

        # Write or append data
        with filename.open('a') as f:
            for year in range(year_from, year_to + 1):
                y = year - year_from
                f.write(f"{year},")
                values = []
                for var in variables:
                    if hasattr(var, 'results'):
                        value = var.results[y][0]
                    else:
                        value = var[-1][y][0]
                    values.append(f"{value:.4e}")
                f.write(",".join(values))
                f.write("\n")

    @staticmethod
    def write_annual_file(
        filename: str,
        variables: List,
        year_from: int,
        year_to: int,
        var_index: int,
        unit_index: int,
        key_index: int
    ):
        """
        Write annual output file.

        Args:
            filename: Output filename
            variables: List of variable configuration objects
            year_from: Starting year
            year_to: Ending year
            var_index: Index of variable name in config
            unit_index: Index of units in config
            key_index: Index of key in config
        """
        # Check if file exists to determine if we need headers
        write_headers = not os.path.exists(filename)

        if write_headers:
            with open(filename, 'w') as f:
                # Write column headers
                f.write("year\t")
                for var in variables:
                    if hasattr(var, 'name'):
                        f.write(f"{var.name}\t")
                    else:
                        f.write(f"{var[var_index]}\t")
                f.write("\n\t")

                # Write units row
                for var in variables:
                    if hasattr(var, 'units'):
                        f.write(f"{var.units}\t")
                    else:
                        f.write(f"{var[unit_index]}\t")
                f.write("\n\t")

                # Write key row
                for var in variables:
                    if hasattr(var, 'key'):
                        f.write(f"{var.key}\t")
                    else:
                        f.write(f"{var[key_index]}\t")
                f.write("\n")

        # Write or append data
        with open(filename, 'a') as f:
            for year in range(year_from, year_to + 1):
                y = year - year_from
                f.write(f"{year}\t")
                for var in variables:
                    if hasattr(var, 'results'):
                        value = var.results[y][0]
                    else:
                        value = var[-1][y][0]
                    f.write(f"{format(value, '.4e')}\t")
                f.write("\n")


    @staticmethod
    def write_spread_file(
        filename: str,
        variables: List,
        year_from: int,
        year_to: int,
        var_index: int,
        unit_index: int,
        key_index: int
    ):
        """
        Write spread (percentile) output file.

        Args:
            filename: Output filename
            variables: List of variable configuration objects
            year_from: Starting year
            year_to: Ending year
            var_index: Index of variable name in config
            unit_index: Index of units in config
            key_index: Index of key in config
        """
        # Check if file exists to determine if we need headers
        write_headers = not os.path.exists(filename)

        if write_headers:
            with open(filename, 'w') as f:
                # Write column headers
                f.write("year\t")
                f.write("month\t")
                for var in variables:
                    for i in range(5):  # 5 percentile columns per variable
                        if hasattr(var, 'name'):
                            f.write(f"{var.name}\t")
                        else:
                            f.write(f"{var[var_index]}\t")
                f.write("\n\t\t")

                # Write units row
                for var in variables:
                    for i in range(5):
                        if hasattr(var, 'units'):
                            f.write(f"{var.units}\t")
                        else:
                            f.write(f"{var[unit_index]}\t")
                f.write("\n\t\t")

                # Write key row
                for var in variables:
                    for i in range(5):
                        if hasattr(var, 'key'):
                            f.write(f"{var.key}\t")
                        else:
                            f.write(f"{var[key_index]}\t")
                f.write("\n\t\t")

                # Write percentile labels
                for var in variables:
                    f.write("min(5pt)\t")
                    f.write("25pt\t")
                    f.write("median\t")
                    f.write("75pt\t")
                    f.write("max(95pt)\t")
                f.write("\n")

        # Write or append data
        with open(filename, 'a') as f:
            for year in range(year_from, year_to + 1):
                y = year - year_from
                for m in range(12):  # 12 months
                    f.write(f"{year}\t")
                    f.write(f"{m}\t")

                    for var in variables:
                        if hasattr(var, 'results'):
                            monthly_data = var.results[y][1][m]
                        else:
                            monthly_data = var[-1][y][1][m]

                        # Write min, 25%, median, 75%, max
                        f.write(f"{format(monthly_data[1], '.4e')}\t")  # min
                        f.write(f"{format(monthly_data[2], '.4e')}\t")  # 25%
                        f.write(f"{format(monthly_data[3], '.4e')}\t")  # median
                        f.write(f"{format(monthly_data[4], '.4e')}\t")  # 75%
                        f.write(f"{format(monthly_data[5], '.4e')}\t")  # max

                    f.write("\n")


    @staticmethod
    def write_monthly_file(
        filename: str,
        variables: List,
        year_from: int,
        year_to: int,
        var_index: int,
        unit_index: int,
        key_index: int
    ):
        """
        Write monthly total output file.

        Args:
            filename: Output filename
            variables: List of variable configuration objects
            year_from: Starting year
            year_to: Ending year
            var_index: Index of variable name in config
            unit_index: Index of units in config
            key_index: Index of key in config
        """
        # Check if file exists to determine if we need headers
        write_headers = not os.path.exists(filename)

        if write_headers:
            with open(filename, 'w') as f:
                # Write column headers
                f.write("year\t")
                f.write("month\t")
                for var in variables:
                    if hasattr(var, 'name'):
                        f.write(f"{var.name}\t")
                    else:
                        f.write(f"{var[var_index]}\t")
                f.write("\n\t\t")

                # Write units row
                for var in variables:
                    if hasattr(var, 'units'):
                        f.write(f"{var.units}\t")
                    else:
                        f.write(f"{var[unit_index]}\t")
                f.write("\n\t\t")

                # Write key row
                for var in variables:
                    if hasattr(var, 'key'):
                        f.write(f"{var.key}\t")
                    else:
                        f.write(f"{var[key_index]}\t")
                f.write("\n\t\t")

                # Write "total" label
                for var in variables:
                    f.write("total\t")
                f.write("\n")

        # Write or append data
        with open(filename, 'a') as f:
            for year in range(year_from, year_to + 1):
                y = year - year_from
                for m in range(12):  # 12 months
                    f.write(f"{year}\t")
                    f.write(f"{m}\t")

                    for var in variables:
                        if hasattr(var, 'results'):
                            month_tot = var.results[y][1][m][0]
                        else:
                            month_tot = var[-1][y][1][m][0]
                        f.write(f"{format(month_tot, '.4e')}\t")

                    f.write("\n")


    @staticmethod
    def write_observation_file(
        filename: str,
        observations: List,
        year_from: int,
        year_to: int
    ):
        """
        Write observation comparison file.

        Args:
            filename: Output filename
            observations: List of observation comparison configs
            year_from: Starting year
            year_to: Ending year
        """
        # Check if file exists to determine if we need headers
        write_headers = not os.path.exists(filename)

        if write_headers:
            with open(filename, 'w') as f:
                # Write column headers
                f.write("year\t")
                for obs in observations:
                    if hasattr(obs, 'model_var'):
                        f.write(f"{obs.model_var}({obs.obs_var})\t")
                    else:
                        f.write(f"{obs[2]}({obs[1]})\t")
                f.write("\n\t")

                # Write key row
                for obs in observations:
                    if hasattr(obs, 'key'):
                        f.write(f"{obs.key}\t")
                    else:
                        f.write(f"{obs[-2]}\t")
                f.write("\n")

        # Write or append data
        with open(filename, 'a') as f:
            for year in range(year_from, year_to + 1):
                y = year - year_from
                f.write(f"{year}\t")
                for obs in observations:
                    if hasattr(obs, 'results'):
                        value = obs.results[y][0]
                    else:
                        value = obs[-1][y][0]
                    f.write(f"{format(value, '.4e')}\t")
                f.write("\n")


    @staticmethod
    def write_property_file(
        filename: str,
        prop_config,
        year_from: int,
        year_to: int
    ):
        """
        Write emergent property file.

        Args:
            filename: Output filename
            prop_config: Property configuration object
            year_from: Starting year
            year_to: Ending year
        """
        # Check if file exists to determine if we need headers
        write_headers = not os.path.exists(filename)

        if hasattr(prop_config, 'results'):
            results = prop_config.results
            prop_name = prop_config.prop_name
            variables = prop_config.variables
        else:
            results = prop_config[-1]
            prop_name = prop_config[0]
            variables = prop_config[1]

        if write_headers:
            with open(filename, 'w') as f:
                # Get headings from first result
                headings = results[0][-2]

                # Write column headers
                f.write("year\t")
                for col in headings:
                    f.write(f"{col}\t")
                f.write("\n\t")

                # Write variable info
                for col in headings:
                    if prop_name == "Trophic":
                        header = f"{variables[0]};{variables[1]};{variables[2]}"
                    else:
                        header = variables
                    f.write(f"{header}\t")
                f.write("\n")

        # Write or append data
        with open(filename, 'a') as f:
            for year in range(year_from, year_to + 1):
                y = year - year_from
                f.write(f"{year}\t")
                headings = results[0][-2]
                for c in range(len(headings)):
                    f.write(f"{format(results[y][c], '.4e')}\t")
                f.write("\n")
