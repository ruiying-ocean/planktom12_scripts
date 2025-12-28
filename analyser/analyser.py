#!/usr/bin/env python
"""
Analyser - Ocean biogeochemical model output analysis tool (Refactored)

This is a refactored version of analyser.py that uses modular components
to improve maintainability while preserving all functionality.
"""

import sys
import argparse
import logging
from pathlib import Path
import numpy as np
import math
from netCDF4 import Dataset

# Import new modular components
from analyser_config import parse_config_file
from analyser_io import load_netcdf_files, load_netcdf_files_xarray, OutputWriter
from analyser_processor import (
    process_variables, process_average_variables_special,
    precompute_region_masks, create_region_mask_cache
)
from analyser_functions import surfaceData, volumeData, integrateData, volumeDataAverage, levelData, aouData, calculate_rls

# ---------- 1 SETUP AND INITIALIZATION ----------

# Parse command line arguments
parser = argparse.ArgumentParser(
    description='Analyser - Ocean biogeochemical model output analysis tool',
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog='''
Examples:
  %(prog)s analyser_config.toml 2000 2010
  %(prog)s my_config.toml 1990 2020
    '''
)
parser.add_argument('parm_file', type=Path,
                    help='Path to the parameter configuration file (TOML or legacy format)')
parser.add_argument('year_from', type=int,
                    help='Starting year for analysis (inclusive)')
parser.add_argument('year_to', type=int,
                    help='Ending year for analysis (inclusive)')

args = parser.parse_args()
parm_file = args.parm_file
year_from = args.year_from
year_to = args.year_to

# Validate inputs
if not parm_file.exists():
    parser.error(f"Parameter file not found: {parm_file}")

if year_to < year_from:
    parser.error(f"End year ({year_to}) must be >= start year ({year_from})")

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
    datefmt='%m-%d %H:%M',
    filename=f'analyser.{year_from}_{year_to}.log',
    filemode='w'
)

console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(levelname)-5s %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)
log = logging.getLogger("Run")

log.info(f"Processing for parameters: {parm_file}")
log.info(f"Processing for years: {year_from} to {year_to}")

# ---------- 2 DEFINE UNITS AND CONSTANTS ----------
missing_val = 1E20
seconds_in_year = 3600. * 24. * 365.
peta = 1e-15
terra = 1e-12
giga = 1e-9
carbon = 12
litre = 1000
micro = 1e6
depth_area = 200. * 3.62e+14

list_of_units = {
    "PgCarbonPerYr": peta * carbon * seconds_in_year,
    "TgCarbonPerYr": terra * carbon * seconds_in_year,
    "TmolPerYr": terra * seconds_in_year,
    "TgPerYr": terra * seconds_in_year,
    "PgPerYr": peta * seconds_in_year,
    "PerYr": 1,
    "giga": 1 / giga,
    "1/giga": 1 / giga,
    "PgCarbon": peta * carbon,
    "Conc->PgCarbon": peta * carbon * litre,
    "u/L": micro,
    "PgCarbonPer200m": peta * carbon * litre * depth_area
}

log.info("UNITS:")
for key, value in list_of_units.items():
    log.info(f"{key} {value}")

# Calculate area grid
Er = 6.3781E6
Ec = 2 * math.pi * Er
area = np.zeros((180, 360))
for y in range(180):
    ang = np.radians(y - 90)
    area[y, :] = Ec * (1 / 360.) * math.cos(ang) * Ec * (1 / 360.)

# ---------- 4 PARSE CONFIGURATION ----------
log.info("Parsing configuration file...")
config = parse_config_file(str(parm_file))

# For backward compatibility, also create the legacy list structures
# These will be used by observation/property processing which hasn't been fully refactored
varSurface = []
varLevel = []
varVolume = []
varInt = []
varTotalAve = []

# Convert config objects to legacy format for processor
for var in config.surface_vars:
    varSurface.append([var.name, var.units, var.lon_limit, var.lat_limit, var.key, var.region, var.results])

for var in config.level_vars:
    varLevel.append([var.name, var.level, var.units, var.lon_limit, var.lat_limit, var.key, var.region, var.results])

for var in config.volume_vars:
    varVolume.append([var.name, var.units, var.lon_limit, var.lat_limit, var.key, var.region, var.results])

for var in config.integration_vars:
    varInt.append([var.name, var.depth_from_m, var.depth_to_m, var.units, var.lon_limit, var.lat_limit, var.key, var.region, var.results])

for var in config.average_vars:
    varTotalAve.append([var.name, var.depth_from_m, var.depth_to_m, var.units, var.lon_limit, var.lat_limit, var.key, var.region, var.results])

# ---------- 5 LOAD NETCDF MASK AND GRID FILES ----------
log.info("Loading mask and grid files...")

# Ancillary data
nc_ancil_id = Dataset(config.ancillary_data, 'r')
vmasked = nc_ancil_id.variables["VOLUME_MASKED"][:, :, :].data

# Mesh mask
nc_mesh_id = Dataset(config.mesh_mask, 'r')
tmeshmask = nc_mesh_id.variables["tmask"][0, :, :, :].data

# Basin mask
nc_basin_id = Dataset(config.basin_mask, 'r')
mask_area = nc_basin_id.variables["AREA"][:].data
mask_vol = nc_basin_id.variables["VOLUME"][:].data
landMask = np.copy(mask_area)
landMask[landMask > 0] = 1
landMask[landMask == 0] = np.nan
volMask = np.copy(mask_vol)
volMask[volMask > 0] = 1
volMask[volMask == 0] = np.nan

# Region masks
nc_reg_id = Dataset(config.region_mask, 'r')
regions = []
regions.append(nc_reg_id.variables['ARCTIC'][:].data)
regions.append(nc_reg_id.variables['A1'][:].data)
regions.append(nc_reg_id.variables['P1'][:].data)
regions.append(nc_reg_id.variables['A2'][:].data + nc_reg_id.variables['P2'][:].data)
regions.append(nc_reg_id.variables['A3'][:].data + nc_reg_id.variables['P3'][:].data + nc_reg_id.variables['I3'][:].data)
regions.append(nc_reg_id.variables['A4'][:].data + nc_reg_id.variables['P4'][:].data + nc_reg_id.variables['I4'][:].data)
regions.append(nc_reg_id.variables['A5'][:].data)
regions.append(nc_reg_id.variables['P5'][:].data)
regions.append(nc_reg_id.variables['I5'][:].data)

# RECCAP regions
nc_reccap_id = Dataset(config.reccap_mask, 'r')
regions.append(nc_reccap_id.variables['open_ocean_0'][:].data)  # 9
regions.append(nc_reccap_id.variables['open_ocean_1'][:].data)  # 10
regions.append(nc_reccap_id.variables['open_ocean_2'][:].data)  # 11
regions.append(nc_reccap_id.variables['open_ocean_3'][:].data)  # 12
regions.append(nc_reccap_id.variables['open_ocean_4'][:].data)  # 13
regions.append(nc_reccap_id.variables['atlantic_0'][:].data)    # 14
regions.append(nc_reccap_id.variables['atlantic_1'][:].data)    # 15
regions.append(nc_reccap_id.variables['atlantic_2'][:].data)    # 16
regions.append(nc_reccap_id.variables['atlantic_3'][:].data)    # 17
regions.append(nc_reccap_id.variables['atlantic_4'][:].data)    # 18
regions.append(nc_reccap_id.variables['atlantic_5'][:].data)    # 19
regions.append(nc_reccap_id.variables['pacific_0'][:].data)     # 20
regions.append(nc_reccap_id.variables['pacific_1'][:].data)     # 21
regions.append(nc_reccap_id.variables['pacific_2'][:].data)     # 22
regions.append(nc_reccap_id.variables['pacific_3'][:].data)     # 23
regions.append(nc_reccap_id.variables['pacific_4'][:].data)     # 24
regions.append(nc_reccap_id.variables['pacific_5'][:].data)     # 25
regions.append(nc_reccap_id.variables['indian_0'][:].data)      # 26
regions.append(nc_reccap_id.variables['indian_1'][:].data)      # 27
regions.append(nc_reccap_id.variables['arctic_0'][:].data)      # 28
regions.append(nc_reccap_id.variables['arctic_1'][:].data)      # 29
regions.append(nc_reccap_id.variables['arctic_2'][:].data)      # 30
regions.append(nc_reccap_id.variables['arctic_3'][:].data)      # 31
regions.append(nc_reccap_id.variables['southern_0'][:].data)    # 32
regions.append(nc_reccap_id.variables['southern_1'][:].data)    # 33
regions.append(nc_reccap_id.variables['southern_2'][:].data)    # 34
regions.append(nc_reccap_id.variables['seamask'][:].data)       # 35
regions.append(nc_reccap_id.variables['coast'][:].data)         # 36

# Additional regions
regions.append(nc_reg_id.variables['ATL'][:].data)    # 37
regions.append(nc_reg_id.variables['PAC'][:].data)    # 38
regions.append(nc_reg_id.variables['IND'][:].data)    # 39
regions.append(nc_reg_id.variables['SO'][:].data)     # 40
regions.append(nc_reg_id.variables['ARCTIC'][:].data) # 41
regions.append(nc_reg_id.variables['P1'][:].data)     # 42
regions.append(nc_reg_id.variables['P2'][:].data)     # 43
regions.append(nc_reg_id.variables['P3'][:].data)     # 44
regions.append(nc_reg_id.variables['P4'][:].data)     # 45
regions.append(nc_reg_id.variables['P5'][:].data)     # 46
regions.append(nc_reg_id.variables['A1'][:].data)     # 47
regions.append(nc_reg_id.variables['A2'][:].data)     # 48
regions.append(nc_reg_id.variables['A3'][:].data)     # 49
regions.append(nc_reg_id.variables['A4'][:].data)     # 50
regions.append(nc_reg_id.variables['A5'][:].data)     # 51
regions.append(nc_reg_id.variables['I3'][:].data)     # 52
regions.append(nc_reg_id.variables['I4'][:].data)     # 53
regions.append(nc_reg_id.variables['I5'][:].data)     # 54

for region in regions:
    region[region == 0] = np.nan

log.info(f"Data read from: {config.region_mask}")

# ---------- 6 CREATE REGION MASK CACHE ----------
log.info("Creating region mask cache...")
region_mask_cache = create_region_mask_cache(landMask, volMask, regions, lazy=True)

# ---------- 7 STREAMING MODE: Year-by-year processing ----------
# Process one year at a time to reduce memory usage

writer = OutputWriter()
all_failed_files = []

log.info(f"Processing {year_to - year_from + 1} years (streaming mode)...")

for year in range(year_from, year_to + 1):
    log.info(f"")
    log.info(f"{'='*60}")
    log.info(f"Processing year {year} ({year - year_from + 1} of {year_to - year_from + 1})")
    log.info(f"{'='*60}")

    # 1. Load NetCDF files for this year only
    nc_run_ids, nc_runFileNames, years, failed_files = load_netcdf_files(year, year)
    all_failed_files.extend(failed_files)

    if len(nc_run_ids) == 0:
        log.warning(f"No data files found for year {year}, skipping...")
        continue

    # 2. Clear previous results (use .clear() to preserve references)
    for var in config.surface_vars:
        var.results.clear()
    for var in config.level_vars:
        var.results.clear()
    for var in config.volume_vars:
        var.results.clear()
    for var in config.integration_vars:
        var.results.clear()
    for var in config.average_vars:
        var.results.clear()

    # 3. Process all variable types for this year
    log.info("Processing surface variables...")
    process_variables(
        varSurface, nc_run_ids, nc_runFileNames,
        list_of_units, regions, landMask, volMask,
        mask_area, mask_vol, missing_val,
        surfaceData, 'surface', region_mask_cache
    )

    log.info("Processing level variables...")
    process_variables(
        varLevel, nc_run_ids, nc_runFileNames,
        list_of_units, regions, landMask, volMask,
        mask_area, mask_vol, missing_val,
        levelData, 'level', region_mask_cache
    )

    log.info("Processing volume variables...")
    process_variables(
        varVolume, nc_run_ids, nc_runFileNames,
        list_of_units, regions, landMask, volMask,
        mask_area, mask_vol, missing_val,
        volumeData, 'volume', region_mask_cache
    )

    log.info("Processing integration variables...")
    process_variables(
        varInt, nc_run_ids, nc_runFileNames,
        list_of_units, regions, landMask, volMask,
        mask_area, mask_vol, missing_val,
        integrateData, 'integration', region_mask_cache
    )

    log.info("Processing average variables...")
    process_average_variables_special(
        varTotalAve, nc_run_ids, nc_runFileNames,
        list_of_units, regions, landMask, volMask,
        mask_vol, missing_val, region_mask_cache
    )

    # 3.5 Compute AOU (derived variable requiring O2, temperature, salinity)
    log.info("Computing AOU...")
    aou_result = None
    try:
        from analyser_io import find_variable_in_files, get_depth_coordinate
        # Find O2, votemper, vosaline in the loaded files
        found_o2, o2_data, lats, lons, _ = find_variable_in_files(
            nc_run_ids[0], nc_runFileNames[0], 'O2'
        )
        found_temp, temp_data, _, _, _ = find_variable_in_files(
            nc_run_ids[0], nc_runFileNames[0], 'votemper'
        )
        found_sal, sal_data, _, _, _ = find_variable_in_files(
            nc_run_ids[0], nc_runFileNames[0], 'vosaline'
        )
        # Get actual depth values from file
        depth_vals = get_depth_coordinate(nc_run_ids[0], nc_runFileNames[0])

        if found_o2 and found_temp and found_sal and depth_vals is not None:
            aou_result = aouData(
                o2_data, temp_data, sal_data,
                lons, lats, landMask, volMask, missing_val,
                [-180, 180], [-90, 90], depth_vals, target_depth_m=300.0
            )
            log.info(f"AOU computed: annual mean = {aou_result[0]:.2f} µmol/L")
        else:
            missing = []
            if not found_o2:
                missing.append('O2')
            if not found_temp:
                missing.append('votemper')
            if not found_sal:
                missing.append('vosaline')
            if depth_vals is None:
                missing.append('depth coordinate')
            log.warning(f"Cannot compute AOU: missing {', '.join(missing)}")
    except Exception as e:
        log.warning(f"Error computing AOU: {e}")

    # 3.6 Compute RLS (remineralization length scale) from EXP and MLD
    log.info("Computing RLS...")
    rls_result = None
    try:
        from analyser_io import find_variable_in_files, get_depth_coordinate
        # Find EXP and mldr10_1 in the loaded files
        found_exp, exp_data, lats, lons, _ = find_variable_in_files(
            nc_run_ids[0], nc_runFileNames[0], 'EXP'
        )
        found_mld, mld_data, _, _, _ = find_variable_in_files(
            nc_run_ids[0], nc_runFileNames[0], 'mldr10_1'
        )
        # Get actual depth values from file
        depth_vals = get_depth_coordinate(nc_run_ids[0], nc_runFileNames[0])

        if found_exp and found_mld and depth_vals is not None:
            # Time-average the data
            exp_mean = np.nanmean(exp_data, axis=0)
            mld_mean = np.nanmean(mld_data, axis=0)
            # Filter missing values in MLD
            mld_mean[mld_mean > missing_val / 10] = np.nan

            # Calculate RLS using actual depth values from file
            rls = calculate_rls(exp_mean, depth_vals, mld_mean, landMask, missing_val)

            # Compute global mean
            rls_result = np.nanmean(rls)
            log.info(f"RLS computed: annual mean = {rls_result:.2f} m")
        else:
            missing = []
            if not found_exp:
                missing.append('EXP')
            if not found_mld:
                missing.append('mldr10_1')
            if depth_vals is None:
                missing.append('depth coordinate')
            log.warning(f"Cannot compute RLS: missing {', '.join(missing)}")
    except Exception as e:
        log.warning(f"Error computing RLS: {e}")

    # 4. Write this year's results to CSV immediately
    log.info(f"Writing results for year {year}...")
    writer.write_annual_csv_streaming("analyser.sur.annual.csv", config.surface_vars, year)
    writer.write_annual_csv_streaming("analyser.lev.annual.csv", config.level_vars, year)
    writer.write_annual_csv_streaming("analyser.vol.annual.csv", config.volume_vars, year)
    writer.write_annual_csv_streaming("analyser.int.annual.csv", config.integration_vars, year)
    writer.write_annual_csv_streaming("analyser.ave.annual.csv", config.average_vars, year)

    # Write AOU and RLS to average file (append columns)
    aou_file = Path("analyser.ave.annual.csv")
    if aou_result is not None or rls_result is not None:
        # Read existing content and add columns
        if aou_file.exists():
            with open(aou_file, 'r') as f:
                lines = f.readlines()

            # First year: add headers
            if year == year_from:
                if lines:
                    header_suffix = ''
                    if aou_result is not None:
                        header_suffix += ',AOU'
                    if rls_result is not None:
                        header_suffix += ',RLS'
                    lines[0] = lines[0].rstrip('\n') + header_suffix + '\n'

            # Add values to the last line (current year)
            if lines:
                value_suffix = ''
                if aou_result is not None:
                    value_suffix += f',{aou_result[0]:.4e}'
                if rls_result is not None:
                    value_suffix += f',{rls_result:.2f}'
                lines[-1] = lines[-1].rstrip('\n') + value_suffix + '\n'

            with open(aou_file, 'w') as f:
                f.writelines(lines)

    # 5. Close NetCDF files to free memory
    for file_list in nc_run_ids:
        for nc_file in file_list:
            try:
                nc_file.close()
            except:
                pass

    log.info(f"Completed year {year}")

failed_files = all_failed_files

log.info("")
log.info("="*60)
log.info("All years processed")
log.info("="*60)

# ---------- 8 FINAL REPORT ----------
if len(failed_files) > 0:
    log.warning("=" * 60)
    log.warning(f"WARNING: {len(failed_files)} file(s) failed to load due to corruption/errors:")
    for file_path, error_msg in failed_files:
        log.warning(f"  - {file_path}")
        log.warning(f"    {error_msg}")
    log.warning("=" * 60)
    log.warning("Processing completed with errors. Output files contain data from successfully loaded files only.")
else:
    log.info("Processing complete!")
