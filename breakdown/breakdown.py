#!/usr/bin/env python
"""
Breakdown - Ocean biogeochemical model output analysis tool (Refactored)

This is a refactored version of breakdown.py that uses modular components
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
from breakdown_config import parse_config_file
from breakdown_io import load_netcdf_files, OutputWriter
from breakdown_processor import process_variables, process_average_variables_special, precompute_region_masks
from breakdown_functions import surfaceData, volumeData, intergrateData, volumeDataAverage, observationData, levelData
from breakdown_functions import bloom, trophic, regrid
from breakdown_observations import observationDatasets
from dateutil import relativedelta
import datetime

# ---------- 1 SETUP AND INITIALIZATION ----------

# Parse command line arguments
parser = argparse.ArgumentParser(
    description='Breakdown - Ocean biogeochemical model output analysis tool',
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog='''
Examples:
  %(prog)s breakdown_config.toml 2000 2010
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

# Validate the parameter file exists
if not parm_file.exists():
    parser.error(f"Parameter file not found: {parm_file}")

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
    datefmt='%m-%d %H:%M',
    filename=f'breakdown.{year_from}_{year_to}.log',
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

# ---------- 2 LOAD OBSERVATION DATA ----------
list_of_observations = observationDatasets()
for l in list_of_observations:
    log.info(f"{l['name']} available from {l['origin']}")

# ---------- 3 DEFINE UNITS AND CONSTANTS ----------
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
obsComparisons = []
properties = []
varMap = []

# Convert config objects to legacy format for parts that still need it
for var in config.surface_vars:
    varSurface.append([var.name, var.units, var.lon_limit, var.lat_limit, var.key, var.region, var.results])

for var in config.level_vars:
    varLevel.append([var.name, var.level, var.units, var.lon_limit, var.lat_limit, var.key, var.region, var.results])

for var in config.volume_vars:
    varVolume.append([var.name, var.units, var.lon_limit, var.lat_limit, var.key, var.region, var.results])

for var in config.integration_vars:
    varInt.append([var.name, var.depth_from, var.depth_to, var.units, var.lon_limit, var.lat_limit, var.key, var.region, var.results])

for var in config.average_vars:
    varTotalAve.append([var.name, var.depth_from, var.depth_to, var.units, var.lon_limit, var.lat_limit, var.key, var.region, var.results])

for obs in config.observations:
    obsComparisons.append([obs.obs_dataset, obs.obs_var, obs.model_var, obs.depth_obs, obs.depth_model, obs.gam_flag, obs.lon_limit, obs.lat_limit, obs.key, obs.region, obs.results])

for prop in config.properties:
    properties.append([prop.prop_name, prop.variables, prop.depth_from, prop.depth_to, prop.lon_limit, prop.lat_limit, prop.key, prop.results])

for map_var in config.map_vars:
    varMap.append([map_var.name, map_var.level])

# ---------- 5 LOAD NETCDF MASK AND GRID FILES ----------
log.info("Loading mask and grid files...")

# Ancillary data
ancil_mask_file = list(Path('.').glob(config.ancillary_data))
nc_ancil_id = Dataset(str(ancil_mask_file[0]), 'r')
vmasked = nc_ancil_id.variables["VOLUME_MASKED"][:, :, :].data

# Mesh mask
mesh_mask_file = list(Path('.').glob(config.mesh_mask))
nc_mesh_id = Dataset(str(mesh_mask_file[0]), 'r')
tmeshmask = nc_mesh_id.variables["tmask"][0, :, :, :].data

# Basin mask
basin_file = list(Path('.').glob(config.basin_mask))
nc_basin_id = Dataset(str(basin_file[0]), 'r')
mask_area = nc_basin_id.variables["AREA"][:].data
mask_vol = nc_basin_id.variables["VOLUME"][:].data
landMask = np.copy(mask_area)
landMask[landMask > 0] = 1
landMask[landMask == 0] = np.nan
volMask = np.copy(mask_vol)
volMask[volMask > 0] = 1
volMask[volMask == 0] = np.nan

# Region masks
reg_file = list(Path('.').glob(config.region_mask))
nc_reg_id = Dataset(str(reg_file[0]), 'r')
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
reccap_file = list(Path('.').glob(config.reccap_mask))
nc_reccap_id = Dataset(str(reccap_file[0]), 'r')
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

# ---------- 6 PRE-COMPUTE REGION MASKS ----------
log.info("Pre-computing region masks for all regions...")
region_mask_cache = precompute_region_masks(landMask, volMask, regions)

# ---------- 7 LOAD MODEL OUTPUT NETCDF FILES ----------
log.info("Loading model output files...")
nc_run_ids, nc_runFileNames, years, failed_files = load_netcdf_files(year_from, year_to)

# ---------- 8 PROCESS VARIABLES (USING UNIFIED PROCESSOR) ----------
log.info("Processing variables...")

# Process surface variables
log.info("Processing surface variables...")
process_variables(
    varSurface, nc_run_ids, nc_runFileNames,
    list_of_units, regions, landMask, volMask,
    mask_area, mask_vol, missing_val,
    surfaceData, 'surface', region_mask_cache
)

# Process level variables
log.info("Processing level variables...")
process_variables(
    varLevel, nc_run_ids, nc_runFileNames,
    list_of_units, regions, landMask, volMask,
    mask_area, mask_vol, missing_val,
    levelData, 'level', region_mask_cache
)

# Process volume variables
log.info("Processing volume variables...")
process_variables(
    varVolume, nc_run_ids, nc_runFileNames,
    list_of_units, regions, landMask, volMask,
    mask_area, mask_vol, missing_val,
    volumeData, 'volume', region_mask_cache
)

# Process integration variables
log.info("Processing integration variables...")
process_variables(
    varInt, nc_run_ids, nc_runFileNames,
    list_of_units, regions, landMask, volMask,
    mask_area, mask_vol, missing_val,
    intergrateData, 'integration', region_mask_cache
)

# Process average variables (special handling for multi-variable sums)
log.info("Processing average variables...")
process_average_variables_special(
    varTotalAve, nc_run_ids, nc_runFileNames,
    list_of_units, regions, landMask, volMask,
    mask_vol, missing_val, region_mask_cache
)

# NOTE: Observation and property processing retained from original code
# These sections are kept as-is for now since they have unique logic

# ----- PROCESS OBSERVATION COMPARISONS -----
log.info("Processing observation comparisons...")
nc_woa_id = -1

for obs in obsComparisons:
    obsName = obs[0]
    obsVar = obs[1]
    varName = obs[2].split('+')
    depthObs = obs[3]
    depthVar = obs[4]
    gamFlag = obs[5]
    reg = obs[9]
    latLim = [-90, 90]
    lonLim = [-180, 180]
    if reg == -1:
        lonLim = [float(obs[6][0]), float(obs[6][1])]
        latLim = [float(obs[7][0]), float(obs[7][1])]

    sourceObs = None
    for l in list_of_observations:
        if obs[0] == l['name']:
            sourceObs = l
    nc_obs_id = Dataset(sourceObs['path'], 'r')

    # Use pre-computed region mask from cache
    regionVolMask = region_mask_cache[f'vol_{reg}']

    found_var = False
    for n in range(len(nc_run_ids)):
        var_data_arr = []
        for i in range(len(nc_run_ids[n])):
            for v in range(len(varName)):
                try:
                    var_data = nc_run_ids[n][i].variables[varName[v]][:].data
                    if len(nc_run_ids[n][i].variables[varName[v]][:].data.shape) == 4:
                        log.info(f"{varName[v]} has 4 dimensions, only taking data from depth {depthVar}")
                        var_data = nc_run_ids[n][i].variables[varName[v]][:, depthVar, :, :].data

                    var_data_arr.append(var_data)
                    val_lats = nc_run_ids[n][i].variables['nav_lat'][:].data
                    val_lons = nc_run_ids[n][i].variables['nav_lon'][:].data
                    found_var = True
                    log.info(f"{varName[v]} found in {nc_runFileNames[n][i]}")
                except KeyError:
                    pass

        if np.array(var_data_arr).shape[0] == 0:
            found_var = False
        else:
            var_data = np.sum(np.array(var_data_arr), axis=0)
            var_data = var_data * sourceObs['factor']

            found_obs = False
            try:
                obs_lat = nc_obs_id.variables[sourceObs['latVar']][:].data
                obs_lon = nc_obs_id.variables[sourceObs['lonVar']][:].data
                obs_time = nc_obs_id.variables[sourceObs['timeVar']][:].data

                t_factor = 1
                if "days" in nc_obs_id.variables[sourceObs['timeVar']].units:
                    t_factor = 60 * 60 * 24

                obs_time = obs_time * t_factor
                first_time = sourceObs['origin'] + datetime.timedelta(0, int(obs_time[0]))

                year = years[n]
                timePoint = (relativedelta.relativedelta(datetime.datetime(year, 1, 1, 0, 0, 0), first_time)).years * 12

                if sourceObs['climatological']:
                    timePoint = 0

                if len(nc_obs_id.variables[obsVar].dimensions) == 3:
                    obs_data = nc_obs_id.variables[obsVar][timePoint:timePoint + 12, :, :].data
                if len(nc_obs_id.variables[obsVar].dimensions) == 4:
                    log.info(f"{obsVar} has 4 dimensions, only taking data from depth {depthObs}")
                    obs_data = nc_obs_id.variables[obsVar][timePoint:timePoint + 12, depthObs, :, :].data

                log.info(f"Timepoint found: {timePoint}")
                found_obs = True
                log.info(f"{obsVar} found in {obsName}")

                if obs_data.shape[0] != 12 or timePoint < 0:
                    found_obs = False
                    log.info(f"{obsVar} data incomplete or model data is pre obs data")
                else:
                    if sourceObs['conversion'] is not None:
                        log.info(f"Conversion File: {sourceObs['conversion']} for {sourceObs['conversionName']}")
                        nc_conv_id = Dataset(sourceObs['conversion'], 'r')

                    if len(nc_conv_id.variables[sourceObs['conversionName']].dimensions) == 3:
                        conv_data = nc_conv_id.variables[sourceObs['conversionName']][timePoint:timePoint + 12, :, :].data
                    if len(nc_conv_id.variables[sourceObs['conversionName']].dimensions) == 4:
                        conv_data = nc_conv_id.variables[sourceObs['conversionName']][timePoint:timePoint + 12, depthObs, :, :].data

                    obs_data = obs_data * conv_data
                    obs_data[obs_data < (-missing_val / 100.)] = missing_val
                    obs_data[obs_data > (missing_val / 100.)] = missing_val
            except KeyError:
                pass

        if found_var and found_obs:
            log.info("Comparing to obs data")
            outputTotal = observationData(obs_data, obs_lon, obs_lat, var_data, val_lons, val_lats,
                                         sourceObs['centered'], regionVolMask[depthVar, :, :], missing_val, gamFlag, lonLim, latLim)
            obs[-1].append(outputTotal)
        else:
            log.info(f"{obs[2]} or {obsVar} not found correctly")
            nullAnnual = -1
            nullMonthly = np.array([-1 for r in range(12)])
            obs[-1].append((nullAnnual, nullMonthly, [[], []]))

# ----- PROCESS EMERGENT PROPERTIES -----
log.info("Processing emergent properties...")

for prop in properties:
    propName = prop[0]

    if propName == "Bloom":
        propVar = prop[1].split('+')
        depthFrom = int(prop[2])
        depthTo = int(prop[3])
        lonLim = [float(prop[4][0]), float(prop[4][1])]
        latLim = [float(prop[5][0]), float(prop[5][1])]

        found_var = False
        for n in range(len(nc_run_ids)):
            var_data_arr = []
            for i in range(len(nc_run_ids[n])):
                for v in range(len(propVar)):
                    try:
                        var_data = nc_run_ids[n][i].variables[propVar[v]][:].data

                        if len(nc_run_ids[n][i].variables[propVar[v]][:].data.shape) == 4:
                            log.info(f"{propVar[v]} has 4 dimensions, summing data for depths {depthFrom} {depthTo}")
                            var_data = nc_run_ids[n][i].variables[propVar[v]][:, depthFrom:depthTo + 1, :, :].data
                            var_data = np.sum(var_data, axis=1)

                        var_data_arr.append(var_data)
                        val_lats = nc_run_ids[n][i].variables['nav_lat'][:].data
                        val_lons = nc_run_ids[n][i].variables['nav_lon'][:].data
                        found_var = True
                        log.info(f"{propVar[v]} found in {nc_runFileNames[n][i]}")
                    except KeyError:
                        pass

            var_data = np.sum(np.array(var_data_arr), axis=0)

            if not found_var:
                log.info(f"{prop[1]} not found")
                prop[-1].append((-1, -1, -1, -1, ["none", "none", "none", "none"], -1))
            else:
                outputProp = bloom(var_data, val_lons, val_lats, missing_val, lonLim, latLim)
                prop[-1].append(outputProp)

    if propName == "Trophic":
        propVarList = [prop[1][0].split('+'), prop[1][1].split('+'), prop[1][2].split('+')]
        depthFrom = int(prop[2])
        depthTo = int(prop[3])
        lonLim = [float(prop[4][0]), float(prop[4][1])]
        latLim = [float(prop[5][0]), float(prop[5][1])]

        all_var_data = [None, None, None]
        found_var = False
        for n in range(len(nc_run_ids)):
            for p in range(3):
                var_data_arr = []
                propVar = propVarList[p]
                for i in range(len(nc_run_ids[n])):
                    try:
                        for v in range(len(propVar)):
                            var_data = nc_run_ids[n][i].variables[propVar[v]][:].data
                            if len(nc_run_ids[n][i].variables[propVar[v]][:].data.shape) == 4:
                                log.info(f"{propVar[v]} has 4 dimensions, summing data for depths {depthFrom} {depthTo}")
                                var_data = nc_run_ids[n][i].variables[propVar[v]][:, depthFrom:depthTo + 1, :, :].data
                                var_data = np.sum(var_data, axis=1)

                            var_data_arr.append(var_data)
                            val_lats = nc_run_ids[n][i].variables['nav_lat'][:].data
                            val_lons = nc_run_ids[n][i].variables['nav_lon'][:].data
                            found_var = True
                            log.info(f"{propVar[v]} found in {nc_runFileNames[n][i]}")
                    except KeyError:
                        pass

                all_var_data[p] = np.sum(np.array(var_data_arr), axis=0)

            if not found_var:
                log.info(f"{prop[1]} none found")
                prop[-1].append((-1, -1, -1, -1, ["none", "none", "none", "none"], -1))
            else:
                outputProp = trophic(all_var_data, val_lons, val_lats, missing_val, lonLim, latLim)
                prop[-1].append(outputProp)

# ----- PROCESS REGRIDDED MAP OUTPUT -----
log.info("Processing map outputs...")

if len(varMap) > 0:
    # Load WOA mask if needed
    woa_file = list(Path('.').glob(config.woa_mask))
    nc_woa_id = Dataset(str(woa_file[0]), 'r')
    woamask = nc_woa_id.variables["mask"][:]

target_lon = np.arange(0.5, 360.5, 1)
target_lat = np.arange(-89.5, 90.5, 1)

for n in range(len(nc_run_ids)):
    for var in varMap:
        varName = var[0]
        level = var[1]
        if level != 'all':
            lev = int(level)

        outputFileName = f"WOA_{varName}_{years[n]}_{level}.nc"
        nc_out_id = Dataset(outputFileName, 'w', format='NETCDF4_CLASSIC')
        nc_out_id.createDimension("lon", 360)
        nc_out_id.createDimension("lat", 180)

        times = nc_run_ids[n][0].variables['time_counter'][:].data
        nc_out_id.createDimension("time", None)
        if level == 'all':
            depths = nc_run_ids[n][0].variables['deptht'][:].data
            nc_out_id.createDimension("deptht", len(depths))
        else:
            depths = nc_run_ids[n][0].variables['deptht'][:].data
            nc_out_id.createDimension("deptht", 1)

        nc_out_id.createVariable("lon", 'f', ('lon'))
        nc_out_id.createVariable("lat", 'f', ('lat'))
        nc_out_id.createVariable("deptht", 'f', ('deptht'))
        nc_out_id.createVariable("time", 'f', ('time'))

        found = False

        for i in range(len(nc_run_ids[n])):
            try:
                data = nc_run_ids[n][i].variables[varName][:].data
                if len(data.shape) == 4:
                    log.info(f"{varName} is volume data")
                    nc_v = nc_out_id.createVariable(varName, 'f', ('time', 'deptht', 'lat', 'lon'), fill_value=missing_val)
                    if level == 'all':
                        nc_out_id.variables['deptht'][:] = depths
                    else:
                        nc_out_id.variables['deptht'][:] = depths[lev]
                else:
                    nc_v = nc_out_id.createVariable(varName, 'f', ('time', 'lat', 'lon'), fill_value=missing_val)

                for ncattr in nc_run_ids[n][i].variables[varName].ncattrs():
                    if ncattr != '_FillValue':
                        nc_v.setncattr(ncattr, nc_run_ids[n][i].variables[varName].getncattr(ncattr))
                nc_v.setncattr("missing_value", np.array(missing_val, 'f'))

                val_lats = nc_run_ids[n][i].variables['nav_lat'][:].data
                val_lons = nc_run_ids[n][i].variables['nav_lon'][:].data
                nc_out_id.variables['lon'][:] = target_lon
                nc_out_id.variables['lat'][:] = target_lat
                found = True
                log.info(f"{varName} found in {nc_runFileNames[n][i]}")
            except KeyError:
                pass

        if found:
            woaind = np.where(woamask == 0)
            if len(data.shape) == 3:
                regriddedData = regrid(data, val_lons, val_lats, target_lon, target_lat, tmeshmask[0, :, :], missing_val)
                for t in range(data.shape[0]):
                    regriddedData_slice = regriddedData[t, :, :]
                    regriddedData_slice[woaind] = missing_val
                    regriddedData_slice[np.isnan(regriddedData_slice)] = missing_val
                    nc_out_id.variables[varName][t, :, :] = regriddedData_slice
                    nc_out_id.variables['time'][t] = times[t]

            if len(data.shape) == 4:
                if level == 'all':
                    regriddedData = np.zeros((data.shape[0], data.shape[1], 180, 360))
                    for z in range(data.shape[1]):
                        regriddedData[:, z, :, :] = regrid(data[:, z, :, :], val_lons, val_lats, target_lon, target_lat, tmeshmask[z, :, :], missing_val)
                else:
                    regriddedData = np.zeros((data.shape[0], 1, 180, 360))
                    regriddedData[:, 0, :, :] = regrid(data[:, lev, :, :], val_lons, val_lats, target_lon, target_lat, tmeshmask[lev, :, :], missing_val)

                for t in range(regriddedData.shape[0]):
                    for z in range(regriddedData.shape[1]):
                        regriddedData_slice = regriddedData[t, z, :, :]
                        if level == 'all':
                            meshind = np.where(vmasked[z, :, :] == 0)
                        else:
                            meshind = np.where(vmasked[lev, :, :] == 0)

                        regriddedData_slice[meshind] = missing_val
                        regriddedData_slice[np.isnan(regriddedData_slice)] = missing_val
                        nc_out_id.variables[varName][t, z, :, :] = regriddedData[t, z, :, :]

                    nc_out_id.variables['time'][t] = times[t]

        nc_out_id.close()

# ---------- 8 WRITE OUTPUT FILES (USING UNIFIED WRITER) ----------
log.info("Writing output files...")

writer = OutputWriter()

# Configuration: Choose output format and what to include
# Options: 'csv' or 'tsv' (tab-separated)
OUTPUT_FORMAT = 'csv'
INCLUDE_UNITS_IN_HEADERS = False  # Set to True to include units like "BAC (Conc->PgCarbon)"
INCLUDE_KEYS_IN_HEADERS = False   # Set to True to include keys like "BAC [global]"

# Annual outputs - CSV format (clean, single header row)
if OUTPUT_FORMAT == 'csv':
    log.info("Writing CSV format outputs...")
    writer.write_annual_csv("breakdown.sur.annual.csv", config.surface_vars, year_from, year_to, 0, INCLUDE_UNITS_IN_HEADERS, INCLUDE_KEYS_IN_HEADERS)
    writer.write_annual_csv("breakdown.lev.annual.csv", config.level_vars, year_from, year_to, 0, INCLUDE_UNITS_IN_HEADERS, INCLUDE_KEYS_IN_HEADERS)
    writer.write_annual_csv("breakdown.vol.annual.csv", config.volume_vars, year_from, year_to, 0, INCLUDE_UNITS_IN_HEADERS, INCLUDE_KEYS_IN_HEADERS)
    writer.write_annual_csv("breakdown.int.annual.csv", config.integration_vars, year_from, year_to, 0, INCLUDE_UNITS_IN_HEADERS, INCLUDE_KEYS_IN_HEADERS)
    writer.write_annual_csv("breakdown.ave.annual.csv", config.average_vars, year_from, year_to, 0, INCLUDE_UNITS_IN_HEADERS, INCLUDE_KEYS_IN_HEADERS)
else:
    # Original TSV format (backward compatible - 3 header rows)
    log.info("Writing TSV format outputs...")
    writer.write_annual_file("breakdown.sur.annual.dat", varSurface, year_from, year_to, 0, 1, -3)
    writer.write_annual_file("breakdown.lev.annual.dat", varLevel, year_from, year_to, 0, 2, -3)
    writer.write_annual_file("breakdown.vol.annual.dat", varVolume, year_from, year_to, 0, 1, -3)
    writer.write_annual_file("breakdown.int.annual.dat", varInt, year_from, year_to, 0, 3, -3)
    writer.write_annual_file("breakdown.ave.annual.dat", varTotalAve, year_from, year_to, 0, 3, -3)

# Observations (commented out - uncomment if needed)
# writer.write_observation_file("breakdown.obs.annual.dat", obsComparisons, year_from, year_to)

# Spread outputs (commented out - uncomment if needed)
# writer.write_spread_file("breakdown.sur.spread.dat", varSurface, year_from, year_to, 0, 1, -3)
# writer.write_spread_file("breakdown.lev.spread.dat", varLevel, year_from, year_to, 0, 2, -3)
# writer.write_spread_file("breakdown.vol.spread.dat", varVolume, year_from, year_to, 0, 1, -3)
# writer.write_spread_file("breakdown.int.spread.dat", varInt, year_from, year_to, 0, 3, -3)
# writer.write_spread_file("breakdown.ave.spread.dat", varTotalAve, year_from, year_to, 0, 3, -3)

# Monthly outputs (commented out - uncomment if needed)
# writer.write_monthly_file("breakdown.sur.monthly.dat", varSurface, year_from, year_to, 0, 1, -3)
# writer.write_monthly_file("breakdown.ave.monthly.dat", varTotalAve, year_from, year_to, 0, 3, -3)

# Property outputs
for prop in properties:
    if hasattr(prop, 'key'):
        filename = f"breakdown.{prop.prop_name}.{prop.key}.dat"
    else:
        filename = f"breakdown.{prop[0]}.{prop[-2]}.dat"
    writer.write_property_file(filename, prop, year_from, year_to)

# Report any files that failed to load
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
