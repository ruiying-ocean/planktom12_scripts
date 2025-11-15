#!/usr/bin/env python3
"""
Migration tool to convert legacy breakdown_parms text format to TOML format.

Usage:
    python breakdown_migration.py <input_file> <output_file>
    python breakdown_migration.py breakdown_parms breakdown_config.toml
"""

import sys
import tomli_w
from breakdown_config import parse_text_config

# Region number to descriptive name mapping
# Variable name to simple descriptive name mapping
VARIABLE_NAMES = {
    # Carbon flux
    "Cflx": "carbon_flux",

    # Export production
    "EXP": "export",
    "ExpARA": "export_aragonite",
    "ExpCO3": "export_calcite",
    "sinksil": "silicate_sink",

    # Primary production
    "PPT": "primary_prod",
    "proara": "prod_aragonite",
    "prococ": "prod_coccolith",
    "probsi": "prod_biogenic_silica",

    # Grazing
    "GRAGEL": "graze_gelatinous",
    "GRACRU": "graze_crustacean",
    "GRAMES": "graze_mesozoo",
    "GRAPRO": "graze_protist",
    "GRAPTE": "graze_pteropod",

    # Plankton functional groups
    "BAC": "bacteria",
    "COC": "coccolithophores",
    "DIA": "diatoms",
    "FIX": "diazotrophs",
    "GEL": "gelatinous",
    "CRU": "crustacean",
    "MES": "mesozooplankton",
    "MIX": "mixotrophs",
    "PHA": "phaeocystis",
    "PIC": "picophyto",
    "PRO": "protists",
    "PTE": "pteropods",

    # Nutrients and chemistry
    "TChl": "chlorophyll",
    "PO4": "phosphate",
    "NO3": "nitrate",
    "Fer": "iron",
    "Si": "silicate",
    "O2": "oxygen",
    "pCO2": "pco2",
    "Alkalini": "alkalinity",

    # Physical properties
    "tos": "sst",  # sea surface temperature
    "sos": "sss",  # sea surface salinity
    "mldr10_1": "mld",  # mixed layer depth
}

REGION_NAMES = {
    -1: "global",
    0: "arctic",
    1: "a1",
    2: "p1",
    3: "a2_p2",
    4: "a3_p3_i3",
    5: "a4_p4_i4",
    6: "a5",
    7: "p5",
    8: "i5",
    9: "open_ocean_0",
    10: "open_ocean_1",
    11: "open_ocean_2",
    12: "open_ocean_3",
    13: "open_ocean_4",
    14: "atlantic_0",
    15: "atlantic_1",
    16: "atlantic_2",
    17: "atlantic_3",
    18: "atlantic_4",
    19: "atlantic_5",
    20: "pacific_0",
    21: "pacific_1",
    22: "pacific_2",
    23: "pacific_3",
    24: "pacific_4",
    25: "pacific_5",
    26: "indian_0",
    27: "indian_1",
    28: "arctic_0",
    29: "arctic_1",
    30: "arctic_2",
    31: "arctic_3",
    32: "southern_0",
    33: "southern_1",
    34: "southern_2",
    35: "seamask",
    36: "coast",
    37: "atlantic",
    38: "pacific",
    39: "indian",
    40: "southern_ocean",
    41: "arctic",
    42: "pacific_north",
    43: "pacific_2",
    44: "pacific_3",
    45: "pacific_4",
    46: "pacific_5",
    47: "atlantic_1",
    48: "atlantic_2",
    49: "atlantic_3",
    50: "atlantic_4",
    51: "atlantic_5",
    52: "indian_3",
    53: "indian_4",
    54: "indian_5",
}


def lat_range_to_name(lat_min, lat_max):
    """Convert latitude range to descriptive oceanographic name."""
    if lat_min == -90 and lat_max == 90:
        return "global"
    elif lat_min >= 45:
        return "subpolar_north"
    elif lat_min >= 15 and lat_max <= 45:
        return "subtrop_north"
    elif lat_min >= -15 and lat_max <= 15:
        return "tropical"
    elif lat_min >= -45 and lat_max <= -15:
        return "subtrop_south"
    elif lat_max <= -45:
        return "subpolar_south"
    else:
        # For custom ranges, use lat coordinates
        return f"{int(lat_min)}_{int(lat_max)}"


def improve_key_name(key, lat_min, lat_max, region):
    """
    Convert legacy key names like 'Reg_1' to descriptive names.

    Args:
        key: Original key (e.g., "total", "Reg_1", "global")
        lat_min: Minimum latitude
        lat_max: Maximum latitude
        region: Region number

    Returns:
        Improved key name
    """
    # If it's a "Reg_X" key, convert to oceanographic name
    if key.startswith("Reg_"):
        lat_region = lat_range_to_name(lat_min, lat_max)
        return lat_region
    # If region is specified, use region name
    elif region != -1:
        return REGION_NAMES.get(region, f"reg_{region}")
    # Otherwise keep the original key
    else:
        return key


def generate_column_name(variable, region, depth_from=None, depth_to=None, level=None, key="", lat_min=-90, lat_max=90):
    """
    Generate descriptive column name using underscores.

    Args:
        variable: Variable name (e.g., "BAC", "Chl", "Cflx")
        region: Region number or -1 for global
        depth_from: Starting depth (for integration/average)
        depth_to: Ending depth (for integration/average)
        level: Depth level (for level variables)
        key: Key from config (for additional context)
        lat_min: Minimum latitude
        lat_max: Maximum latitude

    Returns:
        Descriptive column name with underscores (e.g., "bacteria_0_15m_global", "carbon_flux_tropical")
    """
    # Use original variable name
    col_name = variable

    # Add depth information if available
    if depth_from is not None and depth_to is not None:
        col_name += f"_{depth_from}_{depth_to}m"
    elif level is not None:
        # Convert level to approximate depth
        # Level 0 ≈ surface, Level 10 ≈ 100m, Level 21 ≈ 1000m
        depth_m = level * 10  # Simplified mapping
        col_name += f"_{depth_m}m"

    # Add region information
    if region != -1:
        # Use region name from mapping
        region_name = REGION_NAMES.get(region, f"reg_{region}")
        col_name += f"_{region_name}"
    else:
        # Use lat/lon range to determine region
        lat_region = lat_range_to_name(lat_min, lat_max)
        if lat_region != "global" or (depth_from is None and level is None):
            col_name += f"_{lat_region}"

    return col_name


def convert_to_toml(text_config_path: str, toml_config_path: str):
    """Convert legacy text format to TOML."""
    print(f"Reading configuration from {text_config_path}...")
    config = parse_text_config(text_config_path)

    toml_data = {
        'files': {
            'basin_mask': config.basin_mask,
            'woa_mask': config.woa_mask,
            'region_mask': config.region_mask,
            'reccap_mask': config.reccap_mask,
            'mesh_mask': config.mesh_mask,
            'ancillary_data': config.ancillary_data,
        },
        'surface': [],
        'level': [],
        'volume': [],
        'integration': [],
        'average': [],
    }

    # Convert surface variables
    print(f"Converting {len(config.surface_vars)} surface variables...")
    for var in config.surface_vars:
        lat_min = float(var.lat_limit[0])
        lat_max = float(var.lat_limit[1])
        column_name = generate_column_name(var.name, var.region, key=var.key, lat_min=lat_min, lat_max=lat_max)
        improved_key = improve_key_name(var.key, lat_min, lat_max, var.region)
        toml_data['surface'].append({
            'variable': var.name,
            'units': var.units,
            'lon_range': [float(var.lon_limit[0]), float(var.lon_limit[1])],
            'lat_range': [lat_min, lat_max],
            'key': improved_key,
            'column_name': column_name,
        })
        if var.region != -1:
            toml_data['surface'][-1]['region'] = var.region

    # Convert level variables
    print(f"Converting {len(config.level_vars)} level variables...")
    for var in config.level_vars:
        lat_min = float(var.lat_limit[0])
        lat_max = float(var.lat_limit[1])
        column_name = generate_column_name(var.name, var.region, level=var.level, key=var.key, lat_min=lat_min, lat_max=lat_max)
        improved_key = improve_key_name(var.key, lat_min, lat_max, var.region)
        toml_data['level'].append({
            'variable': var.name,
            'level': var.level,
            'units': var.units,
            'lon_range': [float(var.lon_limit[0]), float(var.lon_limit[1])],
            'lat_range': [lat_min, lat_max],
            'key': improved_key,
            'column_name': column_name,
        })
        if var.region != -1:
            toml_data['level'][-1]['region'] = var.region

    # Convert volume variables
    print(f"Converting {len(config.volume_vars)} volume variables...")
    for var in config.volume_vars:
        lat_min = float(var.lat_limit[0])
        lat_max = float(var.lat_limit[1])
        column_name = generate_column_name(var.name, var.region, key=var.key, lat_min=lat_min, lat_max=lat_max)
        improved_key = improve_key_name(var.key, lat_min, lat_max, var.region)
        toml_data['volume'].append({
            'variable': var.name,
            'units': var.units,
            'lon_range': [float(var.lon_limit[0]), float(var.lon_limit[1])],
            'lat_range': [lat_min, lat_max],
            'key': improved_key,
            'column_name': column_name,
        })
        if var.region != -1:
            toml_data['volume'][-1]['region'] = var.region

    # Convert integration variables
    print(f"Converting {len(config.integration_vars)} integration variables...")
    for var in config.integration_vars:
        lat_min = float(var.lat_limit[0])
        lat_max = float(var.lat_limit[1])
        column_name = generate_column_name(
            var.name, var.region,
            depth_from=var.depth_from, depth_to=var.depth_to,
            key=var.key,
            lat_min=lat_min, lat_max=lat_max
        )
        improved_key = improve_key_name(var.key, lat_min, lat_max, var.region)
        toml_data['integration'].append({
            'variable': var.name,
            'depth_from': var.depth_from,
            'depth_to': var.depth_to,
            'units': var.units,
            'lon_range': [float(var.lon_limit[0]), float(var.lon_limit[1])],
            'lat_range': [lat_min, lat_max],
            'key': improved_key,
            'column_name': column_name,
        })
        if var.region != -1:
            toml_data['integration'][-1]['region'] = var.region

    # Convert average variables
    print(f"Converting {len(config.average_vars)} average variables...")
    for var in config.average_vars:
        lat_min = float(var.lat_limit[0])
        lat_max = float(var.lat_limit[1])
        column_name = generate_column_name(
            var.name, var.region,
            depth_from=var.depth_from, depth_to=var.depth_to,
            key=var.key,
            lat_min=lat_min, lat_max=lat_max
        )
        improved_key = improve_key_name(var.key, lat_min, lat_max, var.region)
        toml_data['average'].append({
            'variable': var.name,
            'depth_from': var.depth_from,
            'depth_to': var.depth_to,
            'units': var.units,
            'lon_range': [float(var.lon_limit[0]), float(var.lon_limit[1])],
            'lat_range': [lat_min, lat_max],
            'key': improved_key,
            'column_name': column_name,
        })
        if var.region != -1:
            toml_data['average'][-1]['region'] = var.region

    # Add observations and properties if any
    if config.observations:
        toml_data['observations'] = []
        for obs in config.observations:
            toml_data['observations'].append({
                'dataset': obs.obs_dataset,
                'obs_variable': obs.obs_var,
                'model_variable': obs.model_var,
                'depth_obs': obs.depth_obs,
                'depth_model': obs.depth_model,
                'gam_flag': obs.gam_flag,
                'lon_range': [float(obs.lon_limit[0]), float(obs.lon_limit[1])],
                'lat_range': [float(obs.lat_limit[0]), float(obs.lat_limit[1])],
                'key': obs.key,
            })
            if obs.region != -1:
                toml_data['observations'][-1]['region'] = obs.region

    if config.properties:
        toml_data['properties'] = []
        for prop in config.properties:
            prop_dict = {
                'type': prop.prop_name,
                'depth_from': prop.depth_from,
                'depth_to': prop.depth_to,
                'lon_range': [float(prop.lon_limit[0]), float(prop.lon_limit[1])],
                'lat_range': [float(prop.lat_limit[0]), float(prop.lat_limit[1])],
                'key': prop.key,
            }
            if prop.prop_name == 'Trophic' and isinstance(prop.variables, list):
                prop_dict['variables'] = prop.variables
            else:
                prop_dict['variable'] = prop.variables
            toml_data['properties'].append(prop_dict)

    if config.map_vars:
        toml_data['maps'] = []
        for m in config.map_vars:
            toml_data['maps'].append({
                'variable': m.name,
                'level': m.level,
            })

    # Write TOML file
    print(f"Writing TOML configuration to {toml_config_path}...")
    with open(toml_config_path, 'wb') as f:
        tomli_w.dump(toml_data, f)

    print(f"\n✓ Successfully converted to TOML format!")
    print(f"\nColumn naming convention:")
    print(f"  - Lowercase with underscores: bac_0_15m_global")
    print(f"  - Variable + depth + region: chl_pacific, cflx_southern_ocean")
    print(f"\nNext steps:")
    print(f"  1. Review {toml_config_path} and adjust column names if needed")
    print(f"  2. Update your breakdown.py script to use: python breakdown.py {toml_config_path} <year_from> <year_to>")


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python breakdown_migration.py <input_file> <output_file>")
        print("Example: python breakdown_migration.py breakdown_parms breakdown_config.toml")
        sys.exit(1)

    convert_to_toml(sys.argv[1], sys.argv[2])
