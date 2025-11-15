#!/usr/bin/env python
"""
Configuration parsing module for breakdown system.

This module handles parsing of the breakdown_config.toml file and provides
structured configuration objects. Also supports legacy text-based format.
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional
import logging

try:
    import tomli
except ImportError:
    tomli = None

log = logging.getLogger("Config")


# ---------- DATA STRUCTURES ----------

@dataclass
class VariableConfig:
    """Base configuration for a variable to process."""
    name: str
    units: str
    key: str  # Key for output identification
    region: int  # -1 means use lon/lat limits instead
    lon_limit: Tuple[str, str]
    lat_limit: Tuple[str, str]
    results: List  # Will store processing results
    column_name: str = ""  # Custom column name for CSV output


@dataclass
class SurfaceVariable(VariableConfig):
    """Configuration for surface variable processing."""
    pass


@dataclass
class LevelVariable(VariableConfig):
    """Configuration for level-specific variable processing."""
    level: int = 0


@dataclass
class VolumeVariable(VariableConfig):
    """Configuration for volume-integrated variable processing."""
    pass


@dataclass
class IntegrationVariable(VariableConfig):
    """Configuration for depth-integrated variable processing."""
    depth_from: int = 0
    depth_to: int = 0


@dataclass
class AverageVariable(VariableConfig):
    """Configuration for depth-averaged variable processing."""
    depth_from: int = 0
    depth_to: int = 0


@dataclass
class ObservationComparison:
    """Configuration for observation comparison."""
    obs_dataset: str
    obs_var: str
    model_var: str  # Can be comma-separated for summing
    depth_obs: int
    depth_model: int
    gam_flag: bool
    lon_limit: Tuple[str, str]
    lat_limit: Tuple[str, str]
    key: str
    region: int
    results: List


@dataclass
class Property:
    """Configuration for emergent property calculation."""
    prop_name: str  # "Bloom" or "Trophic"
    variables: any  # String or list depending on property type
    depth_from: int
    depth_to: int
    lon_limit: Tuple[str, str]
    lat_limit: Tuple[str, str]
    key: str
    results: List


@dataclass
class MapVariable:
    """Configuration for map output generation."""
    name: str
    level: str  # Can be 'all' or a specific level number


@dataclass
class BreakdownConfig:
    """Complete configuration for breakdown processing."""
    basin_mask: str = ""
    woa_mask: str = ""
    region_mask: str = ""
    reccap_mask: str = ""
    mesh_mask: str = ""
    ancillary_data: str = ""

    surface_vars: List[SurfaceVariable] = None
    level_vars: List[LevelVariable] = None
    volume_vars: List[VolumeVariable] = None
    integration_vars: List[IntegrationVariable] = None
    average_vars: List[AverageVariable] = None
    observations: List[ObservationComparison] = None
    properties: List[Property] = None
    map_vars: List[MapVariable] = None

    def __post_init__(self):
        """Initialize empty lists if None."""
        if self.surface_vars is None:
            self.surface_vars = []
        if self.level_vars is None:
            self.level_vars = []
        if self.volume_vars is None:
            self.volume_vars = []
        if self.integration_vars is None:
            self.integration_vars = []
        if self.average_vars is None:
            self.average_vars = []
        if self.observations is None:
            self.observations = []
        if self.properties is None:
            self.properties = []
        if self.map_vars is None:
            self.map_vars = []


# ---------- PARSING FUNCTIONS ----------

def parse_toml_config(file_path: str) -> BreakdownConfig:
    """
    Parse TOML configuration file.

    Args:
        file_path: Path to the .toml configuration file

    Returns:
        BreakdownConfig object with all parsed configuration
    """
    if tomli is None:
        raise ImportError("tomli library is required for TOML configuration. Install with: pip install tomli")

    config = BreakdownConfig()

    with open(file_path, 'rb') as f:
        data = tomli.load(f)

    # Parse file paths
    if 'files' in data:
        config.basin_mask = data['files'].get('basin_mask', '')
        config.woa_mask = data['files'].get('woa_mask', '')
        config.region_mask = data['files'].get('region_mask', '')
        config.reccap_mask = data['files'].get('reccap_mask', '')
        config.mesh_mask = data['files'].get('mesh_mask', '')
        config.ancillary_data = data['files'].get('ancillary_data', '')

    # Parse surface variables
    for surf in data.get('surface', []):
        config.surface_vars.append(SurfaceVariable(
            name=surf['variable'],
            units=surf['units'],
            key=surf.get('key', ''),
            region=surf.get('region', -1),
            lon_limit=tuple(map(str, surf.get('lon_range', ['-180', '180']))),
            lat_limit=tuple(map(str, surf.get('lat_range', ['-90', '90']))),
            results=[],
            column_name=surf.get('column_name', '')
        ))

    # Parse level variables
    for lev in data.get('level', []):
        config.level_vars.append(LevelVariable(
            name=lev['variable'],
            units=lev['units'],
            key=lev.get('key', ''),
            region=lev.get('region', -1),
            lon_limit=tuple(map(str, lev.get('lon_range', ['-180', '180']))),
            lat_limit=tuple(map(str, lev.get('lat_range', ['-90', '90']))),
            results=[],
            column_name=lev.get('column_name', ''),
            level=lev['level']
        ))

    # Parse volume variables
    for vol in data.get('volume', []):
        config.volume_vars.append(VolumeVariable(
            name=vol['variable'],
            units=vol['units'],
            key=vol.get('key', ''),
            region=vol.get('region', -1),
            lon_limit=tuple(map(str, vol.get('lon_range', ['-180', '180']))),
            lat_limit=tuple(map(str, vol.get('lat_range', ['-90', '90']))),
            results=[],
            column_name=vol.get('column_name', '')
        ))

    # Parse integration variables
    for integ in data.get('integration', []):
        config.integration_vars.append(IntegrationVariable(
            name=integ['variable'],
            units=integ['units'],
            key=integ.get('key', ''),
            region=integ.get('region', -1),
            lon_limit=tuple(map(str, integ.get('lon_range', ['-180', '180']))),
            lat_limit=tuple(map(str, integ.get('lat_range', ['-90', '90']))),
            results=[],
            column_name=integ.get('column_name', ''),
            depth_from=integ['depth_from'],
            depth_to=integ['depth_to']
        ))

    # Parse average variables
    for avg in data.get('average', []):
        config.average_vars.append(AverageVariable(
            name=avg['variable'],
            units=avg['units'],
            key=avg.get('key', ''),
            region=avg.get('region', -1),
            lon_limit=tuple(map(str, avg.get('lon_range', ['-180', '180']))),
            lat_limit=tuple(map(str, avg.get('lat_range', ['-90', '90']))),
            results=[],
            column_name=avg.get('column_name', ''),
            depth_from=avg['depth_from'],
            depth_to=avg['depth_to']
        ))

    # Parse observation comparisons (if any)
    for obs in data.get('observations', []):
        config.observations.append(ObservationComparison(
            obs_dataset=obs['dataset'],
            obs_var=obs['obs_variable'],
            model_var=obs['model_variable'],
            depth_obs=obs['depth_obs'],
            depth_model=obs['depth_model'],
            gam_flag=obs.get('gam_flag', False),
            lon_limit=tuple(map(str, obs.get('lon_range', ['-180', '180']))),
            lat_limit=tuple(map(str, obs.get('lat_range', ['-90', '90']))),
            key=obs.get('key', ''),
            region=obs.get('region', -1),
            results=[]
        ))

    # Parse properties (if any)
    for prop in data.get('properties', []):
        prop_type = prop['type']
        if prop_type == 'Bloom':
            variables = prop['variable']
        elif prop_type == 'Trophic':
            variables = prop['variables']
        else:
            variables = prop.get('variable', '')

        config.properties.append(Property(
            prop_name=prop_type,
            variables=variables,
            depth_from=prop['depth_from'],
            depth_to=prop['depth_to'],
            lon_limit=tuple(map(str, prop.get('lon_range', ['-180', '180']))),
            lat_limit=tuple(map(str, prop.get('lat_range', ['-90', '90']))),
            key=prop.get('key', ''),
            results=[]
        ))

    # Parse map variables (if any)
    for m in data.get('maps', []):
        config.map_vars.append(MapVariable(
            name=m['variable'],
            level=str(m.get('level', 'all'))
        ))

    log.info(f"Parsed {len(config.surface_vars)} surface variables from TOML")
    log.info(f"Parsed {len(config.level_vars)} level variables from TOML")
    log.info(f"Parsed {len(config.volume_vars)} volume variables from TOML")
    log.info(f"Parsed {len(config.integration_vars)} integration variables from TOML")
    log.info(f"Parsed {len(config.average_vars)} average variables from TOML")
    log.info(f"Parsed {len(config.observations)} observation comparisons from TOML")
    log.info(f"Parsed {len(config.properties)} properties from TOML")
    log.info(f"Parsed {len(config.map_vars)} map variables from TOML")

    return config


def parse_text_config(file_path: str) -> BreakdownConfig:
    """
    Parse legacy text-based configuration file (deprecated format).

    Args:
        file_path: Path to the legacy text configuration file

    Returns:
        BreakdownConfig object with all parsed configuration
    """
    config = BreakdownConfig()

    with open(file_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()

        # Skip empty lines and comments
        if len(line) == 0 or line[0] == '#':
            continue

        # Split on colon
        parts = line.split(':', 1)
        if len(parts) != 2:
            continue

        param_type = parts[0].strip()
        value = parts[1].strip()

        # Parse based on parameter type
        if param_type == 'BasinMask':
            config.basin_mask = value
        elif param_type == 'WOAMask':
            config.woa_mask = value
        elif param_type == 'RegionMask':
            config.region_mask = value
        elif param_type == 'RECCAPmask':
            config.reccap_mask = value
        elif param_type == 'Meshmask':
            config.mesh_mask = value
        elif param_type == 'AncillaryData':
            config.ancillary_data = value
        elif param_type == 'Surface':
            config.surface_vars.append(_parse_surface(value))
        elif param_type == 'Level':
            config.level_vars.append(_parse_level(value))
        elif param_type == 'Volume':
            config.volume_vars.append(_parse_volume(value))
        elif param_type == 'Integration':
            config.integration_vars.append(_parse_integration(value))
        elif param_type == 'Avg':
            config.average_vars.append(_parse_average(value))
        elif param_type == 'Observations':
            config.observations.append(_parse_observation(value))
        elif param_type == 'Property':
            config.properties.append(_parse_property(value))
        elif param_type == 'Map':
            config.map_vars.append(_parse_map(value))

    log.info(f"Parsed {len(config.surface_vars)} surface variables")
    log.info(f"Parsed {len(config.level_vars)} level variables")
    log.info(f"Parsed {len(config.volume_vars)} volume variables")
    log.info(f"Parsed {len(config.integration_vars)} integration variables")
    log.info(f"Parsed {len(config.average_vars)} average variables")
    log.info(f"Parsed {len(config.observations)} observation comparisons")
    log.info(f"Parsed {len(config.properties)} properties")
    log.info(f"Parsed {len(config.map_vars)} map variables")

    return config


def parse_config_file(file_path: str) -> BreakdownConfig:
    """
    Parse configuration file (TOML format).

    Args:
        file_path: Path to the .toml configuration file

    Returns:
        BreakdownConfig object with all parsed configuration
    """
    # Since we're fully migrating to TOML, just call the TOML parser
    return parse_toml_config(file_path)


def _parse_surface(value: str) -> SurfaceVariable:
    """Parse Surface parameter line."""
    parts = value.split(',')
    variable = parts[0]
    units = parts[1]

    # Check if using region or lon/lat limits
    if parts[2] == 'Region':
        region = int(parts[3])
        lon_limit = ('-90', '90')
        lat_limit = ('-180', '180')
        key = parts[4]
    else:
        region = -1
        lon_limit = tuple(parts[2].split(';'))
        lat_limit = tuple(parts[3].split(';'))
        key = parts[4]

    log.info(f"Surface: {variable} {units}")

    return SurfaceVariable(
        name=variable,
        units=units,
        key=key,
        region=region,
        lon_limit=lon_limit,
        lat_limit=lat_limit,
        results=[]
    )


def _parse_level(value: str) -> LevelVariable:
    """Parse Level parameter line."""
    parts = value.split(',')
    variable = parts[0]
    level = int(parts[1])
    units = parts[2]

    # Check if using region or lon/lat limits
    if parts[3] == 'Region':
        region = int(parts[4])
        lon_limit = ('-90', '90')
        lat_limit = ('-180', '180')
        key = parts[5]
    else:
        region = -1
        lon_limit = tuple(parts[3].split(';'))
        lat_limit = tuple(parts[4].split(';'))
        key = parts[5]

    log.info(f"Level: {variable} {units}")

    return LevelVariable(
        name=variable,
        units=units,
        key=key,
        region=region,
        lon_limit=lon_limit,
        lat_limit=lat_limit,
        level=level,
        results=[]
    )


def _parse_volume(value: str) -> VolumeVariable:
    """Parse Volume parameter line."""
    parts = value.split(',')
    variable = parts[0]
    units = parts[1]

    # Check if using region or lon/lat limits
    if parts[2] == 'Region':
        region = int(parts[3])
        lon_limit = ('-90', '90')
        lat_limit = ('-180', '180')
        key = parts[4]
    else:
        region = -1
        lon_limit = tuple(parts[2].split(';'))
        lat_limit = tuple(parts[3].split(';'))
        key = parts[4]

    log.info(f"Volume: {variable} {units}")

    return VolumeVariable(
        name=variable,
        units=units,
        key=key,
        region=region,
        lon_limit=lon_limit,
        lat_limit=lat_limit,
        results=[]
    )


def _parse_integration(value: str) -> IntegrationVariable:
    """Parse Integration parameter line."""
    parts = value.split(',')
    variable = parts[0]
    depth_from = int(parts[1])
    depth_to = int(parts[2])
    units = parts[3]

    # Check if using region or lon/lat limits
    if parts[4] == 'Region':
        region = int(parts[5])
        lon_limit = ('-90', '90')
        lat_limit = ('-180', '180')
        key = parts[6]
    else:
        region = -1
        lon_limit = tuple(parts[4].split(';'))
        lat_limit = tuple(parts[5].split(';'))
        key = parts[6]

    log.info(f"Integration: {variable} {depth_from} {depth_to} {units}")

    return IntegrationVariable(
        name=variable,
        units=units,
        key=key,
        region=region,
        lon_limit=lon_limit,
        lat_limit=lat_limit,
        depth_from=depth_from,
        depth_to=depth_to,
        results=[]
    )


def _parse_average(value: str) -> AverageVariable:
    """Parse Avg parameter line."""
    parts = value.split(',')
    variables = parts[0]
    depth_from = int(parts[1])
    depth_to = int(parts[2])
    units = parts[3]

    # Check if using region or lon/lat limits
    if parts[4] == 'Region':
        region = int(parts[5])
        lon_limit = ('-90', '90')
        lat_limit = ('-180', '180')
        key = parts[6]
    else:
        region = -1
        lon_limit = tuple(parts[4].split(';'))
        lat_limit = tuple(parts[5].split(';'))
        key = parts[6]

    log.info(f"Avg: {variables} {units}")

    return AverageVariable(
        name=variables,
        units=units,
        key=key,
        region=region,
        lon_limit=lon_limit,
        lat_limit=lat_limit,
        depth_from=depth_from,
        depth_to=depth_to,
        results=[]
    )


def _parse_observation(value: str) -> ObservationComparison:
    """Parse Observations parameter line."""
    parts = value.split(',')
    obs_data = parts[0]
    obs_var = parts[1]
    var = parts[2]
    depth_obs = int(parts[3])
    depth_var = int(parts[4])
    gam_flag = (parts[5] == 'T')

    # Check if using region or lon/lat limits
    if parts[6] == 'Region':
        region = int(parts[7])
        lon_limit = ('-90', '90')
        lat_limit = ('-180', '180')
        key = parts[8]
    else:
        region = -1
        lon_limit = tuple(parts[6].split(';'))
        lat_limit = tuple(parts[7].split(';'))
        key = parts[8]

    log.info(f"Observations: {obs_data} {var}")

    return ObservationComparison(
        obs_dataset=obs_data,
        obs_var=obs_var,
        model_var=var,
        depth_obs=depth_obs,
        depth_model=depth_var,
        gam_flag=gam_flag,
        lon_limit=lon_limit,
        lat_limit=lat_limit,
        key=key,
        region=region,
        results=[]
    )


def _parse_property(value: str) -> Property:
    """Parse Property parameter line."""
    parts = value.split(',')
    prop_name = parts[0]

    if prop_name == "Bloom":
        prop_var = parts[1]
        depth_from = int(parts[2])
        depth_to = int(parts[3])
        lon_limit = tuple(parts[4].split(';'))
        lat_limit = tuple(parts[5].split(';'))
        key = parts[6]

        log.info(f"Property: {prop_name} from depth level {depth_from} to {depth_to}")

        return Property(
            prop_name=prop_name,
            variables=prop_var,
            depth_from=depth_from,
            depth_to=depth_to,
            lon_limit=lon_limit,
            lat_limit=lat_limit,
            key=key,
            results=[]
        )

    elif prop_name == "Trophic":
        prop_var_1 = parts[1]
        prop_var_2 = parts[2]
        prop_var_3 = parts[3]
        depth_from = int(parts[4])
        depth_to = int(parts[5])
        lon_limit = tuple(parts[6].split(';'))
        lat_limit = tuple(parts[7].split(';'))
        key = parts[8]

        prop_var = [prop_var_1, prop_var_2, prop_var_3]

        log.info(f"Property: {prop_name} from depth level {depth_from} to {depth_to}")

        return Property(
            prop_name=prop_name,
            variables=prop_var,
            depth_from=depth_from,
            depth_to=depth_to,
            lon_limit=lon_limit,
            lat_limit=lat_limit,
            key=key,
            results=[]
        )


def _parse_map(value: str) -> MapVariable:
    """Parse Map parameter line."""
    parts = value.split(',')
    variable = parts[0]
    level = parts[1]

    log.info(f"Map: {variable}")

    return MapVariable(
        name=variable,
        level=level
    )
