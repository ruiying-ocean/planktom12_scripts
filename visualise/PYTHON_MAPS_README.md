## Python-Based Map Generation for PlankTom

### Overview

This directory now contains Python-based map generation tools that **replace Ferret** for creating oceanographic visualizations. The new system:

- ✅ Uses **xarray** for NetCDF handling
- ✅ Uses **cartopy** for map projections
- ✅ Uses **cmocean** colormaps (perceptually uniform, oceanography-specific)
- ✅ Matches the style from `~/tompy/code/` notebooks
- ✅ No Ferret dependency required
- ✅ Generates PNG output directly (no GIF → PNG conversion needed)

### Files

#### Core Python Modules

**`ocean_maps.py`** - Reusable plotting infrastructure
- `OceanMapPlotter` class for consistent map styling
- Variable metadata (units, colormaps, limits)
- Land masking utilities
- Unit conversion functions

**`python_maps.py`** - Main driver script
- Generates all map types (PFTs, diagnostics, nutrients)
- Command-line interface
- Replaces: `maps.jnl`, `mapsPFT.jnl`, `mapsDiff.jnl`

#### Shell Scripts

**`annualMaps_python.sh`** - NEW Python-based workflow
- Replacement for `annualMaps.sh`
- Calls `python_maps.py` instead of Ferret
- No module loading or ImageMagick needed

**`annualMaps.sh`** - LEGACY Ferret-based workflow
- Original script (kept for reference)
- Requires Ferret and ImageMagick modules

### Usage

#### Basic Usage

```bash
# New Python workflow
./annualMaps_python.sh <run_name> <year> <base_directory>

# Example
./annualMaps_python.sh ORCA2_test 2020 /path/to/runs/
```

#### Direct Python Script Usage

```bash
# More control over output
python python_maps.py <run_name> <year_start> <year_end> \
    --basedir <path_to_runs> \
    --output-dir <output_directory> \
    --mask-path <path_to_basin_mask.nc>

# Example
python python_maps.py ORCA2_test 2020 2020 \
    --basedir .. \
    --output-dir ./maps/ \
    --mask-path /gpfs/data/greenocean/software/resources/breakdown/basin_mask.nc
```

### Input Requirements

The script expects NEMO NetCDF files in this structure:

```
<basedir>/<run_name>/
    ├── ORCA2_1m_YYYYMMDD_YYYYMMDD_ptrc_T.nc   # Passive tracers
    └── ORCA2_1m_YYYYMMDD_YYYYMMDD_diad_T.nc   # Diagnostics
```

### Output

The script generates three main figure types:

1. **`<run>_<year>_diagnostics.png`** - Ecosystem diagnostics
   - Total chlorophyll (`_TChl`)
   - Export at 100m (`_EXP`)
   - Integrated primary production (`_PPINT`)

2. **`<run>_<year>_phytos.png`** - Phytoplankton functional types (2×3 grid)
   - PIC (Picophytoplankton)
   - FIX (Nitrogen Fixers)
   - COC (Coccolithophores)
   - DIA (Diatoms)
   - MIX (Mixotrophs)
   - PHA (Phaeocystis)
   - **Shows**: Integrated biomass (Pg C) with observational ranges

3. **`<run>_<year>_zoos.png`** - Zooplankton functional types (2×3 grid)
   - BAC (Bacteria)
   - PRO (Protozooplankton)
   - MES (Mesozooplankton)
   - PTE (Pteropods)
   - CRU (Macrozooplankton/Crustaceans)
   - GEL (Gelatinous Zooplankton)
   - **Shows**: Integrated biomass (Pg C) with observational ranges

### Dependencies

Required Python packages:

```bash
# Core
numpy
pandas
xarray

# Visualization
matplotlib
cartopy
cmocean

# NetCDF
netCDF4
```

Install with conda:

```bash
conda install -c conda-forge xarray cartopy cmocean netCDF4 matplotlib numpy pandas
```

### Key Differences from Ferret

| Feature | Ferret (old) | Python (new) |
|---------|-------------|--------------|
| **Output format** | GIF → PNG (via ImageMagick) | PNG directly |
| **Colormaps** | `rainbow_cmyk`, `light_centered` | `cmocean` (perceptually uniform) |
| **Projections** | PlateCarree only | Any cartopy projection |
| **Processing** | Batch script with module loads | Pure Python |
| **Debugging** | Limited | Full Python debugging |
| **Extensibility** | Ferret scripting | Python ecosystem |

### Migration Status

- ✅ **PFT maps** - Fully implemented (phyto + zoo)
- ✅ **Ecosystem diagnostics** - Fully implemented
- ✅ **Module infrastructure** - Complete
- ⚠️ **Nutrient comparison** - Implemented, needs obs data paths
- ⚠️ **Difference maps** - Framework ready, needs implementation
- ⚠️ **Physical variables** - Framework ready, needs implementation

### Customization

#### Changing Colormaps

Edit `ocean_maps.py` variable dictionaries:

```python
ECOSYSTEM_VARS = {
    '_TChl': {
        'cmap': 'cmocean:algae',  # Change to any matplotlib/cmocean colormap
        ...
    }
}
```

Available cmocean colormaps:
- `cmocean:thermal` - SST
- `cmocean:haline` - SSS
- `cmocean:dense` - Nutrients
- `cmocean:algae` - Chlorophyll
- `cmocean:matter` - Biomass
- `cmocean:balance` - Diverging (differences)

#### Changing Projections

In `python_maps.py`, modify subplot creation:

```python
# Robinson projection (like warming_map.ipynb)
fig, axs = plotter.create_subplot_grid(
    nrows=2, ncols=3,
    projection=ccrs.Robinson(),  # Change here
    figsize=(10, 5)
)
```

#### Adding New Variables

1. Add metadata to `ocean_maps.py`:

```python
ECOSYSTEM_VARS['new_var'] = {
    'long_name': 'My Variable',
    'units': 'unit',
    'vmax': 100,
    'cmap': 'cmocean:matter'
}
```

2. Add to plotting function in `python_maps.py`

### Troubleshooting

**Problem**: `FileNotFoundError: basin_mask.nc`

**Solution**: Specify correct mask path:
```bash
python python_maps.py ... --mask-path /correct/path/to/basin_mask.nc
```

**Problem**: Variable not found in dataset

**Solution**: Check variable names in NetCDF files:
```bash
ncdump -h ORCA2_1m_*_ptrc_T.nc | grep variables
```

**Problem**: Maps look different from Ferret

**Solution**: This is expected! Python uses better colormaps (cmocean). To match Ferret more closely, you can use `'rainbow'` or `'RdBu_r'` instead of cmocean maps.

### References

- **Style inspiration**: `~/tompy/code/OBio_state.ipynb`, `warming_map.ipynb`
- **Cartopy docs**: https://scitools.org.uk/cartopy/docs/latest/
- **cmocean**: https://matplotlib.org/cmocean/
- **xarray**: http://xarray.pydata.org/
