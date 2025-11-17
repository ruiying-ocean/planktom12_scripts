# PlankTom Runner

A comprehensive workflow system for running NEMO-PlankTom ocean biogeochemistry models and generating visualizations and analyses.

## Overview

This repository provides tools for:
- Setting up and running NEMO-PlankTom model simulations
- Extracting statistics from model output (breakdown files)
- Generating visualizations (time series, spatial maps, vertical profiles)
- Comparing multiple model runs
- Creating HTML reports

## Directory Structure

```
PlankTomRunner/
├── breakdown/          # Statistics extraction from NetCDF output
├── visualise/          # Single model visualization tools
│   └── multimodel/     # Multi-model comparison tools
├── setUpRun.sh         # Model run setup script
├── setup_spin.sh       # Spin-up run setup script
└── README.md
```

## Prerequisites

- Python 3.8+
- Required Python packages: xarray, numpy, pandas, matplotlib, cartopy, netCDF4
- NEMO ocean model
- Quarto (for HTML report generation)

## Quick Start

### Single Model Workflow

1. **Run the model** (generates NetCDF output in `~/scratch/ModelRuns/<model_id>/`)

2. **Generate breakdown statistics**:
   ```bash
   cd ~/scratch/ModelRuns/<model_id>/
   python /path/to/PlankTomRunner/breakdown/breakdown.py \
       /path/to/PlankTomRunner/breakdown/breakdown_config.toml \
       <start_year> <end_year>
   ```

3. **Create visualizations**:
   ```bash
   python visualise/visualise.py <model_id> ~/scratch/ModelRuns
   ./visualise/annualMaps_python.sh <model_id> <year>
   ```

4. **Generate HTML report**:
   ```bash
   ./visualise/createHTML.sh <model_id>
   # Or specify custom base directory:
   # ./visualise/createHTML.sh <model_id> ~/scratch/ModelRuns
   ```

### Multi-Model Comparison Workflow

1. **Run multiple models** (each generates output in `~/scratch/ModelRuns/<model_id>/`)

2. **Generate breakdown files for each model**:
   ```bash
   cd ~/scratch/ModelRuns/TOM12_RY_SPE2/
   python /path/to/PlankTomRunner/breakdown/breakdown.py \
       /path/to/PlankTomRunner/breakdown/breakdown_config.toml 1750 1790

   cd ~/scratch/ModelRuns/TOM12_RY_SPE5/
   python /path/to/PlankTomRunner/breakdown/breakdown.py \
       /path/to/PlankTomRunner/breakdown/breakdown_config.toml 1750 1790
   ```

3. **Create comparison configuration** (`modelsToPlot.csv`):
   ```csv
   model_id,description,start_year,to_year
   TOM12_RY_SPE2,Control,1750,1790
   TOM12_RY_SPE5,High Export,1750,1790
   ```

4. **Generate comparison report**:
   ```bash
   cd /path/to/output/directory
   cp modelsToPlot.csv .
   /path/to/PlankTomRunner/visualise/multimodel/multimodel.sh
   ```

## Detailed Documentation

### Breakdown Tools

**Purpose**: Extract integrated statistics from model NetCDF output files.

**Main Script**: `breakdown/breakdown.py`

**Usage**:
```bash
cd <model_output_directory>
python /path/to/breakdown/breakdown.py <config_file.toml> <start_year> <end_year>
```

**Important**: The breakdown script must be run from the model output directory where the NetCDF files are located (it looks for files in the current directory).

**Output**: Creates CSV files in `<model_dir>/<model_id>/`:
- `breakdown.sur.annual.csv` - Surface variables (e.g., air-sea carbon flux)
- `breakdown.lev.annual.csv` - Level variables (e.g., export at 100m)
- `breakdown.vol.annual.csv` - Volume-integrated variables (e.g., primary production)
- `breakdown.ave.annual.csv` - Volume-averaged variables (e.g., nutrients)
- `breakdown.int.annual.csv` - Depth-integrated variables (e.g., phytoplankton biomass)
- Monthly versions: `*.monthly.csv`

**Configuration**: `breakdown/breakdown_config.toml`

### Single Model Visualization

**Location**: `visualise/`

#### Time Series Plots

**Script**: `visualise.py`

**Usage**:
```bash
python visualise.py <model_id> <model_dir>
```

**Output**: Generates time series plots for:
- Global carbon fluxes
- Primary production and export
- Nutrient concentrations
- Plankton functional types
- Physical variables (temperature, salinity)

**Output location**: `<model_dir>/monitor/<model_id>/`

#### Spatial Maps

**Script**: `make_maps.py`

**Usage**:
```bash
python make_maps.py <run_name> <start_year> <end_year> \
    --basedir <model_dir> \
    --output-dir <output_dir> \
    --obs-dir <observations_dir>
```

**Convenience wrapper**:
```bash
./annualMaps_python.sh <run_name> <year> [model_output_dir]
```

**Output**: Spatial maps for:
- Nutrients (NO₃, PO₄, Si, Fe)
- Ecosystem variables (chlorophyll, export, primary production)
- Phytoplankton functional types (6 types)
- Zooplankton functional types (6 types)
- Model-observation comparisons (if observations provided)

#### Vertical Profiles

**Script**: `verticalDepth.py`

**Usage**:
```bash
python verticalDepth.py <model_id> <model_dir> <year>
```

**Output**: Vertical depth profiles for nutrients, carbon fluxes, and ecological variables.

#### Monthly Summaries

**Script**: `monthly.py`

**Usage**:
```bash
python monthly.py <model_id> <model_dir>
```

**Output**: Monthly climatology plots for key variables.

#### HTML Report

**Script**: `createHTML.sh`

**Usage**:
```bash
./createHTML.sh <model_id> [base_dir]
```

**Arguments**:
- `model_id`: Model run identifier (required)
- `base_dir`: Base directory for model output (optional, defaults to `~/scratch/ModelRuns`)

**Example**:
```bash
./createHTML.sh TOM12_RY_SPE2
# Or with custom base directory:
./createHTML.sh TOM12_RY_SPE2 /custom/path/to/models
```

**Output**: Complete HTML report with all visualizations for a single model.

**Note**: The script automatically detects the latest year from generated map files or breakdown data.

### Multi-Model Comparison

**Location**: `visualise/multimodel/`

#### Configuration File

Create `modelsToPlot.csv` with model information:

```csv
model_id,description,start_year,to_year,location
TOM12_RY_SPE2,Control,1750,1790,
TOM12_RY_SPE5,High Export,1750,1790,
```

**Column descriptions**:
- `model_id`: Model run identifier (must match directory name)
- `description`: Short description for plots
- `start_year`: First year to include in comparison
- `to_year`: Last year to include in comparison
- `location`: (Optional) Base directory for model output. Defaults to `~/scratch/ModelRuns` if empty or omitted.

**Note**: The location column can be completely omitted:
```csv
model_id,description,start_year,to_year
TOM12_RY_SPE2,Control,1750,1790
TOM12_RY_SPE5,High Export,1750,1790
```

#### Complete Workflow

**Script**: `multimodel.sh`

**Usage**:
```bash
cd /path/to/output/directory
cp modelsToPlot.csv .
/path/to/PlankTomRunner/visualise/multimodel/multimodel.sh
```

**Output**: Generates complete comparison including:
- Time series comparisons
- Spatial map comparisons
- Vertical transect comparisons (Atlantic 35°W, Pacific 170°W)
- Interactive HTML report

**Output location**: `<current_dir>/<model1>-<model2>/`

#### Individual Components

**Time series comparisons**:
```bash
python multimodel.py <save_dir>
```

**Spatial maps**:
```bash
python multimodel_maps.py modelsToPlot.csv <output_dir>
```

**Vertical transects**:
```bash
python multimodel_transects.py modelsToPlot.csv <output_dir>
```

**HTML report**:
```bash
python generate_multimodel_html.py modelsToPlot.csv <output_dir>
```

## Configuration

### Visualization Configuration

**File**: `visualise/visualise_config.toml`

**Key settings**:
- `dpi`: Figure resolution (default: 300)
- `format`: Output format - "png" (lossless), "svg" (vector), or "jpg" (compressed)
- Observational data ranges for validation
- Color schemes and plot styles

**Example**:
```toml
[figure]
dpi = 300
format = "png"

[observations.global]
PPT = [51, 65]  # Primary production range [PgC/yr]
EXP = [7.8, 12.2]  # Export production range [PgC/yr]
```

### Breakdown Configuration

**File**: `breakdown/breakdown_config.toml`

Defines which variables to extract and their spatial/temporal integration:
- Surface variables
- Level (depth) variables
- Volume-integrated variables
- Depth-integrated variables
- Volume-averaged variables

**Example**:
```toml
[[surface]]
variable = "Cflx"
units = "PgCarbonPerYr"
lon_range = [-180, 180]
lat_range = [-90, 90]

[[level]]
variable = "EXP"
level = 10  # ~100m depth
units = "PgCarbonPerYr"
lon_range = [-180, 180]
lat_range = [-90, 90]
```

## Model Setup

### New Model Run

**Script**: `setUpRun.sh`

**Usage**:
```bash
./setUpRun.sh <setup_file.dat>
```

**Setup file format**: See `setUpData_*.dat` examples in the repository.

### Spin-up Run

**Script**: `setup_spin.sh`

**Usage**:
```bash
./setup_spin.sh <setup_file.dat>
```

## Output Locations

### Default Paths

- **Model output**: `~/scratch/ModelRuns/<model_id>/`
- **Single model reports**: `~/scratch/ModelRuns/monitor/<model_id>/`
- **Multi-model reports**: Current directory when running `multimodel.sh`

### Model Output Structure

```
~/scratch/ModelRuns/<model_id>/
├── ORCA2_1m_YYYY0101_YYYY1231_diad_T.nc  # Diagnostic variables
├── ORCA2_1m_YYYY0101_YYYY1231_ptrc_T.nc  # Tracer variables
├── ORCA2_1m_YYYY0101_YYYY1231_grid_T.nc  # Physical variables
├── breakdown.sur.annual.csv               # Breakdown files
├── breakdown.lev.annual.csv
├── breakdown.vol.annual.csv
├── breakdown.ave.annual.csv
└── breakdown.int.annual.csv
```

## Variables and Units

### Key Output Variables

**Ecosystem**:
- `TChl` - Total chlorophyll [µg/L]
- `PPT` - Primary production [PgC/yr]
- `EXP` - Export production at 100m [PgC/yr]
- `Cflx` - Air-sea carbon flux [PgC/yr]

**Nutrients**:
- `NO3` - Nitrate [µmol/L]
- `PO4` - Phosphate [µmol/L]
- `Si` - Silicate [µmol/L]
- `Fer` - Iron [nmol/L]

**Phytoplankton** (6 types):
- `PIC` - Picophytoplankton
- `FIX` - Nitrogen fixers
- `COC` - Coccolithophores
- `DIA` - Diatoms
- `MIX` - Mixotrophs
- `PHA` - Phaeocystis

**Zooplankton** (6 types):
- `BAC` - Bacteria
- `PRO` - Protozooplankton
- `MES` - Mesozooplankton
- `PTE` - Pteropods
- `CRU` - Crustaceans
- `GEL` - Jellyfish

### Unit Conversions

The visualization tools automatically convert NetCDF variables to appropriate units:
- Nutrients: mol/L → µmol/L (×10⁶)
- Iron: mol/L → nmol/L (×10⁹)
- Phosphate: mol/L → µmol/L with Redfield ratio (×10⁶/122)
- Fluxes: mol/m²/s → gC/m²/yr (×31536000×12.01)
- Production: mol/m³/s → PgC/yr (integrated over volume)

## Observational Data

For model-observation comparisons, observational data should be placed in:
```
/gpfs/home/vhf24tbu/Observations/
├── woa_orca_bil.nc          # World Ocean Atlas (nutrients)
├── fe_tagliabue_orca_bil.nc # Iron observations
├── occci_chl_clim.nc        # Ocean Colour CCI chlorophyll
└── ...
```

## Troubleshooting

### Common Issues

**1. "Variable not found" warnings in multimodel transects**
- Fixed in latest version - now properly maps derived variables to base variables with unit conversion

**2. Very low EXP values in spatial maps**
- Fixed in latest version - now uses correct depth level (100m) instead of surface

**3. Missing images in HTML reports**
- Check `visualise_config.toml` format setting matches what's expected
- Default is PNG format

**4. "File not found" errors**
- Verify model output files exist in expected location
- Check NetCDF file naming convention: `ORCA2_1m_YYYY0101_YYYY1231_<type>.nc`
- Can specify custom location in `modelsToPlot.csv` if needed

**5. Empty or missing breakdown files**
- Ensure `breakdown.py` completed successfully
- Check breakdown configuration in `breakdown/breakdown_config.toml`
- Verify NetCDF variables exist in model output

### Getting Help

1. Check script usage: `python <script>.py --help`
2. Review configuration files in `visualise/` and `breakdown/`
3. Ensure all required Python packages are installed
4. Verify model output files exist and follow naming conventions

## Recent Improvements

- **Nutrient transect fixes**: Properly handles derived variables with unit conversions
- **Full depth nutrient transects**: Shows complete water column instead of just top 500m
- **Correct EXP depth extraction**: Uses 100m depth instead of surface for export production
- **Simplified CSV configuration**: Location column now optional with sensible defaults
- **Image format fixes**: Corrected Quarto template to use PNG format

## Contributing

When making changes:
1. Test with a small dataset first
2. Update relevant documentation
3. Write clear commit messages
4. Consider backwards compatibility
5. Update this README if adding new features

## License

[To be determined]

## Citation

If using this code in publications, please cite:
[Citation information to be added]
