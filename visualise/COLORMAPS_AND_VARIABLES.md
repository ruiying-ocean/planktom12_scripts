## Colormaps and Variable Names Reference

### Variable Naming Convention

The plotting system uses **two types of variables**:

#### 1. Native NEMO Variables (no underscore)
These are the raw outputs from the model:
- **From `diad_T.nc`**: `tchl`, `Cflx`, `PPINT`, `EXP`, `dpco2`, `TChl`
- **From `ptrc_T.nc`**: `NO3`, `PO4`, `Si`, `Fer`, `O2`, `PIC`, `FIX`, `COC`, `DIA`, `MIX`, `PHA`, `BAC`, `PRO`, `MES`, `PTE`, `CRU`, `GEL`
- **From `grid_T.nc`**: `tos`, `sos`, `mldr10_1`

#### 2. Derived Variables (with underscore prefix)
These are computed automatically by `ocean_maps.py` following the `nemo_func.py` pattern:

**From ptrc_T.nc:**
- `_NO3` = `NO3 * 1e6` (mol/L → µmol/L)
- `_PO4` = `PO4 * 1e6 / 122` (mol/L → µmol/L, P-normalized)
- `_Si` = `Si * 1e6` (mol/L → µmol/L)
- `_Fer` = `Fer * 1e9` (mol/L → nmol/L)
- `_O2` = `O2 * 1e6` (mol/L → µmol/L)
- `_PHY` = `PIC + FIX + COC + DIA + MIX + PHA` (total phytoplankton)
- `_ZOO` = `BAC + PRO + MES + PTE + CRU + GEL` (total zooplankton)

**From diad_T.nc:**
- `_TChl` = `TChl * 1e6` (mg/L → µg/L)
- `_EXP` = `EXP * 31536000 * 12.01` (mol/m²/s → gC/m²/yr)
- `_PPINT` = `PPINT * 31536000 * 12.01` (mol/m²/s → gC/m²/yr)
- `_SP` = sum of grazing fluxes (GRAPRO + GRAMES + GRAPTE + GRACRU + GRAGEL)
- `_RECYCLE` = `PPT - _SP` (recycled production)
- `_NPP` = `PPT` (net primary production)

### Colormap Assignments

Based on the current `ocean_maps.py` configuration:

#### Ecosystem Diagnostics (diad_T.nc)
```python
Variable       Colormap              Units              Range
--------       --------              -----              -----
_TChl          NCV_jet               µg Chl L⁻¹         0-1.2
Cflx           RdYlBu_r              µmol m⁻² s⁻¹       -0.2 to 0.4
_PPINT         Spectral_r            gC m⁻² yr⁻¹        0-400
_EXP           Spectral_r            gC m⁻² yr⁻¹        0-80
dpco2          RdYlBu_r              ppm                -100 to 100
```

#### Nutrients (ptrc_T.nc)
```python
Variable       Colormap              Units              Range
--------       --------              -----              -----
_NO3           viridis               µmol L⁻¹           0-30
_PO4           viridis               µmol L⁻¹           0-2.5
_Si            viridis               µmol L⁻¹           0-60
_Fer           viridis               nmol L⁻¹           0-1.5
_O2            viridis               µmol L⁻¹           200-400
```

#### Physical Variables (grid_T.nc)
```python
Variable       Colormap              Units              Range
--------       --------              -----              -----
tos            cmocean:thermal       °C                 0-30
sos            cmocean:haline        PSU                30-37
mldr10_1       cmocean:deep          m                  0-500
```

#### Plankton Functional Types (ptrc_T.nc)
```python
Variable       Colormap              Units              Range
--------       --------              -----              -----
PIC-GEL        NCV_jet               µmol L⁻¹           0-5
(surface)
```

### Comparison with Ferret

#### Ferret → Python Colormap Mapping

| Ferret Palette | Python Equivalent | Use Case |
|---------------|-------------------|----------|
| `rainbow_cmyk` | `NCV_jet` (chlorophyll, PFTs) or `viridis` (nutrients) | Sequential data |
| `light_centered` | `RdYlBu_r` | Diverging data (Cflx, dpco2, differences) |

#### Colormap Rationale

**NCV_jet** (for chlorophyll & PFTs):
- Similar visual appearance to Ferret's rainbow_cmyk
- Matches your notebook style
- Familiar to oceanographers

**viridis** (for nutrients):
- Perceptually uniform: Equal data changes = equal visual changes
- Colorblind-friendly: Accessible to ~8% of population
- Print-safe: Works in grayscale
- Modern standard in scientific visualization

**RdYlBu_r** (for diverging data):
- Red-Yellow-Blue reversed (blue for negative, red for positive)
- Intuitive for ocean carbon flux (blue = uptake, red = outgassing)
- Good contrast at zero

**Spectral_r** (for production/export):
- Rainbow-like but more balanced than jet
- Good for highlighting spatial patterns
- Reversed to go from cool to warm

### Your Notebook Colormaps (from tompy)

If you prefer to match your existing notebooks exactly, you use:

**From OBio_state.ipynb:**
```python
from palettable.mycarta import LinearL_12
from palettable.scientific.diverging import Cork_19
import colormaps as cmps

chl_cmap = Cork_19.mpl_colormap           # For chlorophyll
exp_cmap = LinearL_12.mpl_colormap        # For export
ppt_cmap = cmps.cet_d_linear_bjy          # For primary production
nut_cmap = cmps.NCV_jet                   # For nutrients
pft_cmap = cmps.NCV_jet                   # For PFTs
```

**From warming_map.ipynb:**
```python
from cmap import Colormap
blc = Colormap('cmocean:balance').to_mpl(21)   # For differences/warming
```

### How to Change Colormaps

#### Option 1: Edit ocean_maps.py directly

```python
# In ocean_maps.py, around line 260
ECOSYSTEM_VARS = {
    '_TChl': {
        'cmap': 'your_colormap_here',  # Change this
        ...
    }
}
```

#### Option 2: Override when plotting

```python
# In python_maps.py or custom scripts
plot_pft_maps(
    plotter=plotter,
    ptrc_ds=ptrc_ds,
    pft_list=PHYTOS,
    pft_type='phyto',
    output_path=output_path,
    cmap='viridis'  # Override default
)
```

### Available Colormap Options

#### Matplotlib Built-in
- Sequential: `'viridis'`, `'plasma'`, `'inferno'`, `'magma'`, `'cividis'`
- Diverging: `'RdBu_r'`, `'seismic'`, `'coolwarm'`, `'bwr'`
- Qualitative: `'tab10'`, `'tab20'`, `'Set1'`, `'Set2'`
- Classic: `'rainbow'`, `'jet'` (not recommended - perceptual artifacts)

#### cmocean (Oceanography-specific, requires `conda install cmocean`)
- **Sequential:**
  - `'cmocean:thermal'` - Temperature (yellow to red)
  - `'cmocean:haline'` - Salinity (blue to yellow-green)
  - `'cmocean:dense'` - Density/nutrients (light to dark blue)
  - `'cmocean:algae'` - Chlorophyll (white to green)
  - `'cmocean:matter'`` - Organic matter (yellow to purple)
  - `'cmocean:turbid'` - Turbidity (brown scale)
  - `'cmocean:speed'` - Velocity (yellow-green to blue)
  - `'cmocean:amp'` - Amplitude (light to dark red)
  - `'cmocean:tempo'` - Time-varying (white to teal)
  - `'cmocean:deep'` - Bathymetry (yellow to blue)

- **Diverging:**
  - `'cmocean:balance'` - Zero-centered (blue-white-red)
  - `'cmocean:delta'` - Anomalies (blue-white-red, muted)
  - `'cmocean:curl'` - Rotation (blue-white-green)
  - `'cmocean:diff'` - Differences (blue-tan-brown)

- **Cyclic:**
  - `'cmocean:phase'` - Phase data (wraps around)

#### Palettable (requires `conda install palettable`)
```python
from palettable.mycarta import LinearL_12
cmap = LinearL_12.mpl_colormap
```

#### Scientific Colormaps (requires `pip install cmasher` or similar)
Many options from various scientific color libraries

### Recommendations by Variable Type

| Variable Type | Recommended Colormap | Reason |
|--------------|---------------------|---------|
| **Chlorophyll** | `cmocean:algae` | Green theme, perceptually uniform |
| **Nutrients** | `cmocean:dense` | Clear progression, ocean context |
| **Temperature** | `cmocean:thermal` | Intuitive warm colors |
| **Salinity** | `cmocean:haline` | Ocean-specific palette |
| **Carbon flux** | `cmocean:balance` | Diverging, highlights ocean uptake/outgassing |
| **Differences** | `cmocean:balance` or `cmocean:delta` | Zero-centered |
| **Biomass** | `cmocean:matter` or `viridis` | Sequential, no artifacts |
| **Export/Production** | `cmocean:tempo` or `cmocean:matter` | Organic carbon theme |

### Implementation Notes

1. **Automatic variable creation**: When you use `plotter.load_data()`, it automatically creates derived variables (those with `_` prefix)

2. **No manual unit conversion needed**: The `_add_derived_variables()` function handles all unit conversions

3. **Depth indexing**: Some variables need depth slicing:
   - Surface: `k=1` in Ferret = `isel(deptht=0)` in Python
   - 100m: `k=11` in Ferret = `isel(deptht=10)` in Python (0-indexed)

4. **Time averaging**: Use `.mean(dim='time_counter')` for annual means

### Quick Start Example

```python
from ocean_maps import OceanMapPlotter

# Initialize plotter
plotter = OceanMapPlotter()

# Load data (automatically adds derived variables)
diad_ds = plotter.load_data('ORCA2_1m_20200101_20201231_diad_T.nc')

# Now you can use both native and derived variables:
print(diad_ds['TChl'])   # Native: mg/L
print(diad_ds['_TChl'])  # Derived: µg/L (converted automatically)
```
