"""
Ocean mapping utilities for PlankTom model output visualization.
Based on the style from tompy notebooks (OBio_state.ipynb, warming_map.ipynb).
"""

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Union


class OceanMapPlotter:
    """
    Handles loading NEMO/PlankTom NetCDF output and creating publication-quality maps.

    Follows the plotting style from tompy:
    - Uses xarray for NetCDF handling
    - Cartopy for map projections
    - Perceptually uniform colormaps
    - Horizontal colorbars with proper units
    """

    def __init__(self, mask_path: str = "/gpfs/data/greenocean/software/resources/breakdown/basin_mask.nc"):
        """
        Initialize the map plotter with land/basin masks.

        Args:
            mask_path: Path to basin mask NetCDF file
        """
        self.mask_path = mask_path
        self.land_mask_2d = None
        self.land_mask_3d = None
        self.area = None
        self.volume = None

        # Load masks if available
        try:
            self._load_masks()
        except FileNotFoundError:
            print(f"Warning: Mask file not found at {mask_path}. Will use data-based masking.")

    def _load_masks(self):
        """Load land masks and area/volume from basin mask file."""
        mask_ds = xr.open_dataset(self.mask_path)

        # Create land mask from area (ocean cells have area > 0)
        self.land_mask_2d = mask_ds['area'] > 0

        # Store area and volume for integration
        if 'area' in mask_ds:
            self.area = mask_ds['area']
        if 'volume' in mask_ds:
            self.volume = mask_ds['volume']

    def load_data(self, filepath: str, variables: Optional[List[str]] = None,
                  volume: Optional[xr.DataArray] = None) -> xr.Dataset:
        """
        Load NEMO NetCDF output file with automatic processing.
        Follows the logic from nemo_func.open_nc()

        Args:
            filepath: Path to NetCDF file
            variables: Optional list of variables to load (loads all if None)
            volume: Volume data for vertical integration (optional)

        Returns:
            xarray Dataset with loaded and processed variables
        """
        ds = xr.open_dataset(filepath, decode_times=False)

        # Determine file type
        if 'ptrc_T' in filepath:
            kind = 'ptrc'
        elif 'diad_T' in filepath:
            kind = 'diad'
        else:
            kind = None

        # Add new variables
        ds = self._add_new_var(ds, suffix=kind)

        # Convert units
        if kind in ['ptrc', 'diad']:
            ds = self._convert_units(ds, suffix=kind)

        # Vertical and global integration
        if volume is not None and kind in ['ptrc', 'diad']:
            ds = self._vint(ds, suffix=kind, volume=volume)
            ds = self._gint(ds, suffix=kind)

        if variables:
            ds = ds[variables]

        return ds

    def _add_new_var(self, ds: xr.Dataset, suffix: str) -> xr.Dataset:
        """
        Calculate new variables based on existing ones.
        Copied from nemo_func.add_new_var()
        """
        ds = ds.copy()
        if suffix == 'ptrc':
            # Total Phytoplankton Carbon
            ds['_PHY'] = ds['PIC'] + ds['FIX'] + ds['COC'] + ds['DIA'] + ds['MIX'] + ds['PHA']
            ds['_ZOO'] = ds['BAC'] + ds['PRO'] + ds['MES'] + ds['PTE'] + ds['CRU'] + ds['GEL']
        elif suffix == 'diad':
            ds['_SP'] = ds['GRAPRO'] + ds['GRAMES'] + ds['GRAPTE'] + ds['GRACRU'] + ds['GRAGEL']
            ds['_RECYCLE'] = ds['PPT'] - ds['_SP']
            ds['_NPP'] = ds['PPT']
        return ds

    def _convert_units(self, ds: xr.Dataset, suffix: str = 'ptrc') -> xr.Dataset:
        """
        Convert units, no dimensional alteration.
        Copied from nemo_func.convert_units()
        """
        ds = ds.copy()
        if suffix == 'ptrc':
            ## concentration conversion
            ds['_NO3'] = 1e6 * ds['NO3']
            ds['_PO4'] = 1e6 / 122 * ds['PO4']
            ds['_Si'] = 1e6 * ds['Si']
            ds['_Fer'] = 1e9 * ds['Fer']
            ds['_O2'] = 1e6 * ds['O2']
            return ds

        elif suffix == 'diad':
            ds['_TChl'] = ds['TChl'] * 1e6  # mg/L to µg/m3

            ## rate unit conversion
            ## EXP: mol/m2/s => gC/m2/yr
            ## PPINT: mol/m2/s => gC/m2/yr
            second_to_year = 60 * 60 * 24 * 365
            mole_to_gC = 12.01
            ds['_EXP'] = ds['EXP'] * second_to_year * mole_to_gC
            ds['_PPINT'] = ds['PPINT'] * second_to_year * mole_to_gC

            ## PPT for each phytoplankton PFT
            ## mol/m3/s => gC/m3/yr for each PFT
            for pft in ['PIC', 'FIX', 'COC', 'DIA', 'MIX', 'PHA']:
                if f'PPT_{pft}' in ds:
                    ds[f'_PPT_{pft}'] = ds[f'PPT_{pft}'] * second_to_year * mole_to_gC

            ## also convert for new variables
            ## GRA* mol/m3/s => gC/m³/yr
            others = ['GRAPRO', 'GRAMES', 'GRAPTE', 'GRACRU', 'GRAGEL', '_SP',
                      '_RECYCLE', '_NPP']

            for var in others:
                if var in ds:
                    new_name = self._new_varname(var, '')
                    ds[new_name] = ds[var] * second_to_year * mole_to_gC  # gC/m³/yr

            return ds

        return ds

    def _new_varname(self, input_name: str, suffix: str) -> str:
        """Remove all the leading underscores."""
        input_name = input_name.lstrip('_')
        return f"_{input_name}{suffix}"

    def _vint(self, ds: xr.Dataset, suffix: str, volume: xr.DataArray) -> xr.Dataset:
        """
        Vertically integrate ptrc variables.
        Copied from nemo_func.vint()
        """
        ds = ds.copy()
        if suffix == 'ptrc':
            ## integrate PFT biomass
            pfts = ['PIC', 'FIX', 'COC', 'DIA', 'MIX', 'PHA',
                   'BAC', 'PRO', 'MES', 'PTE', 'CRU', 'GEL',
                   '_PHY', '_ZOO']

            for pft in pfts:
                if pft in ds:
                    new_name = self._new_varname(pft, 'INT')
                    ds[new_name] = (ds[pft] * volume * 12.01 * 1e3 * 1e-12).sum(dim='deptht')  ## Tg C
            return ds

        elif suffix == 'diad':
            ## rate to flux conversion
            ## integrate NPP, SP, RECYCLE, and PP for each PFT
            grazoos = ['GRAPRO', 'GRAMES', 'GRAPTE', 'GRACRU', 'GRAGEL']

            for grazoo in grazoos:
                if grazoo in ds:
                    new_name = self._new_varname(grazoo, 'INT')
                    ds[new_name] = (ds[grazoo] * volume).sum(dim='deptht') / 1e12  ## Tg C/yr

            ## vint for new variables
            if '_SP' in ds:
                ds['_SPINT'] = (ds['_SP'] * volume).sum(dim='deptht') / 1e12  ## Tg C/yr
            if '_RECYCLE' in ds:
                ds['_RECYCLEINT'] = (ds['_RECYCLE'] * volume).sum(dim='deptht') / 1e12  ## Tg C/yr
            if '_NPP' in ds:
                ds['_NPPINT'] = (ds['_NPP'] * volume).sum(dim='deptht') / 1e12  ## Tg C/yr

            return ds

        return ds

    def _gint(self, ds: xr.Dataset, suffix: str) -> xr.Dataset:
        """
        Globally integrate ptrc variables.
        Copied from nemo_func.gint()
        """
        ds = ds.copy()
        if suffix == 'ptrc':
            ## integrate PFT biomass
            pfts = ['_PICINT', '_FIXINT', '_COCINT', '_DIAINT', '_MIXINT', '_PHAINT',
                   '_BACINT', '_PROINT', '_MESINT', '_PTEINT', '_CRUINT', '_GELINT',
                   '_PHYINT', '_ZOOINT']

            for pft in pfts:
                if pft in ds:
                    new_name = self._new_varname(pft, 'GS')
                    ## already multiplied by volume in vint function
                    ds[new_name] = (ds[pft] * 1e-3).sum(dim=['x', 'y'])  ## Pg C
            return ds

        elif suffix == 'diad':
            vars = ['_GRAPROINT', '_GRAMESINT', '_GRAPTEINT', '_GRACRUINT', '_GRAGELINT',
                    '_SPINT', '_RECYCLEINT', '_NPPINT']
            for var in vars:
                if var in ds:
                    new_name = self._new_varname(var, 'GS')
                    ds[new_name] = (ds[var] * 1e-3).sum(dim=['x', 'y'])  ## Pg C/yr

            return ds

        return ds

    def apply_mask(self, data: xr.DataArray, mask_2d: bool = True) -> xr.DataArray:
        """
        Apply land mask to data array.

        Args:
            data: xarray DataArray to mask
            mask_2d: If True, use 2D mask; if False, use 3D mask

        Returns:
            Masked DataArray
        """
        if mask_2d and self.land_mask_2d is not None:
            return data.where(self.land_mask_2d != 0)
        elif not mask_2d and self.land_mask_3d is not None:
            return data.where(self.land_mask_3d != 0)
        else:
            # Fallback: mask where data is NaN or zero
            return data.where(data != 0)

    def create_subplot_grid(
        self,
        nrows: int,
        ncols: int,
        projection: ccrs.Projection = ccrs.PlateCarree(),
        figsize: Tuple[float, float] = (10, 4)
    ) -> Tuple[plt.Figure, np.ndarray]:
        """
        Create a grid of map subplots with consistent styling.

        Args:
            nrows: Number of rows
            ncols: Number of columns
            projection: Cartopy projection (default PlateCarree)
            figsize: Figure size in inches

        Returns:
            Tuple of (figure, axes array)
        """
        fig, axs = plt.subplots(
            nrows, ncols,
            subplot_kw={'projection': projection},
            figsize=figsize,
            constrained_layout=True
        )

        # Ensure axs is always an array
        if nrows == 1 and ncols == 1:
            axs = np.array([axs])
        elif nrows == 1 or ncols == 1:
            axs = axs.reshape(nrows, ncols)

        # Add coastlines to all axes
        for ax in axs.flat:
            ax.coastlines(resolution='110m')

        return fig, axs

    def plot_variable(
        self,
        ax: plt.Axes,
        data: xr.DataArray,
        cmap: str = 'viridis',
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        add_colorbar: bool = False,
        cbar_label: str = '',
        transform: ccrs.Projection = ccrs.PlateCarree(),
        **kwargs
    ) -> plt.cm.ScalarMappable:
        """
        Plot a single variable on a map axis.

        Args:
            ax: Matplotlib axis with map projection
            data: xarray DataArray to plot
            cmap: Colormap name
            vmin: Minimum value for colorscale
            vmax: Maximum value for colorscale
            add_colorbar: Whether to add colorbar to this plot
            cbar_label: Label for colorbar
            transform: Coordinate transform (default PlateCarree)
            **kwargs: Additional arguments passed to plot()

        Returns:
            Mappable object for creating colorbars
        """
        im = data.plot(
            ax=ax,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            x='nav_lon' if 'nav_lon' in data.coords else 'x',
            y='nav_lat' if 'nav_lat' in data.coords else 'y',
            add_colorbar=add_colorbar,
            transform=transform,
            **kwargs
        )

        return im

    def add_shared_colorbar(
        self,
        fig: plt.Figure,
        im: plt.cm.ScalarMappable,
        axs: Union[plt.Axes, np.ndarray],
        label: str = '',
        orientation: str = 'horizontal',
        **kwargs
    ) -> plt.colorbar.Colorbar:
        """
        Add a shared colorbar for multiple subplots.

        Args:
            fig: Figure object
            im: Mappable object from plot
            axs: Axis or array of axes to attach colorbar to
            label: Colorbar label
            orientation: 'horizontal' or 'vertical'
            **kwargs: Additional arguments for colorbar

        Returns:
            Colorbar object
        """
        cbar_kwargs = {
            'orientation': orientation,
            'pad': 0.05,
            'shrink': 0.7,
            'aspect': 30 if orientation == 'horizontal' else 20
        }
        cbar_kwargs.update(kwargs)

        cbar = fig.colorbar(im, ax=axs, **cbar_kwargs)
        cbar.set_label(label, fontsize=12)
        cbar.ax.tick_params(labelsize=10)

        return cbar


# ============================================================================
# Variable definitions and metadata
# ============================================================================

# Plankton functional types
PHYTOS = ['PIC', 'FIX', 'COC', 'DIA', 'MIX', 'PHA']
ZOOS = ['BAC', 'PRO', 'MES', 'PTE', 'CRU', 'GEL']

PHYTO_NAMES = {
    'PIC': 'Picophytoplankton',
    'FIX': 'Nitrogen Fixers',
    'COC': 'Coccolithophores',
    'DIA': 'Diatoms',
    'MIX': 'Mixotrophs',
    'PHA': 'Phaeocystis'
}

ZOO_NAMES = {
    'BAC': 'Bacteria',
    'PRO': 'Protozooplankton',
    'MES': 'Mesozooplankton',
    'PTE': 'Pteropods',
    'CRU': 'Macrozooplankton/Crustaceans',
    'GEL': 'Gelatinous Zooplankton'
}

# Observational biomass ranges (PgC) from literature
BIOMASS_RANGES = {
    # Phytoplankton biomass 0-200 m (PgC)
    '_PICINT': (0.28, 0.52),
    '_FIXINT': (0.008, 0.12),
    '_COCINT': (0.001, 0.032),
    '_PHAINT': (0.11, 0.69),
    '_DIAINT': (0.013, 0.75),
    '_MIXINT': (np.nan, np.nan),

    # Heterotrophs biomass 0-200 m (PgC)
    '_BACINT': (0.25, 0.26),
    '_PROINT': (0.10, 0.37),
    '_MESINT': (0.21, 0.34),
    '_CRUINT': (0.010, 0.64),
    '_PTEINT': (0.048, 0.057),
    '_GELINT': (0.1, 3.1)
}

# Ecosystem diagnostics
ECOSYSTEM_VARS = {
    '_TChl': {
        'long_name': 'Total Surface Chlorophyll',
        'units': 'µg Chl L⁻¹',
        'vmax': 1.2,
        'depth_index': 0,
        'cmap': 'NCV_jet'
    },
    'Cflx': {
        'long_name': 'Surface Carbon Flux',
        'units': 'µmol m⁻² s⁻¹',
        'vmax': 0.4,
        'vmin': -0.2,
        'depth_index': None,
        'cmap': 'RdYlBu_r'
    },
    '_PPINT': {
        'long_name': 'Integrated Primary Production',
        'units': 'g C m⁻² yr⁻¹',
        'vmax': 400,
        'depth_index': None,
        'cmap': 'Spectral_r'
    },
    '_EXP': {
        'long_name': 'Export at 100m',
        'units': 'g C m⁻² yr⁻¹',
        'vmax': 80,
        'depth_index': 10,  # ~100m depth
        'cmap': 'Spectral_r'
    },
    'dpco2': {
        'long_name': 'dpCO2',
        'units': 'ppm',
        'vmax': 100,
        'vmin': -100,
        'depth_index': None,
        'cmap': 'RdYlBu_r'
    }
}

# Nutrients
NUTRIENT_VARS = {
    '_NO3': {
        'long_name': 'Nitrate',
        'units': 'µmol L⁻¹',
        'vmax': 30,
        'cmap': 'viridis'
    },
    '_PO4': {
        'long_name': 'Phosphate',
        'units': 'µmol L⁻¹',
        'vmax': 2.5,
        'cmap': 'viridis'
    },
    '_Si': {
        'long_name': 'Silica',
        'units': 'µmol L⁻¹',
        'vmax': 60,
        'cmap': 'viridis'
    },
    '_Fer': {
        'long_name': 'Iron',
        'units': 'nmol L⁻¹',
        'vmax': 1.5,
        'cmap': 'viridis'
    },
    '_O2': {
        'long_name': 'Oxygen',
        'units': 'µmol L⁻¹',
        'vmax': 400,
        'vmin': 200,
        'cmap': 'viridis'
    }
}

# Physical variables
PHYSICAL_VARS = {
    'tos': {
        'long_name': 'Sea Surface Temperature',
        'units': '°C',
        'vmax': 30,
        'cmap': 'cmocean:thermal'
    },
    'sos': {
        'long_name': 'Sea Surface Salinity',
        'units': 'PSU',
        'vmax': 37,
        'cmap': 'cmocean:haline'
    },
    'mldr10_1': {
        'long_name': 'Mixed Layer Depth',
        'units': 'm',
        'vmax': 500,
        'cmap': 'cmocean:deep'
    }
}


def get_variable_metadata(var_name: str) -> Dict:
    """
    Get metadata for a variable.

    Args:
        var_name: Variable name

    Returns:
        Dictionary with metadata (units, colormap, limits, etc.)
    """
    for var_dict in [ECOSYSTEM_VARS, NUTRIENT_VARS, PHYSICAL_VARS]:
        if var_name in var_dict:
            return var_dict[var_name]

    # Return defaults if not found
    return {
        'long_name': var_name,
        'units': '',
        'vmax': None,
        'cmap': 'viridis'
    }


def convert_units(data: xr.DataArray, var_name: str) -> xr.DataArray:
    """
    Convert data to appropriate units for plotting.

    Args:
        data: Input data array
        var_name: Variable name to determine conversion

    Returns:
        Data converted to plotting units
    """
    # Convert based on variable name patterns
    if 'INT' in var_name:
        # Integrated quantities - convert from umol to Tg C
        # Typically already in right units from breakdown
        return data

    if '_Fer' in var_name or 'Fer' in var_name:
        # Iron: convert to nmol/L (multiply by 1e9)
        return data * 1e9

    if '_TChl' in var_name or 'tchl' in var_name:
        # Chlorophyll: convert to µg/L (multiply by 1e6)
        return data * 1e6

    if var_name in ['_NO3', '_PO4', '_Si', 'O2']:
        # Other nutrients: convert to µmol/L (multiply by 1e6)
        return data * 1e6

    # Default: no conversion
    return data
