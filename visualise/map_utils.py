"""
Ocean mapping utilities for PlankTom model output visualization.
Based on the style from tompy notebooks (OBio_state.ipynb, warming_map.ipynb).
"""

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colorbar
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
        # Try uppercase first (AREA, VOLUME), then lowercase
        area_key = 'AREA' if 'AREA' in mask_ds else 'area'
        volume_key = 'VOLUME' if 'VOLUME' in mask_ds else 'volume'

        if area_key in mask_ds:
            self.land_mask_2d = mask_ds[area_key] > 0
            self.area = mask_ds[area_key]

        if volume_key in mask_ds:
            self.volume = mask_ds[volume_key]

            # Rename 'z' dimension to 'deptht' for consistency with NEMO output
            if 'z' in self.volume.dims:
                self.volume = self.volume.rename({'z': 'deptht'})

    def load_data(self, filepath: str, variables: Optional[List[str]] = None,
                  volume: Optional[xr.DataArray] = None, chunks: dict = None) -> xr.Dataset:
        """
        Load NEMO NetCDF output file with automatic processing.
        Follows the logic from nemo_func.open_nc()

        Args:
            filepath: Path to NetCDF file
            variables: Optional list of variables to load (loads all if None)
            volume: Volume data for vertical integration (optional)
            chunks: Dask chunking spec for lazy loading (e.g., {'time_counter': 1})

        Returns:
            xarray Dataset with loaded and processed variables
        """
        # Use lazy loading with dask if chunks specified
        if chunks:
            ds = xr.open_dataset(filepath, decode_times=False, chunks=chunks)
        else:
            ds = xr.open_dataset(filepath, decode_times=False)

        # When using dask chunks, xarray may rename dimensions to 'z' instead of 'deptht'
        # Rename back to 'deptht' for consistency with volume and nemo_func.py
        if 'z' in ds.dims:
            ds = ds.rename({'z': 'deptht'})

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
            # Secondary production (grazing) - sum of all grazing terms
            ds['_SP'] = ds['GRAPRO'] + ds['GRAMES'] + ds['GRAPTE'] + ds['GRACRU'] + ds['GRAGEL']

            # NPP and derived variables
            ds['_NPP'] = ds['PPT']
            ds['_RECYCLE'] = ds['PPT'] - ds['_SP']

            # Transfer efficiency and export ratio
            if 'EXP' in ds and 'EXP1000' in ds:
                ds['_Teff'] = ds['EXP1000'] / ds['EXP']  # EXP1000/EXP
            if 'EXP' in ds and 'PPT' in ds:
                ds['_eratio'] = ds['EXP'] / ds['PPT']
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
            others = ['GRAPRO', 'GRAMES', 'GRAPTE', 'GRACRU', 'GRAGEL',
                      '_SP', '_RECYCLE', '_NPP']

            for var in others:
                if var in ds:
                    new_name = self._new_varname(var, '')
                    ds[new_name] = ds[var] * second_to_year * mole_to_gC  # gC/m³/yr

            ## Dimensionless ratios (no unit conversion needed, just copy with underscore prefix)
            for var in ['_Teff', '_eratio']:
                if var in ds:
                    new_name = self._new_varname(var, '')
                    ds[new_name] = ds[var]

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

        Note: Assumes depth dimension is named 'deptht' (ensured by load_data)
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
        # Check if we have proper 2D coordinates
        has_nav_coords = 'nav_lon' in data.coords and 'nav_lat' in data.coords

        if has_nav_coords:
            # Use nav_lon and nav_lat directly as x/y coordinates
            lon = data.coords['nav_lon']
            lat = data.coords['nav_lat']

            # If nav_lon/nav_lat are 2D, use them directly with pcolormesh
            if lon.ndim == 2 and lat.ndim == 2:
                im = ax.pcolormesh(
                    lon.values,
                    lat.values,
                    data.values,
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                    transform=transform,
                    **kwargs
                )
            else:
                # Fallback to xarray plot for 1D coordinates
                im = data.plot.pcolormesh(
                    ax=ax,
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                    x='nav_lon',
                    y='nav_lat',
                    add_colorbar=add_colorbar,
                    transform=transform,
                    **kwargs
                )
        else:
            # Use dimension-based coordinates (x, y)
            im = data.plot.pcolormesh(
                ax=ax,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
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
    ) -> matplotlib.colorbar.Colorbar:
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
        'long_name': 'Chl',
        'units': 'µg Chl L⁻¹',
        'vmax': 1.2,
        'depth_index': 0,
        'cmap': 'turbo'
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
        'long_name': 'NPP',
        'units': 'g C m⁻² yr⁻¹',
        'vmax': 400,
        'depth_index': None,
        'cmap': 'viridis'
    },
    '_EXP': {
        'long_name': 'EXP',
        'units': 'g C m⁻² yr⁻¹',
        'vmax': 80,
        'depth_index': 10,  # ~100m depth
        'cmap': 'viridis'
    },
    'dpco2': {
        'long_name': 'dpCO2',
        'units': 'ppm',
        'vmax': 100,
        'vmin': -100,
        'depth_index': None,
        'cmap': 'RdYlBu_r'
    },
    '_SP': {
        'long_name': 'Secondary Production',
        'units': 'g C m⁻³ yr⁻¹',
        'vmax': 50,
        'depth_index': 0,
        'cmap': 'viridis'
    },
    '_SPINT': {
        'long_name': 'Secondary Production',
        'units': 'Tg C yr⁻¹',
        'vmax': 20,
        'depth_index': None,
        'cmap': 'viridis'
    },
    '_RECYCLE': {
        'long_name': 'Recycled Production',
        'units': 'g C m⁻³ yr⁻¹',
        'vmax': 200,
        'depth_index': 0,
        'cmap': 'viridis'
    },
    '_RECYCLEINT': {
        'long_name': 'Recycled Production',
        'units': 'Tg C yr⁻¹',
        'vmax': 10,
        'depth_index': None,
        'cmap': 'viridis'
    },
    '_eratio': {
        'long_name': 'Export Ratio (e-ratio)',
        'units': 'dimensionless',
        'vmax': 0.5,
        'vmin': 0,
        'depth_index': None,
        'cmap': 'plasma'
    },
    '_Teff': {
        'long_name': 'Transfer Efficiency',
        'units': 'dimensionless',
        'vmax': 0.2,
        'vmin': 0,
        'depth_index': None,
        'cmap': 'plasma'
    }
}

# Nutrients
NUTRIENT_VARS = {
    '_NO3': {
        'long_name': 'Nitrate',
        'units': 'µmol L⁻¹',
        'vmax': 30,
        'cmap': 'turbo'
    },
    '_PO4': {
        'long_name': 'Phosphate',
        'units': 'µmol L⁻¹',
        'vmax': 2.5,
        'cmap': 'turbo'
    },
    '_Si': {
        'long_name': 'Silica',
        'units': 'µmol L⁻¹',
        'vmax': 60,
        'cmap': 'turbo'
    },
    '_Fer': {
        'long_name': 'Iron',
        'units': 'nmol L⁻¹',
        'vmax': 1.5,
        'cmap': 'turbo'
    },
    '_O2': {
        'long_name': 'Oxygen',
        'units': 'µmol L⁻¹',
        'vmax': 250,
        'vmin': 0,
        'depth_index': 17,  # ~300m depth
        'cmap': 'turbo'
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
    NOTE: Variables starting with '_' are already converted by _convert_units()

    Args:
        data: Input data array
        var_name: Variable name to determine conversion

    Returns:
        Data converted to plotting units
    """
    # If variable starts with '_', it's already been converted by _convert_units()
    if var_name.startswith('_'):
        return data

    # Convert based on variable name patterns (for raw variables)
    if 'INT' in var_name:
        # Integrated quantities - already in right units
        return data

    if 'Fer' in var_name:
        # Iron: convert to nmol/L (multiply by 1e9)
        return data * 1e9

    if 'tchl' in var_name.lower():
        # Chlorophyll: convert to µg/L (multiply by 1e6)
        return data * 1e6

    if var_name in ['NO3', 'PO4', 'Si', 'O2']:
        # Other nutrients: convert to µmol/L (multiply by 1e6)
        return data * 1e6

    # Default: no conversion
    return data
