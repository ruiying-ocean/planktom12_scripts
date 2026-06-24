"""
AMOC (Atlantic Meridional Overturning Circulation) utilities.

Reads MOC files produced by CDFtools cdfmoc and extracts AMOC metrics.
"""

import numpy as np
import xarray as xr


def read_moc_file(path: str) -> xr.Dataset:
    """
    Read a MOC NetCDF file produced by cdfmoc.

    Args:
        path: Path to moc_{year}.nc file

    Returns:
        xarray Dataset with MOC variables
    """
    return xr.open_dataset(path, decode_times=False)


def extract_amoc_26n(moc_ds: xr.Dataset,
                     lat_band: tuple = (24.0, 28.0),
                     min_depth: float = 500.0) -> float:
    """
    Extract a RAPID-comparable AMOC strength index.

    The index is the maximum of the Atlantic MOC streamfunction *below
    ``min_depth``* (default 500 m), to exclude the shallow wind-driven
    (Ekman) surface cell which can otherwise dominate a plain
    max-over-depth. That depth-max is taken for each model row whose
    latitude falls in ``lat_band`` (default 24-28N, centred on the RAPID
    array at 26.5N) and then averaged across the band to reduce
    single-gridpoint noise. Finally an annual (time) mean is taken.

    Args:
        moc_ds: MOC dataset from cdfmoc (from read_moc_file)
        lat_band: (south, north) latitude bounds in degrees N to average
            over (default (24.0, 28.0), centred on RAPID's 26.5N)
        min_depth: depths shallower than this (m) are excluded before the
            max-over-depth, removing the surface Ekman cell (default 500)

    Returns:
        AMOC strength in Sv (annual mean of the banded sub-surface max)
    """
    # cdfmoc output variable names vary by version
    # Common names: zomsfatl (Atlantic MOC), zomsfglo (Global MOC)
    moc_var = _find_moc_variable(moc_ds, ['zomsfatl', 'zomsf_atl', 'moc_atl'])

    if moc_var is None:
        raise ValueError(
            f"No Atlantic MOC variable found in dataset. "
            f"Available variables: {list(moc_ds.data_vars)}"
        )

    moc = moc_ds[moc_var]

    # Identify dimensions
    y_dim = None
    for dim in moc.dims:
        if dim.startswith('y') or dim == 'nav_y' or dim == 'lat':
            y_dim = dim
            break
    if y_dim is None:
        # Use second-to-last spatial dimension
        spatial_dims = [d for d in moc.dims if d not in ('time_counter', 'time')]
        if len(spatial_dims) >= 2:
            y_dim = spatial_dims[-2]
    if y_dim is None:
        raise ValueError("Cannot identify y-dimension in MOC dataset")

    depth_dim = None
    for dim in moc.dims:
        if 'depth' in dim or dim == 'z' or dim == 'nav_lev':
            depth_dim = dim
            break
    if depth_dim is None:
        spatial_dims = [d for d in moc.dims if d not in ('time_counter', 'time')]
        if len(spatial_dims) >= 1:
            depth_dim = spatial_dims[-1]

    # --- exclude the surface Ekman cell: keep only depths >= min_depth ---
    if depth_dim is not None and depth_dim in moc.dims:
        if depth_dim in moc_ds.variables or depth_dim in moc_ds.coords:
            depth_vals = np.abs(np.asarray(moc_ds[depth_dim].values))
            keep = np.where(depth_vals >= min_depth)[0]
            if keep.size:
                moc = moc.isel({depth_dim: keep})
            else:
                print(f"Warning: no levels below {min_depth} m; "
                      f"using full column for AMOC")
        else:
            print(f"Warning: depth coordinate '{depth_dim}' has no values; "
                  f"cannot exclude surface cell, using full column")

    # --- select the latitude band (centred on RAPID 26.5N) ---
    lat_y = None
    if 'nav_lat' in moc_ds:
        navlat = np.asarray(moc_ds['nav_lat'].values).squeeze()
        if navlat.ndim == 2:
            # latitude is ~zonally constant in the subtropics; collapse x
            navlat = np.where(navlat == 0, np.nan, navlat)
            lat_y = np.nanmean(navlat, axis=1)
        elif navlat.ndim == 1:
            lat_y = navlat

    if lat_y is not None:
        in_band = np.where((lat_y >= lat_band[0]) & (lat_y <= lat_band[1]))[0]
        if in_band.size == 0:
            # fall back to the single row nearest the band centre
            centre = 0.5 * (lat_band[0] + lat_band[1])
            in_band = np.array([int(np.nanargmin(np.abs(lat_y - centre)))])
        moc_band = moc.isel({y_dim: in_band})
    else:
        # no latitude info: fall back to ORCA2 index for ~26.5N
        print("Warning: nav_lat not found; falling back to y-index 94 (~26.5N)")
        moc_band = moc.isel({y_dim: 94})

    # Max over (sub-surface) depth at each latitude
    if depth_dim is not None and depth_dim in moc_band.dims:
        amoc_lat = moc_band.max(dim=depth_dim)
    else:
        amoc_lat = moc_band

    # Average across the latitude band
    if y_dim in amoc_lat.dims:
        amoc_ts = amoc_lat.mean(dim=y_dim)
    else:
        amoc_ts = amoc_lat

    # Time average
    time_dim = None
    for dim in amoc_ts.dims:
        if 'time' in dim:
            time_dim = dim
            break
    if time_dim is not None:
        amoc_ts = amoc_ts.mean(dim=time_dim)

    # Collapse any remaining dim (the zonally-integrated x, size 1) to a scalar
    return float(amoc_ts.mean().values)


def _find_moc_variable(moc_ds: xr.Dataset, candidates: list):
    """Find a MOC variable by trying candidate names, then fallback search."""
    for name in candidates:
        if name in moc_ds:
            return name
    for name in moc_ds.data_vars:
        if 'msf' in name.lower() or 'moc' in name.lower():
            return name
    return None


def plot_amoc_streamfunction(moc_ds: xr.Dataset, output_path: str) -> None:
    """
    Plot Atlantic and Global MOC streamfunctions side by side.

    Creates a two-panel depth-latitude contour plot showing the Atlantic
    and Global meridional overturning streamfunctions.

    Args:
        moc_ds: MOC dataset from cdfmoc
        output_path: Path to save the output PNG
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # --- resolve coordinate arrays ---
    depths = -moc_ds['depthw'].values
    lats = moc_ds['nav_lat'].values.squeeze()

    levels = np.arange(-22, 22, 1)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4),
                             layout='constrained', sharey=True)

    # --- Atlantic MOC ---
    atl_var = _find_moc_variable(
        moc_ds, ['zomsfatl', 'zomsf_atl', 'moc_atl'])
    if atl_var is None:
        print("Warning: No Atlantic MOC variable found, skipping plot")
        plt.close(fig)
        return

    atl_yslice = slice(20, 130)  # crop to avoid tripolar artifacts
    atl = moc_ds[atl_var].squeeze(dim='x').mean(dim='time_counter')
    atl_plot = atl.isel(y=atl_yslice).values

    cs = axes[0].contourf(lats[atl_yslice], depths, atl_plot,
                          levels=levels, cmap='RdBu_r', extend='both')
    cl = axes[0].contour(lats[atl_yslice], depths, atl_plot,
                         levels=levels[::2], colors='k',
                         linewidths=0.4, alpha=0.5)
    axes[0].clabel(cl, inline=True, fontsize=8, fmt='%.0f')
    axes[0].set_ylim(5500, 0)
    axes[0].set_xlim(-34, 70)
    axes[0].set_xlabel('Latitude (\u00b0N)', fontsize=13)
    axes[0].set_ylabel('Depth (m)', fontsize=13)
    axes[0].set_title('Atlantic Meridional Overturning Streamfunction',
                      fontsize=11)

    # --- Global MOC ---
    glo_var = _find_moc_variable(
        moc_ds, ['zomsfglo', 'zomsf_glo', 'moc_glo'])
    if glo_var is None:
        print("Warning: No Global MOC variable found, skipping global panel")
        axes[1].set_visible(False)
    else:
        glo_yslice = slice(1, 148)  # full range, skip boundary
        glo = moc_ds[glo_var].squeeze(dim='x').mean(dim='time_counter')
        glo_plot = glo.isel(y=glo_yslice).values

        axes[1].contourf(lats[glo_yslice], depths, glo_plot,
                         levels=levels, cmap='RdBu_r', extend='both')
        cl2 = axes[1].contour(lats[glo_yslice], depths, glo_plot,
                              levels=levels[::2], colors='k',
                              linewidths=0.4, alpha=0.5)
        axes[1].clabel(cl2, inline=True, fontsize=8, fmt='%.0f')
        axes[1].set_xlim(-80, 80)
        axes[1].set_xlabel('Latitude (\u00b0N)', fontsize=13)
        axes[1].set_title('Global Meridional Overturning Streamfunction',
                          fontsize=11)

    fig.colorbar(cs, ax=axes.tolist(), pad=0.02, shrink=0.9, label='Sv')

    fig.savefig(output_path, dpi=200, bbox_inches='tight',
                pil_kwargs={'optimize': True, 'compress_level': 9})
    plt.close(fig)
    print(f"Saved: {output_path}")
