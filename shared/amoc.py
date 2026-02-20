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


def extract_amoc_26n(moc_ds: xr.Dataset, y_index: int = 94) -> float:
    """
    Extract AMOC strength at 26N (annual mean, max over depth).

    The AMOC strength is defined as the maximum of the Atlantic MOC
    streamfunction over depth at approximately 26.5N, matching the
    RAPID array latitude.

    Args:
        moc_ds: MOC dataset from cdfmoc (from read_moc_file)
        y_index: y-grid index corresponding to ~26.5N in ORCA2 (default 94)

    Returns:
        AMOC strength in Sv (annual mean of max-over-depth at 26N)
    """
    # cdfmoc output variable names vary by version
    # Common names: zomsfatl (Atlantic MOC), zomsfglo (Global MOC)
    moc_var = None
    for name in ['zomsfatl', 'zomsf_atl', 'moc_atl']:
        if name in moc_ds:
            moc_var = name
            break

    if moc_var is None:
        # Fall back to first variable that looks like a streamfunction
        for name in moc_ds.data_vars:
            if 'msf' in name.lower() or 'moc' in name.lower():
                moc_var = name
                break

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

    depth_dim = None
    for dim in moc.dims:
        if 'depth' in dim or dim == 'z' or dim == 'nav_lev':
            depth_dim = dim
            break
    if depth_dim is None:
        spatial_dims = [d for d in moc.dims if d not in ('time_counter', 'time')]
        if len(spatial_dims) >= 1:
            depth_dim = spatial_dims[-1]

    # Select latitude index for 26N
    if y_dim is not None:
        moc_26n = moc.isel({y_dim: y_index})
    else:
        raise ValueError("Cannot identify y-dimension in MOC dataset")

    # Max over depth
    if depth_dim is not None and depth_dim in moc_26n.dims:
        amoc_ts = moc_26n.max(dim=depth_dim)
    else:
        amoc_ts = moc_26n

    # Time average
    time_dim = None
    for dim in amoc_ts.dims:
        if 'time' in dim:
            time_dim = dim
            break

    if time_dim is not None:
        amoc_value = float(amoc_ts.mean(dim=time_dim).values)
    else:
        amoc_value = float(amoc_ts.values)

    return amoc_value


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

    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_path}")
