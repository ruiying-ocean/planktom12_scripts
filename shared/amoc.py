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


def plot_amoc_streamfunction(moc_ds: xr.Dataset, output_path: str) -> None:
    """
    Plot the Atlantic MOC streamfunction as a depth-latitude contour plot.

    Args:
        moc_ds: MOC dataset from cdfmoc
        output_path: Path to save the output PNG
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # Find Atlantic MOC variable
    moc_var = None
    for name in ['zomsfatl', 'zomsf_atl', 'moc_atl']:
        if name in moc_ds:
            moc_var = name
            break
    if moc_var is None:
        for name in moc_ds.data_vars:
            if 'msf' in name.lower() or 'moc' in name.lower():
                moc_var = name
                break

    if moc_var is None:
        print(f"Warning: No MOC variable found, skipping streamfunction plot")
        return

    moc = moc_ds[moc_var]

    # Time average if time dimension exists
    for dim in list(moc.dims):
        if 'time' in dim:
            moc = moc.mean(dim=dim)

    # Remove any singleton x dimension
    for dim in list(moc.dims):
        if dim.startswith('x') or dim == 'nav_x':
            moc = moc.squeeze(dim=dim)

    # Get 2D array (depth x lat)
    data = moc.values.squeeze()
    if data.ndim != 2:
        print(f"Warning: MOC data has {data.ndim} dimensions after squeezing, expected 2")
        return

    # Crop to useful latitude range (y=20:130 in ORCA2 ≈ 30S-80N)
    data = data[:, 20:130]

    # Determine axis values
    # Approximate ORCA2 latitudes for y indices 20-130
    lats = np.linspace(-30, 80, data.shape[1])

    # Get depth values if available
    depth_dim = None
    for dim in moc.dims:
        if 'depth' in dim or dim == 'z' or dim == 'nav_lev':
            depth_dim = dim
            break

    if depth_dim is not None and depth_dim in moc.coords:
        depths = moc.coords[depth_dim].values
    else:
        depths = np.arange(data.shape[0])

    # Extract AMOC value at 26N for annotation
    y_26n = int((26.5 - (-30)) / (80 - (-30)) * data.shape[1])
    amoc_value = np.nanmax(data[:, y_26n])

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Symmetric colorbar limits
    vmax = np.nanpercentile(np.abs(data), 98)
    levels = np.linspace(-vmax, vmax, 21)

    cf = ax.contourf(lats, depths, data, levels=levels, cmap='RdBu_r', extend='both')
    ax.contour(lats, depths, data, levels=levels, colors='k', linewidths=0.3, alpha=0.5)

    # Zero contour
    ax.contour(lats, depths, data, levels=[0], colors='k', linewidths=1.0)

    ax.set_ylabel('Depth (m)')
    ax.set_xlabel('Latitude')
    ax.set_title('Atlantic Meridional Overturning Streamfunction')
    ax.invert_yaxis()

    cbar = fig.colorbar(cf, ax=ax, orientation='vertical', pad=0.02, shrink=0.9)
    cbar.set_label('Sv')

    # Annotate AMOC value
    ax.annotate(
        f'AMOC at 26.5N: {amoc_value:.1f} Sv',
        xy=(26.5, depths[0]), xytext=(0.02, 0.02),
        textcoords='axes fraction',
        fontsize=10, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8)
    )

    # Mark 26.5N line
    ax.axvline(26.5, color='k', linestyle='--', alpha=0.5, linewidth=0.8)

    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_path}")
