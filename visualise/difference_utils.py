#!/usr/bin/env python3
"""
Shared utilities for calculating and plotting differences/anomalies.
Used for model-model comparisons, model-observation comparisons, and climatologies.
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from typing import Optional, Tuple

from map_utils import OceanMapPlotter, get_variable_metadata


def calculate_difference(
    data1: xr.DataArray,
    data2: xr.DataArray,
    relative: bool = False
) -> Optional[xr.DataArray]:
    """
    Calculate difference between two datasets.

    Args:
        data1: First dataset (e.g., model or current state)
        data2: Second dataset (e.g., reference, observation, or climatology)
        relative: If True, calculate relative difference (data1 - data2) / data2 * 100

    Returns:
        Difference or relative difference as DataArray
    """
    if data1 is None or data2 is None:
        return None

    try:
        if relative:
            # Avoid division by zero
            diff = ((data1 - data2) / data2.where(data2 != 0)) * 100
        else:
            diff = data1 - data2

        return diff
    except Exception as e:
        print(f"Error calculating difference: {e}")
        return None


def calculate_surface_difference(
    data1: xr.DataArray,
    data2: xr.DataArray,
    depth_index: int = 0,
    relative: bool = False
) -> Optional[xr.DataArray]:
    """
    Calculate surface (or specific depth level) difference between two 3D datasets.

    Args:
        data1: First 3D dataset
        data2: Second 3D dataset
        depth_index: Depth level to extract (default: 0 for surface)
        relative: If True, calculate relative difference

    Returns:
        2D difference at specified depth level
    """
    if data1 is None or data2 is None:
        return None

    # Extract surface or specified depth level if 3D
    if 'deptht' in data1.dims or 'nav_lev' in data1.dims:
        depth_dim = 'deptht' if 'deptht' in data1.dims else 'nav_lev'
        surf1 = data1.isel({depth_dim: depth_index})
        surf2 = data2.isel({depth_dim: depth_index})
    else:
        surf1 = data1
        surf2 = data2

    return calculate_difference(surf1, surf2, relative=relative)


def calculate_rmse(
    data1: xr.DataArray,
    data2: xr.DataArray,
    mask: Optional[xr.DataArray] = None
) -> float:
    """
    Calculate Root Mean Square Error between two datasets.

    Args:
        data1: First dataset
        data2: Second dataset
        mask: Optional land mask (True for ocean, False for land)

    Returns:
        RMSE value
    """
    if data1 is None or data2 is None:
        return np.nan

    diff = data1 - data2

    if mask is not None:
        diff = diff.where(mask)

    rmse = np.sqrt((diff ** 2).mean(skipna=True))
    return float(rmse.values)


def calculate_bias(
    data1: xr.DataArray,
    data2: xr.DataArray,
    mask: Optional[xr.DataArray] = None
) -> float:
    """
    Calculate mean bias between two datasets.

    Args:
        data1: First dataset
        data2: Second dataset
        mask: Optional land mask

    Returns:
        Mean bias value
    """
    if data1 is None or data2 is None:
        return np.nan

    diff = data1 - data2

    if mask is not None:
        diff = diff.where(mask)

    bias = diff.mean(skipna=True)
    return float(bias.values)


def get_symmetric_colorbar_limits(
    data: xr.DataArray,
    percentile: float = 95
) -> Tuple[float, float]:
    """
    Get symmetric colorbar limits for diverging colormaps.

    Args:
        data: Data array
        percentile: Percentile to use for limit calculation

    Returns:
        Tuple of (vmin, vmax) with symmetric limits
    """
    abs_max = np.nanpercentile(np.abs(data.values), percentile)
    return -abs_max, abs_max


def plot_difference_map(
    plotter: OceanMapPlotter,
    diff_data: xr.DataArray,
    ax: plt.Axes,
    title: str,
    variable: str = None,
    cmap: str = 'RdBu_r',
    symmetric: bool = True,
    add_colorbar: bool = True,
    vmin: float = None,
    vmax: float = None
):
    """
    Plot a single difference/anomaly map.

    Args:
        plotter: OceanMapPlotter instance
        diff_data: Difference data to plot
        ax: Matplotlib axes
        title: Plot title
        variable: Variable name (for getting metadata)
        cmap: Colormap (default: RdBu_r for diverging)
        symmetric: Whether to use symmetric colorbar limits
        add_colorbar: Whether to add colorbar
        vmin: Manual vmin (overrides symmetric)
        vmax: Manual vmax (overrides symmetric)
    """
    # Apply land mask
    diff_data = plotter.apply_mask(diff_data)

    # Get colorbar limits
    if vmin is None or vmax is None:
        if symmetric:
            vmin, vmax = get_symmetric_colorbar_limits(diff_data)
        else:
            vmin = float(np.nanpercentile(diff_data.values, 5))
            vmax = float(np.nanpercentile(diff_data.values, 95))

    # Plot
    im = plotter.plot_variable(
        ax=ax,
        data=diff_data,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        add_colorbar=False
    )

    ax.set_title(title, fontsize=10)

    if add_colorbar:
        # Get units from metadata if variable name provided
        if variable:
            meta = get_variable_metadata(variable)
            units = meta.get('units', '')
        else:
            units = ''

        cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.05, shrink=0.8)
        cbar.set_label(f'Difference ({units})' if units else 'Difference', fontsize=9)

    return im


def plot_comparison_panel(
    plotter: OceanMapPlotter,
    data1: xr.DataArray,
    data2: xr.DataArray,
    variable: str,
    label1: str = "Model",
    label2: str = "Reference",
    output_path: str = None,
    show_difference: bool = True,
    show_stats: bool = True
):
    """
    Create a 3-panel comparison plot: data1, data2, and difference.

    Args:
        plotter: OceanMapPlotter instance
        data1: First dataset (e.g., model)
        data2: Second dataset (e.g., observation/reference)
        variable: Variable name
        label1: Label for first dataset
        label2: Label for second dataset
        output_path: Where to save the plot
        show_difference: Whether to show difference panel
        show_stats: Whether to show RMSE/bias statistics
    """
    n_cols = 3 if show_difference else 2

    fig, axs = plotter.create_subplot_grid(
        nrows=1,
        ncols=n_cols,
        projection=ccrs.PlateCarree(),
        figsize=(5 * n_cols, 4)
    )

    # Get metadata
    meta = get_variable_metadata(variable)

    # Apply mask
    data1 = plotter.apply_mask(data1)
    data2 = plotter.apply_mask(data2)

    # Calculate common vmin/vmax for data panels
    combined_data = xr.concat([data1, data2], dim='temp')
    vmin = float(np.nanpercentile(combined_data.values, 5))
    vmax = float(np.nanpercentile(combined_data.values, 95))

    # Panel 1: First dataset
    im1 = plotter.plot_variable(
        ax=axs[0],
        data=data1,
        cmap=meta['cmap'],
        vmin=vmin,
        vmax=vmax,
        add_colorbar=False
    )
    axs[0].set_title(f"{label1} - {meta['long_name']}", fontsize=10)
    cbar1 = plt.colorbar(im1, ax=axs[0], orientation='horizontal', pad=0.05, shrink=0.8)
    cbar1.set_label(meta['units'], fontsize=9)

    # Panel 2: Second dataset
    plotter.plot_variable(
        ax=axs[1],
        data=data2,
        cmap=meta['cmap'],
        vmin=vmin,
        vmax=vmax,
        add_colorbar=False
    )
    axs[1].set_title(f"{label2} - {meta['long_name']}", fontsize=10)
    cbar2 = plt.colorbar(im1, ax=axs[1], orientation='horizontal', pad=0.05, shrink=0.8)
    cbar2.set_label(meta['units'], fontsize=9)

    # Panel 3: Difference
    if show_difference:
        diff = calculate_difference(data1, data2)

        # Calculate statistics
        if show_stats:
            rmse = calculate_rmse(data1, data2, mask=~np.isnan(data1))
            bias = calculate_bias(data1, data2, mask=~np.isnan(data1))
            diff_title = f"Difference ({label1} - {label2})\nRMSE: {rmse:.3f}, Bias: {bias:.3f}"
        else:
            diff_title = f"Difference ({label1} - {label2})"

        plot_difference_map(
            plotter=plotter,
            diff_data=diff,
            ax=axs[2],
            title=diff_title,
            variable=variable,
            cmap='RdBu_r',
            symmetric=True,
            add_colorbar=True
        )

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {output_path}")
    else:
        return fig


def calculate_climatology(
    data: xr.DataArray,
    time_dim: str = 'time_counter'
) -> xr.DataArray:
    """
    Calculate climatology from time series data.

    Args:
        data: Time series data
        time_dim: Name of time dimension

    Returns:
        Climatological mean
    """
    if time_dim in data.dims:
        return data.mean(dim=time_dim)
    else:
        return data


def calculate_anomaly(
    data: xr.DataArray,
    climatology: xr.DataArray,
    time_dim: str = 'time_counter'
) -> xr.DataArray:
    """
    Calculate anomaly from climatology.

    Args:
        data: Time series data
        climatology: Climatological mean
        time_dim: Name of time dimension

    Returns:
        Anomaly (data - climatology)
    """
    return data - climatology
