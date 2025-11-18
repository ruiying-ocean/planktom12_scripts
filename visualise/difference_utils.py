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

    ax.set_title(title, fontsize=12)

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


def plot_transect_difference(
    ax: plt.Axes,
    diff_data: xr.DataArray,
    title: str,
    variable: str = None,
    cmap: str = 'RdBu_r',
    symmetric: bool = True,
    add_colorbar: bool = True,
    vmin: float = None,
    vmax: float = None,
    max_depth: float = None
):
    """
    Plot a single transect difference/anomaly.

    Args:
        ax: Matplotlib axes
        diff_data: Difference data to plot (2D: depth x latitude)
        title: Plot title
        variable: Variable name (for getting metadata)
        cmap: Colormap (default: RdBu_r for diverging)
        symmetric: Whether to use symmetric colorbar limits
        add_colorbar: Whether to add colorbar
        vmin: Manual vmin (overrides symmetric)
        vmax: Manual vmax (overrides symmetric)
        max_depth: Maximum depth to plot in meters
    """
    # Mask very small values (likely land/masked regions)
    diff_data_masked = diff_data.where(np.abs(diff_data) > 1e-10)

    # Get colorbar limits
    if vmin is None or vmax is None:
        if symmetric:
            vmin, vmax = get_symmetric_colorbar_limits(diff_data_masked)
        else:
            vmin = float(np.nanpercentile(diff_data_masked.values, 5))
            vmax = float(np.nanpercentile(diff_data_masked.values, 95))

    # Plot - always use pcolormesh for transect data (2D depth x latitude)
    # Squeeze to remove any singleton dimensions first
    diff_data_plot = diff_data_masked.squeeze()

    # Ensure we have exactly 2 dimensions
    if diff_data_plot.ndim != 2:
        print(f"Warning: Difference data has {diff_data_plot.ndim} dimensions: {diff_data_plot.dims}")
        print(f"Shape: {diff_data_plot.shape}")
        # Try to drop any remaining singleton dimensions
        for dim in diff_data_plot.dims:
            if diff_data_plot.sizes[dim] == 1:
                diff_data_plot = diff_data_plot.drop_vars(dim, errors='ignore').squeeze()

    im = diff_data_plot.plot.pcolormesh(
        ax=ax,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        add_colorbar=add_colorbar,
        cbar_kwargs={'label': f'Δ', 'shrink': 0.8, 'pad': 0.02} if add_colorbar else None
    )

    ax.set_title(title, fontsize=12)
    ax.invert_yaxis()

    if max_depth is not None:
        ax.set_ylim(max_depth, 0)

    return im


def plot_three_panel_transect(
    axs: list,
    model_data: xr.DataArray,
    obs_data: xr.DataArray,
    variable: str,
    label_model: str = "Model",
    label_obs: str = "Observations",
    show_ylabel: bool = True,
    show_xlabel: bool = True,
    max_depth: float = None
):
    """
    Create a 3-panel transect comparison: Model | Observations | Difference.

    Args:
        axs: List of 3 axes [ax_model, ax_obs, ax_diff]
        model_data: Model transect data (2D: depth x latitude)
        obs_data: Observation transect data (2D: depth x latitude)
        variable: Variable name
        label_model: Label for model panel
        label_obs: Label for observations panel
        show_ylabel: Whether to show y-axis label
        show_xlabel: Whether to show x-axis label
        max_depth: Maximum depth to plot in meters

    Returns:
        Tuple of (ax_model, ax_obs, ax_diff)
    """
    from map_utils import get_variable_metadata

    # Get metadata
    meta = get_variable_metadata(variable)
    var_name = meta.get('long_name', variable)
    var_unit = meta.get('units', '')
    cmap = meta.get('cmap', 'Spectral_r')

    # Unpack axes
    ax_model, ax_obs, ax_diff = axs

    # Mask very small values
    model_masked = model_data.where(model_data > 1e-10) if model_data is not None else None
    obs_masked = obs_data.where(obs_data > 1e-10) if obs_data is not None else None

    # Calculate common vmax for model and obs
    vmax = None
    if model_masked is not None:
        vmax = float(np.nanpercentile(model_masked.values, 95))
    if obs_masked is not None:
        obs_vmax = float(np.nanpercentile(obs_masked.values, 95))
        vmax = max(vmax, obs_vmax) if vmax is not None else obs_vmax

    # Panel 1: Model
    if model_masked is not None:
        model_masked.plot.pcolormesh(
            ax=ax_model,
            cmap=cmap,
            vmin=0,
            vmax=vmax,
            add_colorbar=True,
            cbar_kwargs={'label': var_unit, 'shrink': 0.8, 'pad': 0.02}
        )
        ax_model.set_title(f"{var_name}\n{label_model}", fontsize=12)
        ax_model.invert_yaxis()
        if max_depth is not None:
            ax_model.set_ylim(max_depth, 0)
    else:
        ax_model.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax_model.transAxes)

    # Panel 2: Observations
    if obs_masked is not None:
        obs_masked.plot.pcolormesh(
            ax=ax_obs,
            cmap=cmap,
            vmin=0,
            vmax=vmax,
            add_colorbar=True,
            cbar_kwargs={'label': var_unit, 'shrink': 0.8, 'pad': 0.02}
        )
        ax_obs.set_title(f"{label_obs}", fontsize=12)
        ax_obs.invert_yaxis()
        if max_depth is not None:
            ax_obs.set_ylim(max_depth, 0)
    else:
        ax_obs.text(0.5, 0.5, 'No observations', ha='center', va='center', transform=ax_obs.transAxes)

    # Panel 3: Difference
    if model_data is not None and obs_data is not None:
        try:
            # Interpolate observation data to model's depth levels
            model_for_diff = model_data.copy()
            obs_for_diff = obs_data.copy()

            # Determine which is the depth dimension for each dataset
            model_depth_dim = 'deptht' if 'deptht' in model_for_diff.dims else 'depth'
            obs_depth_dim = 'depth' if 'depth' in obs_for_diff.dims else 'deptht'

            # If they have different depth dimensions, interpolate obs to model levels
            if model_depth_dim != obs_depth_dim or not np.array_equal(
                model_for_diff[model_depth_dim].values,
                obs_for_diff[obs_depth_dim].values
            ):
                # Interpolate observation to model depth levels
                obs_for_diff = obs_for_diff.interp(
                    {obs_depth_dim: model_for_diff[model_depth_dim].values},
                    method='linear',
                    kwargs={"fill_value": "extrapolate"}
                )
                # Rename to match model
                if obs_depth_dim != model_depth_dim:
                    obs_for_diff = obs_for_diff.rename({obs_depth_dim: model_depth_dim})

            diff = model_for_diff - obs_for_diff

            # Check if we have valid data
            if diff.count() > 0:
                plot_transect_difference(
                    ax=ax_diff,
                    diff_data=diff,
                    title=f"Difference\n({label_model} - {label_obs})",
                    variable=variable,
                    cmap='RdBu_r',
                    symmetric=True,
                    add_colorbar=True,
                    max_depth=max_depth
                )
            else:
                ax_diff.text(0.5, 0.5, 'No overlapping\ndata',
                           ha='center', va='center', transform=ax_diff.transAxes)
        except Exception as e:
            print(f"Warning: Could not compute difference: {e}")
            ax_diff.text(0.5, 0.5, 'Cannot compute\ndifference',
                       ha='center', va='center', transform=ax_diff.transAxes)
    else:
        ax_diff.text(0.5, 0.5, 'Cannot compute\ndifference',
                     ha='center', va='center', transform=ax_diff.transAxes)

    # Set labels
    if show_ylabel:
        ax_model.set_ylabel('Depth (m)', fontsize=10)
    else:
        ax_model.set_ylabel('')
        ax_obs.set_ylabel('')
        ax_diff.set_ylabel('')

    if show_xlabel:
        ax_model.set_xlabel('Latitude (°N)', fontsize=10)
        ax_obs.set_xlabel('Latitude (°N)', fontsize=10)
        ax_diff.set_xlabel('Latitude (°N)', fontsize=10)
    else:
        ax_model.set_xlabel('')
        ax_obs.set_xlabel('')
        ax_diff.set_xlabel('')

    return ax_model, ax_obs, ax_diff


def plot_multimodel_transect_row(
    axs: list,
    model_transects: list,
    variable: str,
    model_labels: list,
    show_anomaly: bool = True,
    show_ylabel: bool = True,
    show_xlabel: bool = True,
    max_depth: float = None
):
    """
    Create a multi-model transect row: Model A | Model B | [Anomaly].

    Args:
        axs: List of axes (length = n_models or n_models + 1 if anomaly)
        model_transects: List of model transect data (2D: depth x latitude)
        variable: Variable name
        model_labels: List of model names/labels
        show_anomaly: Whether to show anomaly panel (only for 2 models)
        show_ylabel: Whether to show y-axis label
        show_xlabel: Whether to show x-axis label
        max_depth: Maximum depth to plot in meters

    Returns:
        List of axes used
    """
    from map_utils import get_variable_metadata, PHYTO_NAMES, ZOO_NAMES

    n_models = len(model_transects)

    # Get metadata
    meta = get_variable_metadata(variable)
    var_name = meta.get('long_name', variable)

    # Override with PFT names if this is a PFT variable
    if variable in PHYTO_NAMES:
        var_name = PHYTO_NAMES[variable]
    elif variable in ZOO_NAMES:
        var_name = ZOO_NAMES[variable]

    var_unit = meta.get('units', '')
    cmap = meta.get('cmap', 'Spectral_r')

    # Calculate common vmax across all models
    vmax = None
    for transect in model_transects:
        if transect is not None:
            transect_masked = transect.where(transect > 1e-10)
            model_vmax = float(np.nanpercentile(transect_masked.values, 95))
            vmax = max(vmax, model_vmax) if vmax is not None else model_vmax

    # Plot each model
    for model_idx, (transect, label) in enumerate(zip(model_transects, model_labels)):
        ax = axs[model_idx]

        if transect is not None:
            transect_masked = transect.where(transect > 1e-10)

            transect_masked.plot.pcolormesh(
                ax=ax,
                cmap=cmap,
                vmin=0,
                vmax=vmax,
                add_colorbar=True,
                cbar_kwargs={'label': var_unit, 'shrink': 0.8, 'pad': 0.02}
            )

            ax.invert_yaxis()
            if max_depth is not None:
                ax.set_ylim(max_depth, 0)

            # Only show model name as title in first row (when show_ylabel=True)
            if show_ylabel:
                ax.set_title(f"{label}", fontsize=12, fontweight='bold')
            else:
                ax.set_title('')
        else:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)

        # Y-label only on first column - show variable name and depth
        if model_idx == 0:
            ax.set_ylabel(f"{var_name}\nDepth (m)", fontsize=10, fontweight='bold')
        else:
            ax.set_ylabel('')

        # X-label only if requested
        if show_xlabel:
            ax.set_xlabel('Latitude (°N)', fontsize=10)
        else:
            ax.set_xlabel('')

    # Plot anomaly if 2 models
    if show_anomaly and n_models == 2 and model_transects[0] is not None and model_transects[1] is not None:
        ax_diff = axs[2]  # Third axis in the row

        diff = model_transects[1] - model_transects[0]
        plot_transect_difference(
            ax=ax_diff,
            diff_data=diff,
            title="Anomaly (B-A)",
            variable=variable,
            cmap='RdBu_r',
            symmetric=True,
            add_colorbar=True,
            max_depth=max_depth
        )

        ax_diff.set_ylabel('')
        if show_xlabel:
            ax_diff.set_xlabel('Latitude (°N)', fontsize=10)
        else:
            ax_diff.set_xlabel('')

    return axs
