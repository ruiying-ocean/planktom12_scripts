#!/usr/bin/env python3

import calendar
import sys
import os
import pathlib
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib
from matplotlib import gridspec

# Use a backend that doesn't require a display
matplotlib.use("Agg")

# Import logging utilities from parent directory
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from logging_utils import print_header, print_info, print_warning, print_error, print_success

# Import shared utilities from parent directory
from timeseries_util import ConfigLoader, DataFileLoader, ObservationData, PlotStyler

# Load configuration using shared utility
CONFIG = ConfigLoader.load_config()


@dataclass
class ModelConfig:
    """Configuration for a single model"""

    name: str
    description: str
    start_year: int
    to_year: int
    model_dir: str

    @property
    def from_year(self):
        """Analysis starts from beginning of run (no spin-up skip by default)"""
        return self.start_year

    @property
    def end_year(self):
        """Analysis ends at the same year as to_year"""
        return self.to_year

    @property
    def label(self):
        return f"{self.name} : {self.description.replace('_', ' ')}"

    def get_year_range_indices(self, actual_years=None):
        """Calculate indices based on actual data or use default calculation"""
        if actual_years is not None and len(actual_years) > 0:
            try:
                from_idx = np.where(actual_years == self.from_year)[0]
                to_idx = np.where(actual_years == self.to_year)[0]

                if len(from_idx) > 0 and len(to_idx) > 0:
                    return int(from_idx[0]), int(to_idx[0]) + 1
                else:
                    min_year = int(actual_years[0])
                    max_year = int(actual_years[-1])

                    if self.from_year < min_year:
                        from_idx = 0
                    elif self.from_year > max_year:
                        return None, None
                    else:
                        from_idx = self.from_year - min_year

                    if self.to_year > max_year:
                        to_idx = len(actual_years)
                    elif self.to_year < min_year:
                        return None, None
                    else:
                        to_idx = self.to_year - min_year + 1

                    return from_idx, to_idx
            except Exception as e:
                print_warning(f"Error calculating indices from actual years: {e}")

        a = 2 + (self.from_year - self.start_year)
        b = a + (self.to_year - self.from_year) + 1
        return a, b

    @property
    def year_range_indices(self):
        """Legacy property for compatibility"""
        return self.get_year_range_indices()

    def get_monthly_index(self, actual_data_length=None):
        """Calculate monthly index with validation"""
        if actual_data_length is not None and actual_data_length >= 12:
            return actual_data_length - 12
        return 3 + (self.to_year - self.start_year) * 12

    @property
    def monthly_index(self):
        """Legacy property for compatibility"""
        return self.get_monthly_index()


# Initialize global PlotStyler for consistent styling across all plotters
PLOT_STYLER = PlotStyler(CONFIG) if CONFIG else None

# Apply matplotlib styling globally
if PLOT_STYLER:
    PLOT_STYLER.apply_style()

# Use colors from multimodel_palette in config, or fall back to standard colors
if CONFIG:
    COLORS = CONFIG.get("colors", {}).get("multimodel_palette", PLOT_STYLER.colors if PLOT_STYLER else [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
    ])
    USE_CONSTRAINED_LAYOUT = CONFIG.get("layout", {}).get("use_constrained_layout", True)
    # Use same subplot sizing as single-model for consistency
    SUBPLOT_WIDTH = CONFIG.get("layout", {}).get("subplot_width", 2.5)
    SUBPLOT_HEIGHT = CONFIG.get("layout", {}).get("subplot_height", 2.0)
else:
    COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
              "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
    USE_CONSTRAINED_LAYOUT = True
    SUBPLOT_WIDTH = 2.5
    SUBPLOT_HEIGHT = 2.0

# Direct access to styler properties (no wrapper class)
TITLE_FONTSIZE = PLOT_STYLER.font_title if PLOT_STYLER else 8
LABEL_FONTSIZE = PLOT_STYLER.font_axis_label if PLOT_STYLER else 10
LINE_WIDTH = PLOT_STYLER.linewidth if PLOT_STYLER else 1.5

# Observation styling from shared styler
HATCH_STYLE = {
    "color": PLOT_STYLER.obs_hatch_color if PLOT_STYLER else "k",
    "alpha": PLOT_STYLER.obs_hatch_alpha if PLOT_STYLER else 0.15,
    "fill": PLOT_STYLER.obs_hatch_fill if PLOT_STYLER else False,
    "hatch": PLOT_STYLER.obs_hatch_pattern if PLOT_STYLER else "///"
}

LINE_STYLE = {
    "color": PLOT_STYLER.obs_line_color if PLOT_STYLER else "k",
    "linestyle": PLOT_STYLER.obs_line_linestyle if PLOT_STYLER else "--",
    "alpha": PLOT_STYLER.obs_line_alpha if PLOT_STYLER else 0.8,
    "linewidth": PLOT_STYLER.obs_line_linewidth if PLOT_STYLER else 1.5
}


def setup_axes(axes):
    """Apply consistent styling to all axes in a figure."""
    for ax in axes:
        ax.grid(True)
        ax.tick_params(axis="both", which="major", labelsize=LABEL_FONTSIZE - 1)
        # Keep all 4 spines visible (full frame border) to match single-model style
        ax.xaxis.label.set_size(LABEL_FONTSIZE)
        ax.yaxis.label.set_size(LABEL_FONTSIZE)


class DataLoader:
    """Handles loading data from CSV files"""

    @staticmethod
    def load_model_configs(csv_path, default_model_dir=None):
        """
        Load model configurations from CSV.

        Args:
            csv_path: Path to CSV file
            default_model_dir: Default base directory if 'location' column is missing or empty
                           (defaults to ~/scratch/ModelRuns if not specified)
        """
        import os
        if default_model_dir is None:
            default_model_dir = os.path.expanduser("~/scratch/ModelRuns")

        df = pd.read_csv(csv_path)

        # Check if location column exists
        has_location = "location" in df.columns

        return [
            ModelConfig(
                name=row["model_id"],
                description=row["description"],
                start_year=row["start_year"],
                to_year=row["to_year"],
                model_dir=row["location"] if has_location and pd.notna(row.get("location")) else default_model_dir,
            )
            for _, row in df.iterrows()
        ]

    @staticmethod
    def load_analyser_data(model_config, data_type, frequency="annual"):
        """Load analyser data using shared utility."""
        base_dir = pathlib.Path(model_config.model_dir)
        df = DataFileLoader.read_analyser_file(base_dir, model_config.name, data_type, frequency)

        if df is None:
            print_warning(f"File not found for {model_config.name}/{data_type}/{frequency}")
            return None

        if df.empty:
            print_warning(f"Empty dataframe loaded for {model_config.name}/{data_type}/{frequency}")
            return None

        print_info(f"Loaded {model_config.name}/{data_type}/{frequency}")
        return df

    @staticmethod
    def safe_load_column(df, column, indices, validate_length=True):
        if df is None:
            return None
        try:
            if column not in df.columns:
                return None
            data = df[column]
            if len(data) == 0:
                return None
            if indices[0] is None or indices[1] is None:
                return None

            if validate_length and indices[1] > len(data):
                if indices[0] >= len(data):
                    return None
                return data[indices[0] : len(data)].to_numpy().astype(float)

            return data[indices[0] : indices[1]].to_numpy().astype(float)
        except Exception:
            return None

    @staticmethod
    def get_actual_years(df):
        if df is None or "year" not in df.columns:
            return None
        try:
            return df["year"].to_numpy().astype(float)
        except Exception:
            return None


class PlotGenerator:
    """Base class for generating plots"""

    def __init__(self, models: List[ModelConfig], save_dir: str):
        self.models = models
        self.save_dir = save_dir
        pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)
        # Font family is already set by PLOT_STYLER.apply_style() at module level

    def add_legend(self, fig, location="top", **kwargs):
        """
        Add a figure-level legend with padding so it doesn't collide with tick labels.
        """
        legend_kwargs = {
            "fontsize": LABEL_FONTSIZE - 2,
            "frameon": True,
            "borderaxespad": 0.6,
        }
        legend_kwargs.update(kwargs)

        # Collect handles/labels from all axes
        handles, labels = [], []
        for ax in fig.get_axes():
            h, l = ax.get_legend_handles_labels()
            handles.extend(h)
            labels.extend(l)

        # Remove duplicate labels/handles while preserving one entry per model
        by_label = dict(zip(labels, handles))
        if not by_label:
            return

        legend_kwargs.setdefault("ncol", min(5, max(1, len(by_label))))
        legend_kwargs.setdefault("bbox_transform", fig.transFigure)

        # Choose a placement that avoids the x tick labels
        if location == "bottom":
            legend_kwargs.setdefault("loc", "upper center")
            legend_kwargs.setdefault("bbox_to_anchor", (0.5, -0.12))
        elif location == "right":
            legend_kwargs.setdefault("loc", "center left")
            legend_kwargs.setdefault("bbox_to_anchor", (1.02, 0.5))
        else:  # top (default)
            legend_kwargs.setdefault("loc", "lower center")
            legend_kwargs.setdefault("bbox_to_anchor", (0.5, 1.02))

        fig.legend(by_label.values(), by_label.keys(), **legend_kwargs)

        # Add a little breathing room when the legend sits outside the axes.
        if USE_CONSTRAINED_LAYOUT:
            fig.set_constrained_layout_pads(w_pad=0.02, h_pad=0.02, wspace=0.05, hspace=0.05)
        elif location == "bottom":
            fig.subplots_adjust(bottom=max(fig.subplotpars.bottom, 0.18))
        elif location == "top":
            fig.subplots_adjust(top=min(fig.subplotpars.top, 0.9))

    def save_figure(self, fig, filename):
        """
        Save figure with config-based DPI and format.
        """
        # Get DPI and format from config
        dpi = CONFIG.get("figure", {}).get("dpi", 300) if CONFIG else 300
        fmt = CONFIG.get("figure", {}).get("format", "png") if CONFIG else "png"

        # Replace file extension with config format
        base_name = filename.rsplit(".", 1)[0]
        filename = f"{base_name}.{fmt}"

        filepath = f"{self.save_dir}/{filename}"
        fig.savefig(filepath, dpi=dpi, bbox_inches="tight")
        print_success(f"Created {filename}")
        plt.close(fig)

    def plot_all_models(self, fig, axes, plot_func):
        successful_plots = 0
        for i, model in enumerate(self.models):
            color = COLORS[i % len(COLORS)]
            try:
                plot_func(model, axes, color)
                successful_plots += 1
            except Exception as e:
                print_error(f"Plotting {model.name}: {e}")
                import traceback

                if hasattr(self, "debug") and self.debug:
                    traceback.print_exc()

        if successful_plots == 0:
            print_warning(
                f"No models were successfully plotted for {self.__class__.__name__}"
            )
        else:
            print_info(f"Successfully plotted {successful_plots}/{len(self.models)} models")


class GlobalSummaryPlotter(PlotGenerator):
    """Generates global summary plots"""

    def generate(self):
        # Use 3x3 grid to match single-model (9 panels)
        fig, axes = plt.subplots(
            3, 3, figsize=(3 * SUBPLOT_WIDTH, 3 * SUBPLOT_HEIGHT), sharex=True,
            constrained_layout=USE_CONSTRAINED_LAYOUT
        )
        axes = axes.flatten()

        setup_axes(axes)

        self.plot_all_models(fig, axes, self._plot_model)
        self._add_observational_data(axes)

        self.add_legend(fig)
        self.save_figure(fig, "multimodel_summary_global.png")

    def _plot_model(self, model, axes, color):
        sur_data = DataLoader.load_analyser_data(model, "sur", "annual")
        if sur_data is None:
            return

        actual_years = DataLoader.get_actual_years(sur_data)
        indices = model.get_year_range_indices(actual_years)
        if indices[0] is None:
            return

        vol_data = DataLoader.load_analyser_data(model, "vol", "annual")
        lev_data = DataLoader.load_analyser_data(model, "lev", "annual")
        ave_data = DataLoader.load_analyser_data(model, "ave", "annual")

        year = DataLoader.safe_load_column(sur_data, "year", indices)
        if year is None:
            return

        # Calculate PROCACO3 and EXPCACO3
        proara = DataLoader.safe_load_column(vol_data, "proara", indices)
        prococ = DataLoader.safe_load_column(vol_data, "prococ", indices)
        procaco3 = None
        if proara is not None and prococ is not None:
            procaco3 = proara + prococ

        expara = DataLoader.safe_load_column(lev_data, "ExpARA", indices)
        expco3 = DataLoader.safe_load_column(lev_data, "ExpCO3", indices)
        expcaco3 = None
        if expara is not None and expco3 is not None:
            expcaco3 = expara + expco3

        plot_configs = [
            (axes[0], sur_data, "Cflx", "Surface Carbon Flux", "PgC/yr", False, None),
            (axes[1], ave_data, "TChl", "Surface Chlorophyll", "μg Chl/L", True, None),
            (axes[2], vol_data, "PPT", "Primary Production", "PgC/yr", False, None),
            (axes[3], lev_data, "EXP", "Export at 100m", "PgC/yr", False, None),
            (axes[4], lev_data, "EXP1000", "Export at 1000m", "PgC/yr", False, None),
            (axes[5], None, None, "CaCO₃ Production", "PgC/yr", False, procaco3),
            (axes[6], None, None, "CaCO₃ Export at 100m", "PgC/yr", False, expcaco3),
            (axes[7], vol_data, "probsi", "Silica Production", "Tmol/yr", False, None),
            (axes[8], lev_data, "sinksil", "Silica Export at 100m", "Tmol/yr", False, None),
        ]

        for idx, (ax, data_df, column, title, ylabel, add_label, custom_data) in enumerate(plot_configs):
            if custom_data is not None:
                # Use custom calculated data
                if len(custom_data) == len(year):
                    label = model.label if add_label else None
                    ax.plot(year, custom_data, color=color, label=label, linewidth=LINE_WIDTH)
                    ax.set_title(title, fontsize=TITLE_FONTSIZE, fontweight='bold', pad=5)
                    ax.set_ylabel(ylabel)
                    # Add xlabel only to bottom row (indices 6, 7, 8 in 3x3 grid)
                    if idx >= 6:
                        ax.set_xlabel("Year", fontweight='bold')
            elif data_df is not None and column is not None:
                values = DataLoader.safe_load_column(data_df, column, indices)
                if values is not None and len(values) == len(year):
                    label = model.label if add_label else None
                    ax.plot(year, values, color=color, label=label, linewidth=LINE_WIDTH)
                    ax.set_title(title, fontsize=TITLE_FONTSIZE, fontweight='bold', pad=5)
                    ax.set_ylabel(ylabel)
                    # Add xlabel only to bottom row (indices 6, 7, 8 in 3x3 grid)
                    if idx >= 6:
                        ax.set_xlabel("Year", fontweight='bold')

    def _add_observational_data(self, axes):
        # Map to shared ObservationData keys (now with 9 panels)
        obs_mapping = {
            1: ("TChl", "line"),
            2: ("PPT", "span"),
            3: ("EXP", "span"),
            4: None,  # EXP1000 - no observation
            5: ("PROCACO3", "span"),
            6: ("EXPCACO3", "span"),
            7: ("probsi", "span"),
            8: ("SI_FLX", "span")
        }
        for i in range(1, 9):
            if obs_mapping[i] is not None:
                obs_key, obs_type = obs_mapping[i]
                if obs_key in ObservationData.GLOBAL:
                    obs = ObservationData.GLOBAL[obs_key]
                    ax = axes[i]
                    if obs.get("type") == "line" or obs_type == "line":
                        ax.axhline(obs["value"], **LINE_STYLE)
                    else:
                        ax.axhspan(obs["min"], obs["max"], **HATCH_STYLE)


class GlobalSummaryNormalizedPlotter(PlotGenerator):
    """Generates normalized global summary plots using first 10 years as baseline"""

    def generate(self):
        fig, axes = plt.subplots(
            3, 3, figsize=(3 * SUBPLOT_WIDTH, 3 * SUBPLOT_HEIGHT), sharex=True,
            constrained_layout=USE_CONSTRAINED_LAYOUT
        )
        axes = axes.flatten()

        setup_axes(axes)

        self.plot_all_models(fig, axes, self._plot_model)

        self.add_legend(fig)
        self.save_figure(fig, "multimodel_summary_global_normalized.png")

    @staticmethod
    def _normalize_series(values: np.ndarray) -> np.ndarray:
        """Return anomalies relative to the mean of the first decade (or available years)."""
        if len(values) == 0:
            return values
        baseline_years = min(10, len(values))
        baseline = np.mean(values[:baseline_years])
        return values - baseline

    def _plot_model(self, model, axes, color):
        sur_data = DataLoader.load_analyser_data(model, "sur", "annual")
        if sur_data is None:
            return

        actual_years = DataLoader.get_actual_years(sur_data)
        indices = model.get_year_range_indices(actual_years)
        if indices[0] is None:
            return

        vol_data = DataLoader.load_analyser_data(model, "vol", "annual")
        lev_data = DataLoader.load_analyser_data(model, "lev", "annual")
        ave_data = DataLoader.load_analyser_data(model, "ave", "annual")

        year = DataLoader.safe_load_column(sur_data, "year", indices)
        if year is None:
            return

        # Derived products used in multiple panels
        proara = DataLoader.safe_load_column(vol_data, "proara", indices)
        prococ = DataLoader.safe_load_column(vol_data, "prococ", indices)
        procaco3 = None
        if proara is not None and prococ is not None:
            procaco3 = proara + prococ

        expara = DataLoader.safe_load_column(lev_data, "ExpARA", indices)
        expco3 = DataLoader.safe_load_column(lev_data, "ExpCO3", indices)
        expcaco3 = None
        if expara is not None and expco3 is not None:
            expcaco3 = expara + expco3

        plot_configs = [
            (axes[0], sur_data, "Cflx", None, "Cflx anomaly", "PgC/yr", True),
            (axes[1], ave_data, "TChl", None, "TChl anomaly", "μg Chl/L", False),
            (axes[2], vol_data, "PPT", None, "PPT anomaly", "PgC/yr", False),
            (axes[3], lev_data, "EXP", None, "Exp@100m anomaly", "PgC/yr", False),
            (axes[4], lev_data, "EXP1000", None, "Exp@1000m anomaly", "PgC/yr", False),
            (axes[5], None, None, procaco3, "CaCO3 prod anomaly", "PgC/yr", False),
            (axes[6], None, None, expcaco3, "CaCO3 exp anomaly", "PgC/yr", False),
            (axes[7], vol_data, "probsi", None, "Si prod anomaly", "Tmol/yr", False),
            (axes[8], lev_data, "sinksil", None, "Si exp anomaly", "Tmol/yr", False),
        ]

        for idx, (ax, data_df, column, custom_data, title, ylabel, add_label) in enumerate(plot_configs):
            values = None
            if custom_data is not None:
                values = custom_data
            elif data_df is not None and column is not None:
                values = DataLoader.safe_load_column(data_df, column, indices)

            if values is None or len(values) != len(year):
                continue

            normalized_values = self._normalize_series(values)
            label = model.label if add_label else None
            ax.plot(
                year,
                normalized_values,
                color=color,
                label=label,
                linewidth=LINE_WIDTH,
            )
            ax.set_title(title, fontsize=TITLE_FONTSIZE, fontweight='bold', pad=5)
            ax.set_ylabel(ylabel)
            # Add xlabel only to bottom row (indices 6, 7, 8 in 3x3 grid)
            if idx >= 6:
                ax.set_xlabel("Year", fontweight='bold')
            ax.axhline(0, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)


class RegionalPlotter(PlotGenerator):
    """Base class for regional plotters"""

    def _create_regional_grid(self, fig):
        # NOTE: This complex, irregular grid layout is best handled with GridSpec.
        # A single call to plt.subplots() cannot create subplots that span
        # multiple rows like this, which is why this method is preserved.
        gs = fig.add_gridspec(5, 2, width_ratios=[2, 3], wspace=0.4)
        ax1 = fig.add_subplot(gs[0:3, 0])
        ax2 = fig.add_subplot(gs[3:5, 0])
        regional_axes = [fig.add_subplot(gs[i, 1]) for i in range(5)]
        return [ax1, ax2] + regional_axes

    def _get_month_names(self, monthly_data):
        # Handle both CSV (month) and TSV (Month) formats
        if "month" in monthly_data.columns:
            months = monthly_data["month"][-12:].to_numpy().astype(int)
        elif "Month" in monthly_data.columns:
            months = monthly_data["Month"][-12:].to_numpy().astype(int)
        else:
            # Fallback: generate months 1-12
            months = list(range(1, 13))
        return [calendar.month_abbr[m if m > 0 else 1] for m in months]

    def _plot_regional_data(self, ax, month_names, data, color, label=None):
        if data is not None and len(data) == 12:
            normalized = data - data[0]
            ax.plot(
                month_names,
                normalized,
                color=color,
                label=label,
                linewidth=LINE_WIDTH,
            )


class CflxPlotter(RegionalPlotter):
    """Generates Cflx summary plots"""

    def generate(self):
        """
        Generates and saves the Cflx summary plot with a balanced, nested grid layout.
        """
        fig = plt.figure(figsize=(3.5 * SUBPLOT_WIDTH, 2 * SUBPLOT_HEIGHT))

        # --- MODIFICATION START ---
        # Use a nested GridSpec for complete control over column layouts.

        # 1. Create a main 1x2 grid to define the left and right columns.
        gs_main = gridspec.GridSpec(1, 2, figure=fig, wspace=0.25, width_ratios=[1, 1])

        # 2. Create a nested grid for the left column (2 rows, 1 column).
        gs_left = gs_main[0].subgridspec(2, 1, hspace=0.15)

        # 3. Create a nested grid for the right column (5 rows, 1 column).
        gs_right = gs_main[1].subgridspec(5, 1, hspace=0.1)

        # 4. Create the axes and add them to a list in the correct order.
        axes = []
        # Add left column plots
        axes.append(fig.add_subplot(gs_left[0]))  # axes[0]: Top-left
        axes.append(fig.add_subplot(gs_left[1]))  # axes[1]: Bottom-left

        # Add right column plots, sharing the x-axis
        right_column_axes = []
        for i in range(5):
            if i == 0:
                ax = fig.add_subplot(gs_right[i])
            else:
                ax = fig.add_subplot(gs_right[i], sharex=right_column_axes[0])
            right_column_axes.append(ax)
        axes.extend(right_column_axes)  # Add to the main list (axes[2] through axes[6])

        # 5. Hide x-axis labels on the upper plots of the right column.
        for ax in right_column_axes[:-1]:
            ax.tick_params(
                axis="x", which="both", bottom=False, top=False, labelbottom=False
            )

        setup_axes(axes)
        # --- MODIFICATION END ---

        self.plot_all_models(fig, axes, self._plot_model)

        self.add_legend(fig)
        self.save_figure(fig, "multimodel_summary_cflx.png")

    def _plot_model(self, model, axes, color):
        sur_annual = DataLoader.load_analyser_data(model, "sur", "annual")
        if sur_annual is None:
            return

        actual_years = DataLoader.get_actual_years(sur_annual)
        indices = model.get_year_range_indices(actual_years)
        if indices[0] is None:
            return

        year = DataLoader.safe_load_column(sur_annual, "year", indices)
        cflx_annual = DataLoader.safe_load_column(sur_annual, "Cflx", indices)

        if year is not None and cflx_annual is not None:
            axes[0].plot(
                year, cflx_annual, color=color, linewidth=LINE_WIDTH
            )
            axes[0].set_title(
                "Surface Cflx (Global) [PgC/yr]", fontsize=TITLE_FONTSIZE
            )

        sur_monthly = DataLoader.load_analyser_data(model, "sur", "monthly")
        if sur_monthly is None:
            return

        monthly_idx = model.get_monthly_index(len(sur_monthly))
        month_names = self._get_month_names(sur_monthly)

        regional_cols = ["Cflx", "Cflx.1", "Cflx.2", "Cflx.3", "Cflx.4", "Cflx.5"]
        labels = [model.label] + [None] * 5
        # Use shorter titles for the regional plots
        titles = ["Global", "45N-90N", "15N-45N", "15S-15N", "45S-15S", "90S-45S"]

        for i, (col, label, title) in enumerate(zip(regional_cols, labels, titles)):
            # The axes list is now structured with axes[1] as the first regional plot.
            ax = axes[i + 1]
            data = DataLoader.safe_load_column(
                sur_monthly, col, (monthly_idx, monthly_idx + 12), validate_length=False
            )
            if data is not None and len(data) == 12:
                self._plot_regional_data(ax, month_names, data, color, label)

            # Use ax.text to place titles to avoid overlapping.
            ax.text(
                0.8,
                0.9,
                f"{title}",
                fontsize=TITLE_FONTSIZE - 2,
                transform=ax.transAxes,
            )


class PFTPlotter(PlotGenerator):
    """Generates PFT summary plots"""

    def generate(self):
        # Use 4x3 grid to match single-model (12 PFT panels)
        fig, axes = plt.subplots(
            4,
            3,
            figsize=(3 * SUBPLOT_WIDTH, 4 * SUBPLOT_HEIGHT),
            sharex=True,
            constrained_layout=USE_CONSTRAINED_LAYOUT
        )
        flat_axes = axes.flatten()

        self._setup_pft_axes(flat_axes)

        self.plot_all_models(fig, flat_axes, self._plot_model)
        self._add_observational_ranges(flat_axes)

        self.add_legend(fig)
        self.save_figure(fig, "multimodel_summary_pfts.png")

    def _setup_pft_axes(self, axes):
        setup_axes(axes)
        # Each PFT will have its own y-axis scale
        # No hardcoded y-limits to allow individual scaling

    def _plot_model(self, model, axes, color):
        int_data = DataLoader.load_analyser_data(model, "int", "annual")
        if int_data is None:
            return

        ## remove VIR column if it exists
        if "VIR" in int_data.columns:
            int_data = int_data.drop(columns=["VIR"])

        actual_years = DataLoader.get_actual_years(int_data)
        indices = model.get_year_range_indices(actual_years)
        if indices[0] is None:
            return

        year = DataLoader.safe_load_column(int_data, "year", indices)
        if year is None:
            return

        pft_mapping = [
            ("PIC", "Picophytoplankton"),
            ("PHA", "Phaeocystis"),
            ("MIX", "Mixotrophs"),
            ("DIA", "Diatoms"),
            ("COC", "Coccolithophores"),
            ("FIX", "Nitrogen Fixers"),
            ("BAC", "Bacteria"),
            ("GEL", "Gelatinous Zooplankton"),
            ("PRO", "Protozooplankton"),
            ("CRU", "Crustaceans"),
            ("PTE", "Pteropods"),
            ("MES", "Mesozooplankton"),
        ]

        for i, (col_name, title) in enumerate(pft_mapping):
            values = DataLoader.safe_load_column(int_data, col_name, indices)
            if values is not None and len(values) == len(year):
                label = model.label if i == 10 else None
                axes[i].plot(
                    year,
                    values,
                    color=color,
                    label=label,
                    linewidth=LINE_WIDTH,
                )
                axes[i].set_title(title, fontsize=TITLE_FONTSIZE, fontweight='bold', pad=5)
                axes[i].set_ylabel("PgC")
                # Add xlabel only to bottom row (indices 9, 10, 11 in 4x3 grid)
                if i >= 9:
                    axes[i].set_xlabel("Year", fontweight='bold')

    def _add_observational_ranges(self, axes):
        obs_indices = {
            0: "PIC",
            1: "PHA",
            2: "MIX",
            3: "DIA",
            4: "COC",
            5: "FIX",
            6: "BAC",
            7: "GEL",
            8: "PRO",
            9: "CRU",  # Changed from MAC to CRU to match shared data
            10: "PTE",
            11: "MES",
        }
        for idx, pft_name in obs_indices.items():
            if pft_name in ObservationData.PFT:
                ranges = ObservationData.PFT[pft_name]
                if ranges["min"] is not None and ranges["max"] is not None:
                    axes[idx].axhspan(
                        ranges["min"], ranges["max"], **HATCH_STYLE
                    )


class PFTNormalizedPlotter(PlotGenerator):
    """Generates normalized/anomaly plots for PFT biomass"""

    def generate(self):
        fig, axes = plt.subplots(
            4,
            3,
            figsize=(3 * SUBPLOT_WIDTH, 4 * SUBPLOT_HEIGHT),
            sharex=True,
            constrained_layout=USE_CONSTRAINED_LAYOUT,
        )
        flat_axes = axes.flatten()

        setup_axes(flat_axes)

        self.plot_all_models(fig, flat_axes, self._plot_model)

        self.add_legend(fig)
        self.save_figure(fig, "multimodel_summary_pfts_normalized.png")

    def _plot_model(self, model, axes, color):
        int_data = DataLoader.load_analyser_data(model, "int", "annual")
        if int_data is None:
            return

        if "VIR" in int_data.columns:
            int_data = int_data.drop(columns=["VIR"])

        actual_years = DataLoader.get_actual_years(int_data)
        indices = model.get_year_range_indices(actual_years)
        if indices[0] is None:
            return

        year = DataLoader.safe_load_column(int_data, "year", indices)
        if year is None:
            return

        pft_mapping = [
            ("PIC", "PIC anomaly"),
            ("PHA", "PHA anomaly"),
            ("MIX", "MIX anomaly"),
            ("DIA", "DIA anomaly"),
            ("COC", "COC anomaly"),
            ("FIX", "FIX anomaly"),
            ("BAC", "BAC anomaly"),
            ("GEL", "GEL anomaly"),
            ("PRO", "PRO anomaly"),
            ("CRU", "CRU anomaly"),
            ("PTE", "PTE anomaly"),
            ("MES", "MES anomaly"),
        ]

        for i, (col_name, title) in enumerate(pft_mapping):
            values = DataLoader.safe_load_column(int_data, col_name, indices)
            if values is None or len(values) != len(year):
                continue

            normalized = GlobalSummaryNormalizedPlotter._normalize_series(values)
            label = model.label if i == 10 else None
            axes[i].plot(year, normalized, color=color, label=label, linewidth=LINE_WIDTH)
            axes[i].set_title(title, fontsize=TITLE_FONTSIZE, fontweight="bold", pad=5)
            axes[i].set_ylabel("PgC anomaly")
            axes[i].axhline(0, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
            if i >= 9:
                axes[i].set_xlabel("Year", fontweight="bold")


class TChlPlotter(RegionalPlotter):
    """Generates TChl summary plots"""

    def generate(self):
        """
        Generates and saves the TChl summary plot with a balanced, nested grid layout.
        """
        fig = plt.figure(figsize=(3.5 * SUBPLOT_WIDTH, 2 * SUBPLOT_HEIGHT))

        # Use a nested GridSpec for complete control over column layouts.
        gs_main = gridspec.GridSpec(1, 2, figure=fig, wspace=0.25, width_ratios=[1, 1])
        gs_left = gs_main[0].subgridspec(2, 1, hspace=0.15)
        gs_right = gs_main[1].subgridspec(5, 1, hspace=0.1)

        axes = []
        axes.append(fig.add_subplot(gs_left[0]))
        axes.append(fig.add_subplot(gs_left[1]))

        right_column_axes = []
        for i in range(5):
            if i == 0:
                ax = fig.add_subplot(gs_right[i])
            else:
                ax = fig.add_subplot(gs_right[i], sharex=right_column_axes[0])
            right_column_axes.append(ax)
        axes.extend(right_column_axes)

        for ax in right_column_axes[:-1]:
            ax.tick_params(
                axis="x", which="both", bottom=False, top=False, labelbottom=False
            )

        setup_axes(axes)

        self.plot_all_models(fig, axes, self._plot_model)
        self._add_observational_data(axes)  # MODIFICATION: This line was added

        self.add_legend(fig)
        self.save_figure(fig, "multimodel_summary_tchl.png")

    def _plot_model(self, model, axes, color):
        ave_annual = DataLoader.load_analyser_data(model, "ave", "annual")
        if ave_annual is None:
            return

        actual_years = DataLoader.get_actual_years(ave_annual)
        indices = model.get_year_range_indices(actual_years)
        if indices[0] is None:
            return

        year = DataLoader.safe_load_column(ave_annual, "year", indices)
        tchl_annual = DataLoader.safe_load_column(ave_annual, "TChl", indices)

        if year is not None and tchl_annual is not None:
            axes[0].plot(
                year, tchl_annual, color=color, linewidth=LINE_WIDTH
            )
            axes[0].set_title(
                "Average TChl (Global) [ug Chl/L]", fontsize=TITLE_FONTSIZE
            )

        ave_monthly = DataLoader.load_analyser_data(model, "ave", "monthly")
        if ave_monthly is None:
            return

        monthly_idx = model.get_monthly_index(len(ave_monthly))
        month_names = self._get_month_names(ave_monthly)

        regional_cols = ["TChl", "TChl.1", "TChl.2", "TChl.3", "TChl.4", "TChl.5"]
        labels = [model.label] + [None] * 5
        titles = ["Global", "45N-90N", "15N-45N", "15S-15N", "45S-15S", "90S-45S"]

        for i, (col, label, title) in enumerate(zip(regional_cols, labels, titles)):
            ax = axes[i + 1]
            data = DataLoader.safe_load_column(
                ave_monthly, col, (monthly_idx, monthly_idx + 12), validate_length=False
            )
            self._plot_regional_data(ax, month_names, data, color, label)
            ax.text(
                0.8,
                0.9,
                f"{title}",
                fontsize=TITLE_FONTSIZE - 2,
                transform=ax.transAxes,
            )

    def _add_observational_data(self, axes):
        """
        MODIFICATION: This entire method was added.
        Plots observational TChl data on the regional summary plots.
        """
        month_names = list(calendar.month_abbr)[1:]
        regions = ["global", "reg1", "reg2", "reg3", "reg4", "reg5"]
        titles = ["Global", "45N-90N", "15N-45N", "15S-15N", "45S-15S", "90S-45S"]

        for i, (region, title) in enumerate(zip(regions, titles)):
            ax = axes[i + 1]
            # Assumes you have a similar data structure for TChl observational data
            data = ObservationData.TCHL_MONTHLY[region]
            normalized = data - data[0]
            label = "data-products" if i == 0 else None
            ax.plot(
                month_names,
                normalized,
                color="k",
                linestyle="dashed",
                label=label,
                linewidth=LINE_WIDTH,
            )
            ax.text(
                0.8,
                0.9,
                f"{title}",
                fontsize=TITLE_FONTSIZE - 2,
                transform=ax.transAxes,
            )


class NutrientPlotter(PlotGenerator):
    """Generates nutrient summary plots"""

    def generate(self):
        # Use 3x3 grid for 7 nutrients (matching single-model)
        fig, axes = plt.subplots(
            3, 3, figsize=(3 * SUBPLOT_WIDTH, 3 * SUBPLOT_HEIGHT), sharex=True,
            constrained_layout=USE_CONSTRAINED_LAYOUT
        )
        flat_axes = axes.flatten()

        setup_axes(flat_axes)

        self.plot_all_models(fig, flat_axes, self._plot_model)
        self._add_observational_data(flat_axes)

        self.add_legend(fig)
        self.save_figure(fig, "multimodel_summary_nutrients.png")

    def _plot_model(self, model, axes, color):
        ave_data = DataLoader.load_analyser_data(model, "ave", "annual")
        if ave_data is None:
            return

        actual_years = DataLoader.get_actual_years(ave_data)
        indices = model.get_year_range_indices(actual_years)
        if indices[0] is None:
            return

        year = DataLoader.safe_load_column(ave_data, "year", indices)
        if year is None:
            return

        plot_configs = [
            ("PO4", axes[0], "Surface Phosphate", "μmol/L", 1 / 122, False),
            ("NO3", axes[1], "Surface Nitrate", "μmol/L", 1, True),
            ("Fer", axes[2], "Surface Iron", "nmol/L", 1000, False),
            ("Si", axes[3], "Surface Silica", "μmol/L", 1, False),
            ("O2", axes[4], "Oxygen at 300m", "μmol/L", 1, False),
            ("Alkalini", axes[5], "Surface Alkalinity", "μmol/L", 1, False),
            ("AOU", axes[6], "AOU at 300m", "μmol/L", 1, False),
        ]

        for idx, (col_name, ax, title, ylabel, scale, add_label) in enumerate(plot_configs):
            values = DataLoader.safe_load_column(ave_data, col_name, indices)
            if values is not None and len(values) == len(year):
                label = model.label if add_label else None
                ax.plot(
                    year,
                    values * scale,
                    color=color,
                    label=label,
                    linewidth=LINE_WIDTH,
                )
                ax.set_title(title, fontsize=TITLE_FONTSIZE, fontweight='bold', pad=5)
                ax.set_ylabel(ylabel)
                # Add xlabel only to bottom row (indices 6, 7, 8 in 3x3 grid)
                if idx >= 6:
                    ax.set_xlabel("Year", fontweight='bold')

    def _add_observational_data(self, axes):
        # Add observation lines for all 7 nutrients
        nutrient_keys = ["PO4", "NO3", "Fer", "Si", "O2", "Alkalini", "AOU"]
        for i, key in enumerate(nutrient_keys):
            if i < len(axes) and key in ObservationData.NUTRIENTS:
                obs_value = ObservationData.NUTRIENTS[key]
                if obs_value is not None:
                    axes[i].axhline(obs_value, **LINE_STYLE)

        for idx in range(len(nutrient_keys), len(axes)):
            axes[idx].set_visible(False)


class PCO2Plotter(RegionalPlotter):
    """Generates pCO2 summary plots"""

    def generate(self):
        """
        Generates and saves the pCO2 summary plot with a balanced, nested grid layout.
        """
        fig = plt.figure(figsize=(3.5 * SUBPLOT_WIDTH, 2 * SUBPLOT_HEIGHT))

        # --- MODIFICATION START ---
        # Use a nested GridSpec for complete control over column layouts.

        # 1. Create a main 1x2 grid to define the left and right columns.
        gs_main = gridspec.GridSpec(1, 2, figure=fig, wspace=0.25, width_ratios=[1, 1])

        # 2. Create a nested grid for the left column (2 rows, 1 column).
        gs_left = gs_main[0].subgridspec(2, 1, hspace=0.15)

        # 3. Create a nested grid for the right column (5 rows, 1 column).
        gs_right = gs_main[1].subgridspec(5, 1, hspace=0.1)

        # 4. Create the axes and add them to a list in the correct order.
        axes = []
        # Add left column plots
        axes.append(fig.add_subplot(gs_left[0]))  # axes[0]: Top-left
        axes.append(fig.add_subplot(gs_left[1]))  # axes[1]: Bottom-left

        # Add right column plots, sharing the x-axis
        right_column_axes = []
        for i in range(5):
            if i == 0:
                ax = fig.add_subplot(gs_right[i])
            else:
                ax = fig.add_subplot(gs_right[i], sharex=right_column_axes[0])
            right_column_axes.append(ax)
        axes.extend(right_column_axes)  # Add to the main list (axes[2] through axes[6])

        # 5. Hide x-axis labels on the upper plots of the right column.
        for ax in right_column_axes[:-1]:
            ax.tick_params(
                axis="x", which="both", bottom=False, top=False, labelbottom=False
            )

        setup_axes(axes)
        # --- MODIFICATION END ---

        self.plot_all_models(fig, axes, self._plot_model)
        self._add_observational_data(axes)

        self.add_legend(fig)
        self.save_figure(fig, "multimodel_summary_pco2.png")

    def _plot_model(self, model, axes, color):
        ave_annual = DataLoader.load_analyser_data(model, "ave", "annual")
        if ave_annual is None:
            return

        actual_years = DataLoader.get_actual_years(ave_annual)
        indices = model.get_year_range_indices(actual_years)
        if indices[0] is None:
            return

        year = DataLoader.safe_load_column(ave_annual, "year", indices)
        pco2_annual = DataLoader.safe_load_column(ave_annual, "pCO2", indices)

        if year is not None and pco2_annual is not None:
            axes[0].plot(
                year, pco2_annual, color=color, linewidth=LINE_WIDTH
            )
            axes[0].set_title(
                "Avg Surface pCO2 (Global) [ppm]", fontsize=TITLE_FONTSIZE
            )

        ave_monthly = DataLoader.load_analyser_data(model, "ave", "monthly")
        if ave_monthly is None:
            return

        monthly_idx = model.get_monthly_index(len(ave_monthly))
        month_names = self._get_month_names(ave_monthly)

        regional_cols = ["pCO2", "pCO2.1", "pCO2.2", "pCO2.3", "pCO2.4", "pCO2.5"]
        for i, col in enumerate(regional_cols):
            label = model.label if i == 0 else None
            data = DataLoader.safe_load_column(
                ave_monthly, col, (monthly_idx, monthly_idx + 12), validate_length=False
            )
            # The axes list is now structured with axes[1] as the first regional plot (Global).
            self._plot_regional_data(axes[i + 1], month_names, data, color, label)

    def _add_observational_data(self, axes):
        month_names = list(calendar.month_abbr)[1:]
        regions = ["global", "reg1", "reg2", "reg3", "reg4", "reg5"]
        titles = ["Global", "45N-90N", "15N-45N", "15S-15N", "45S-15S", "90S-45S"]

        for i, (region, title) in enumerate(zip(regions, titles)):
            # The axes list is now structured with axes[1] as the first regional plot.
            ax = axes[i + 1]
            data = ObservationData.PCO2_MONTHLY[region]
            normalized = data - data[0]
            label = "data-products" if i == 0 else None
            ax.plot(
                month_names,
                normalized,
                color="k",
                linestyle="dashed",
                label=label,
                linewidth=LINE_WIDTH,
            )
            # Use ax.text to prevent titles from overlapping in the new layout
            ax.text(
                0.8,
                0.9,
                f"{title}",
                fontsize=TITLE_FONTSIZE - 2,
                transform=ax.transAxes,
            )


class PhysicsPlotter(PlotGenerator):
    """Generates physics summary plots"""

    def generate(self):
        # Use 1x3 grid to match single-model (3 panels: SST, SSS, MLD)
        fig, axes = plt.subplots(
            1, 3, figsize=(3 * SUBPLOT_WIDTH, 1 * SUBPLOT_HEIGHT), sharex=True,
            constrained_layout=USE_CONSTRAINED_LAYOUT
        )

        setup_axes(axes)

        self.plot_all_models(fig, axes, self._plot_model)

        self.add_legend(fig)
        self.save_figure(fig, "multimodel_summary_physics.png")

    def _plot_model(self, model, axes, color):
        ave_data = DataLoader.load_analyser_data(model, "ave", "annual")
        if ave_data is None:
            return

        actual_years = DataLoader.get_actual_years(ave_data)
        indices = model.get_year_range_indices(actual_years)
        if indices[0] is None:
            return

        year = DataLoader.safe_load_column(ave_data, "year", indices)
        if year is None:
            return

        sst = DataLoader.safe_load_column(ave_data, "tos", indices)
        sss = DataLoader.safe_load_column(ave_data, "sos", indices)
        mld = DataLoader.safe_load_column(ave_data, "mldr10_1", indices)

        if sst is not None:
            axes[0].plot(year, sst, color=color, linewidth=LINE_WIDTH)
            axes[0].set_title("Sea Surface Temperature", fontsize=TITLE_FONTSIZE, fontweight='bold', pad=5)
            axes[0].set_ylabel("°C")
            axes[0].set_xlabel("Year", fontweight='bold')

        if sss is not None:
            axes[1].plot(
                year,
                sss,
                color=color,
                label=model.label,
                linewidth=LINE_WIDTH,
            )
            axes[1].set_title("Sea Surface Salinity", fontsize=TITLE_FONTSIZE, fontweight='bold', pad=5)
            axes[1].set_ylabel("‰")
            axes[1].set_xlabel("Year", fontweight='bold')

        if mld is not None:
            axes[2].plot(year, mld, color=color, linewidth=LINE_WIDTH)
            axes[2].set_title("Mixed Layer Depth", fontsize=TITLE_FONTSIZE, fontweight='bold', pad=5)
            axes[2].set_ylabel("m")
            axes[2].set_xlabel("Year", fontweight='bold')


class DerivedSummaryPlotter(PlotGenerator):
    """Generates derived ecosystem variable summary plots"""

    def generate(self):
        fig, axes = plt.subplots(
            2, 3, figsize=(3 * SUBPLOT_WIDTH, 2 * SUBPLOT_HEIGHT), sharex=True,
            constrained_layout=USE_CONSTRAINED_LAYOUT
        )
        axes = axes.flatten()

        # Hide unused subplots
        axes[4].set_visible(False)
        axes[5].set_visible(False)

        setup_axes(axes[:4])  # Only setup first 4 axes

        self.plot_all_models(fig, axes, self._plot_model)

        self.add_legend(fig)
        self.save_figure(fig, "multimodel_summary_derived.png")

    def _plot_model(self, model, axes, color):
        vol_data = DataLoader.load_analyser_data(model, "vol", "annual")
        lev_data = DataLoader.load_analyser_data(model, "lev", "annual")

        if vol_data is None or lev_data is None:
            return

        actual_years = DataLoader.get_actual_years(vol_data)
        indices = model.get_year_range_indices(actual_years)
        if indices[0] is None:
            return

        year = DataLoader.safe_load_column(vol_data, "year", indices)
        if year is None:
            return

        # Load required columns for derived variables
        ppt = DataLoader.safe_load_column(vol_data, "PPT", indices)
        exp = DataLoader.safe_load_column(lev_data, "EXP", indices)
        exp1000 = DataLoader.safe_load_column(lev_data, "EXP1000", indices)

        # Load grazing terms for secondary production
        grazing_cols = ["GRAGEL", "GRACRU", "GRAMES", "GRAPRO", "GRAPTE"]
        grazing_data = {}
        for col in grazing_cols:
            val = DataLoader.safe_load_column(vol_data, col, indices)
            if val is not None:
                grazing_data[col] = val

        # Calculate derived variables
        if grazing_data:
            sp = sum(grazing_data.values())
            axes[0].plot(year, sp, color=color, label=model.label, linewidth=LINE_WIDTH)
            axes[0].set_title("Secondary Production", fontsize=TITLE_FONTSIZE, fontweight='bold', pad=5)
            axes[0].set_ylabel("PgC/yr")

            if ppt is not None and exp is not None:
                recycle = ppt - exp - sp
                axes[1].plot(year, recycle, color=color, linewidth=LINE_WIDTH)
                axes[1].set_title("Residual Production", fontsize=TITLE_FONTSIZE, fontweight='bold', pad=5)
                axes[1].set_ylabel("PgC/yr")

        if exp is not None and ppt is not None:
            eratio = exp / ppt
            axes[2].plot(year, eratio, color=color, linewidth=LINE_WIDTH)
            axes[2].set_title("Export Ratio (e-ratio)", fontsize=TITLE_FONTSIZE, fontweight='bold', pad=5)
            axes[2].set_ylabel("Dimensionless")
            # Add xlabel to bottom row for the 4-panel layout
            axes[2].set_xlabel("Year", fontweight='bold')

        if exp1000 is not None and exp is not None:
            teff = exp1000 / exp
            axes[3].plot(year, teff, color=color, linewidth=LINE_WIDTH)
            axes[3].set_title("Transfer Efficiency", fontsize=TITLE_FONTSIZE, fontweight='bold', pad=5)
            axes[3].set_ylabel("Dimensionless")
            axes[3].set_xlabel("Year", fontweight='bold')



class DerivedSummaryNormalizedPlotter(PlotGenerator):
    """Generates normalized/anomaly plots for derived ecosystem variables"""

    def generate(self):
        fig, axes = plt.subplots(
            2, 2, figsize=(2 * SUBPLOT_WIDTH, 2 * SUBPLOT_HEIGHT), sharex=True,
            constrained_layout=USE_CONSTRAINED_LAYOUT
        )
        axes = axes.flatten()

        setup_axes(axes)

        self.plot_all_models(fig, axes, self._plot_model)

        self.add_legend(fig)
        self.save_figure(fig, "multimodel_summary_derived_normalized.png")

    def _plot_model(self, model, axes, color):
        vol_data = DataLoader.load_analyser_data(model, "vol", "annual")
        lev_data = DataLoader.load_analyser_data(model, "lev", "annual")

        if vol_data is None or lev_data is None:
            return

        actual_years = DataLoader.get_actual_years(vol_data)
        indices = model.get_year_range_indices(actual_years)
        if indices[0] is None:
            return

        year = DataLoader.safe_load_column(vol_data, "year", indices)
        if year is None:
            return

        ppt = DataLoader.safe_load_column(vol_data, "PPT", indices)
        exp = DataLoader.safe_load_column(lev_data, "EXP", indices)
        exp1000 = DataLoader.safe_load_column(lev_data, "EXP1000", indices)

        grazing_cols = ["GRAGEL", "GRACRU", "GRAMES", "GRAPRO", "GRAPTE"]
        grazing_data = {}
        for col in grazing_cols:
            val = DataLoader.safe_load_column(vol_data, col, indices)
            if val is not None:
                grazing_data[col] = val

        if grazing_data:
            sp = sum(grazing_data.values())
            sp_norm = GlobalSummaryNormalizedPlotter._normalize_series(sp)
            axes[0].plot(year, sp_norm, color=color, label=model.label, linewidth=LINE_WIDTH)
            axes[0].set_title("Secondary Production anomaly", fontsize=TITLE_FONTSIZE, fontweight='bold', pad=5)
            axes[0].set_ylabel("PgC/yr")
            axes[0].axhline(0, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)

            if ppt is not None and exp is not None:
                recycle = ppt - exp - sp
                recycle_norm = GlobalSummaryNormalizedPlotter._normalize_series(recycle)
                axes[1].plot(year, recycle_norm, color=color, linewidth=LINE_WIDTH)
                axes[1].set_title("Residual Production anomaly", fontsize=TITLE_FONTSIZE, fontweight='bold', pad=5)
                axes[1].set_ylabel("PgC/yr")
                axes[1].axhline(0, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)

        if exp is not None and ppt is not None:
            eratio = exp / ppt
            eratio_norm = GlobalSummaryNormalizedPlotter._normalize_series(eratio)
            axes[2].plot(year, eratio_norm, color=color, linewidth=LINE_WIDTH)
            axes[2].set_title("Export Ratio anomaly", fontsize=TITLE_FONTSIZE, fontweight='bold', pad=5)
            axes[2].set_ylabel("Dimensionless")
            axes[2].set_xlabel("Year", fontweight='bold')
            axes[2].axhline(0, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)

        if exp1000 is not None and exp is not None:
            teff = exp1000 / exp
            teff_norm = GlobalSummaryNormalizedPlotter._normalize_series(teff)
            axes[3].plot(year, teff_norm, color=color, linewidth=LINE_WIDTH)
            axes[3].set_title("Transfer Efficiency anomaly", fontsize=TITLE_FONTSIZE, fontweight='bold', pad=5)
            axes[3].set_ylabel("Dimensionless")
            axes[3].set_xlabel("Year", fontweight='bold')
            axes[3].axhline(0, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)


class MultiModelPlotter:
    """Main class coordinating all plot generation"""

    def __init__(self, model_csv_path: str, save_dir: str, debug: bool = False):
        self.models = DataLoader.load_model_configs(model_csv_path)
        self.save_dir = save_dir
        self.debug = debug

        print_info(f"Loaded {len(self.models)} models:")
        for model in self.models:
            print(f"  - {model.name}: {model.description}")

    def generate_all_plots(self):
        plotters = [
            GlobalSummaryPlotter(self.models, self.save_dir),
            GlobalSummaryNormalizedPlotter(self.models, self.save_dir),
            CflxPlotter(self.models, self.save_dir),
            TChlPlotter(self.models, self.save_dir),
            PFTPlotter(self.models, self.save_dir),
            PFTNormalizedPlotter(self.models, self.save_dir),
            NutrientPlotter(self.models, self.save_dir),
            PCO2Plotter(self.models, self.save_dir),
            PhysicsPlotter(self.models, self.save_dir),
            DerivedSummaryPlotter(self.models, self.save_dir),
            DerivedSummaryNormalizedPlotter(self.models, self.save_dir),
        ]

        for plotter in plotters:
            if self.debug:
                plotter.debug = True
            try:
                print_header(f"Generating {plotter.__class__.__name__}")
                plotter.generate()
            except Exception as e:
                print_error(f"Generating {plotter.__class__.__name__}: {e}")
                if self.debug:
                    import traceback

                    traceback.print_exc()


def main():
    if len(sys.argv) < 2:
        print_error("Usage: python script.py <save_directory> [--debug]")
        sys.exit(1)

    model_csv = "modelsToPlot.csv"
    save_dir = sys.argv[1]
    debug = "--debug" in sys.argv

    plotter = MultiModelPlotter(model_csv, save_dir, debug=debug)
    plotter.generate_all_plots()


if __name__ == "__main__":
    main()
