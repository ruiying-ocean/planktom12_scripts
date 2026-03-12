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
    def align_year_and_values(year, values):
        """Align year and values arrays to common length when they differ."""
        if year is None or values is None:
            return None, None
        min_len = min(len(year), len(values))
        if min_len == 0:
            return None, None
        return year[:min_len], values[:min_len]

    @staticmethod
    def get_actual_years(df):
        if df is None or "year" not in df.columns:
            return None
        try:
            return df["year"].to_numpy().astype(float)
        except Exception:
            return None

    @staticmethod
    def compute_relative_change(values, baseline_years=10):
        """
        Compute relative change against the first-decade mean baseline.
        """
        if values is None:
            return None

        values = np.asarray(values, dtype=float)
        if len(values) == 0:
            return None

        baseline_len = min(baseline_years, len(values))
        baseline = np.nanmean(values[:baseline_len])
        if not np.isfinite(baseline) or baseline == 0:
            return None

        with np.errstate(divide="ignore", invalid="ignore"):
            relative_change = (values - baseline) / baseline

        relative_change[~np.isfinite(relative_change)] = np.nan
        return relative_change


class PlotGenerator:
    """Base class for generating plots"""

    def __init__(self, models: List[ModelConfig], save_dir: str):
        self.models = models
        self.save_dir = save_dir
        self._norm_mode = None  # None, 'anomaly', or 'anompct'
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
        save_kwargs = {"dpi": dpi, "bbox_inches": "tight"}
        if fmt == "png":
            save_kwargs["pil_kwargs"] = {"optimize": True, "compress_level": 9}
        fig.savefig(filepath, **save_kwargs)
        print_success(f"Created {filename}")
        plt.close(fig)

    def _apply_norm(self, values: np.ndarray) -> np.ndarray:
        """Apply normalization based on current _norm_mode."""
        if self._norm_mode == 'anomaly':
            return GlobalSummaryNormalizedPlotter._normalize_series(values)
        elif self._norm_mode == 'anompct':
            return GlobalSummaryNormalizedPlotter._normalize_pct_series(values)
        return values

    def _norm_ylabel(self, original: str) -> str:
        """Return ylabel adjusted for current norm mode."""
        if self._norm_mode == 'anompct':
            return '%'
        return original

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
        self.save_figure(fig, "mm_global.png")

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
                # Use custom calculated data - align with year array
                plot_year, plot_data = DataLoader.align_year_and_values(year, custom_data)
                if plot_year is not None:
                    label = model.label if add_label else None
                    ax.plot(plot_year, plot_data, color=color, label=label, linewidth=LINE_WIDTH)
                    ax.set_title(title, fontsize=TITLE_FONTSIZE, fontweight='bold', pad=5)
                    ax.set_ylabel(ylabel)
                    # Add xlabel only to bottom row (indices 6, 7, 8 in 3x3 grid)
                    if idx >= 6:
                        ax.set_xlabel("Year", fontweight='bold')
            elif data_df is not None and column is not None:
                values = DataLoader.safe_load_column(data_df, column, indices)
                plot_year, plot_values = DataLoader.align_year_and_values(year, values)
                if plot_year is not None:
                    label = model.label if add_label else None
                    ax.plot(plot_year, plot_values, color=color, label=label, linewidth=LINE_WIDTH)
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
                if obs_key in ObservationData.get_global():
                    obs = ObservationData.get_global()[obs_key]
                    ax = axes[i]
                    if obs.get("type") == "line" or obs_type == "line":
                        ax.axhline(obs["value"], **LINE_STYLE)
                    else:
                        ax.axhspan(obs["min"], obs["max"], **HATCH_STYLE)


class GlobalSummaryNormalizedPlotter(PlotGenerator):
    """Generates normalized global summary plots using first 10 years as baseline"""

    def generate(self):
        for mode, filename in [
            ('anomaly', 'mm_global_normalized.png'),
            ('anompct', 'mm_global_anompct.png'),
        ]:
            self._norm_mode = mode
            fig, axes = plt.subplots(
                3, 3, figsize=(3 * SUBPLOT_WIDTH, 3 * SUBPLOT_HEIGHT), sharex=True,
                constrained_layout=USE_CONSTRAINED_LAYOUT
            )
            axes = axes.flatten()

            setup_axes(axes)

            self.plot_all_models(fig, axes, self._plot_model)

            self.add_legend(fig)
            self.save_figure(fig, filename)

    @staticmethod
    def _normalize_series(values: np.ndarray) -> np.ndarray:
        """Return anomalies relative to the mean of the first decade (or available years)."""
        if len(values) == 0:
            return values
        baseline_years = min(10, len(values))
        baseline = np.mean(values[:baseline_years])
        return values - baseline

    @staticmethod
    def _normalize_pct_series(values: np.ndarray) -> np.ndarray:
        """Return percent anomalies relative to the mean of the first decade."""
        if len(values) == 0:
            return values
        baseline_years = min(10, len(values))
        baseline = np.nanmean(values[:baseline_years])
        if not np.isfinite(baseline) or baseline == 0:
            return np.full(len(values), np.nan, dtype=float)
        return (values - baseline) / abs(baseline) * 100

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

            plot_year, plot_values = DataLoader.align_year_and_values(year, values)
            if plot_year is None:
                continue

            normalized_values = self._apply_norm(plot_values)
            label = model.label if add_label else None
            ax.plot(
                plot_year,
                normalized_values,
                color=color,
                label=label,
                linewidth=LINE_WIDTH,
            )
            ax.set_title(title, fontsize=TITLE_FONTSIZE, fontweight='bold', pad=5)
            ax.set_ylabel(self._norm_ylabel(ylabel))
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
        self.save_figure(fig, "mm_pfts.png")

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
            plot_year, plot_values = DataLoader.align_year_and_values(year, values)
            if plot_year is not None:
                label = model.label if i == 10 else None
                axes[i].plot(
                    plot_year,
                    plot_values,
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
            if pft_name in ObservationData.get_pft():
                obs = ObservationData.get_pft()[pft_name]
                if obs.get("type") == "line":
                    axes[idx].axhline(obs["value"], **LINE_STYLE)
                elif obs.get("type") == "range" and obs["min"] is not None and obs["max"] is not None:
                    axes[idx].axhspan(obs["min"], obs["max"], **HATCH_STYLE)


class PFTNormalizedPlotter(PlotGenerator):
    """Generates normalized/anomaly plots for PFT biomass"""

    def generate(self):
        for mode, filename in [
            ('anomaly', 'mm_pfts_normalized.png'),
            ('anompct', 'mm_pfts_anompct.png'),
        ]:
            self._norm_mode = mode
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
            self.save_figure(fig, filename)

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
            plot_year, plot_values = DataLoader.align_year_and_values(year, values)
            if plot_year is None:
                continue

            normalized = self._apply_norm(plot_values)
            label = model.label if i == 10 else None
            axes[i].plot(plot_year, normalized, color=color, label=label, linewidth=LINE_WIDTH)
            axes[i].set_title(title, fontsize=TITLE_FONTSIZE, fontweight="bold", pad=5)
            axes[i].set_ylabel(self._norm_ylabel("PgC anomaly"))
            axes[i].axhline(0, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
            if i >= 9:
                axes[i].set_xlabel("Year", fontweight="bold")


class PPTByPFTPlotter(PlotGenerator):
    """Generates PPT by phytoplankton PFT plots"""

    def generate(self):
        # Use 2x3 grid for 6 phytoplankton PFTs
        fig, axes = plt.subplots(
            2,
            3,
            figsize=(3 * SUBPLOT_WIDTH, 2 * SUBPLOT_HEIGHT),
            sharex=True,
            constrained_layout=USE_CONSTRAINED_LAYOUT,
        )
        flat_axes = axes.flatten()

        setup_axes(flat_axes)

        self.plot_all_models(fig, flat_axes, self._plot_model)

        self.add_legend(fig)
        self.save_figure(fig, "mm_ppt_by_pft.png")

    def _plot_model(self, model, axes, color):
        vol_data = DataLoader.load_analyser_data(model, "vol", "annual")
        if vol_data is None:
            return

        actual_years = DataLoader.get_actual_years(vol_data)
        indices = model.get_year_range_indices(actual_years)
        if indices[0] is None:
            return

        year = DataLoader.safe_load_column(vol_data, "year", indices)
        if year is None:
            return

        ppt_mapping = [
            ("PPT_PIC", "Picophytoplankton PP"),
            ("PPT_PHA", "Phaeocystis PP"),
            ("PPT_MIX", "Mixotrophs PP"),
            ("PPT_DIA", "Diatoms PP"),
            ("PPT_COC", "Coccolithophores PP"),
            ("PPT_FIX", "Nitrogen Fixers PP"),
        ]

        for i, (col_name, title) in enumerate(ppt_mapping):
            values = DataLoader.safe_load_column(vol_data, col_name, indices)
            plot_year, plot_values = DataLoader.align_year_and_values(year, values)
            if plot_year is not None:
                label = model.label if i == 4 else None
                axes[i].plot(
                    plot_year,
                    plot_values,
                    color=color,
                    label=label,
                    linewidth=LINE_WIDTH,
                )
                axes[i].set_title(title, fontsize=TITLE_FONTSIZE, fontweight='bold', pad=5)
                axes[i].set_ylabel("PgC/yr")
                # Add xlabel only to bottom row (indices 3, 4, 5 in 2x3 grid)
                if i >= 3:
                    axes[i].set_xlabel("Year", fontweight='bold')


class PPTByPFTNormalizedPlotter(PlotGenerator):
    """Generates normalized/anomaly plots for PPT by phytoplankton PFT"""

    def generate(self):
        for mode, filename in [
            ('anomaly', 'mm_ppt_by_pft_normalized.png'),
            ('anompct', 'mm_ppt_by_pft_anompct.png'),
        ]:
            self._norm_mode = mode
            fig, axes = plt.subplots(
                2,
                3,
                figsize=(3 * SUBPLOT_WIDTH, 2 * SUBPLOT_HEIGHT),
                sharex=True,
                constrained_layout=USE_CONSTRAINED_LAYOUT,
            )
            flat_axes = axes.flatten()

            setup_axes(flat_axes)

            self.plot_all_models(fig, flat_axes, self._plot_model)

            self.add_legend(fig)
            self.save_figure(fig, filename)

    def _plot_model(self, model, axes, color):
        vol_data = DataLoader.load_analyser_data(model, "vol", "annual")
        if vol_data is None:
            return

        actual_years = DataLoader.get_actual_years(vol_data)
        indices = model.get_year_range_indices(actual_years)
        if indices[0] is None:
            return

        year = DataLoader.safe_load_column(vol_data, "year", indices)
        if year is None:
            return

        ppt_mapping = [
            ("PPT_PIC", "PIC PP anomaly"),
            ("PPT_PHA", "PHA PP anomaly"),
            ("PPT_MIX", "MIX PP anomaly"),
            ("PPT_DIA", "DIA PP anomaly"),
            ("PPT_COC", "COC PP anomaly"),
            ("PPT_FIX", "FIX PP anomaly"),
        ]

        for i, (col_name, title) in enumerate(ppt_mapping):
            values = DataLoader.safe_load_column(vol_data, col_name, indices)
            plot_year, plot_values = DataLoader.align_year_and_values(year, values)
            if plot_year is None:
                continue

            normalized = self._apply_norm(plot_values)
            label = model.label if i == 4 else None
            axes[i].plot(plot_year, normalized, color=color, label=label, linewidth=LINE_WIDTH)
            axes[i].set_title(title, fontsize=TITLE_FONTSIZE, fontweight="bold", pad=5)
            axes[i].set_ylabel(self._norm_ylabel("PgC/yr anomaly"))
            axes[i].axhline(0, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
            if i >= 3:
                axes[i].set_xlabel("Year", fontweight="bold")


class TChlPlotter(PlotGenerator):
    """Generates TChl seasonal summary plots by region, consistent with single-model output"""

    TCHL_REGION_DEFS = [
        {"title": "N. Pacific subtropical",  "lat_range": (15.0, 35.0),  "lon_range": (160.0, -130.0)},
        {"title": "N. Pacific subpolar",      "lat_range": (50.0, 62.0),  "lon_range": (160.0, -140.0)},
        {"title": "N. Atlantic subtropical",  "lat_range": (20.0, 35.0),  "lon_range": (-80.0, -20.0)},
        {"title": "N. Atlantic subpolar",     "lat_range": (45.0, 65.0),  "lon_range": (-60.0, -10.0)},
        {"title": "Equatorial Pacific",       "lat_range": (-5.0, 5.0),   "lon_range": (160.0, -90.0)},
        {"title": "S. Pacific subtropical",   "lat_range": (-35.0, -15.0),"lon_range": (170.0, -100.0)},
        {"title": "S. Atlantic subtropical",  "lat_range": (-30.0, -10.0),"lon_range": (-50.0, 10.0)},
        {"title": "Sub-Antarctic Zone (Indian)", "lat_range": (-50.0, -40.0),"lon_range": (20.0, 120.0)},
        {"title": "Antarctic Zone",           "lat_range": (-65.0, -55.0),"lon_range": None},
    ]

    @staticmethod
    def _build_lon_mask(lon, lon_range):
        import xarray as xr
        if lon_range is None:
            return xr.ones_like(lon, dtype=bool)
        lon_min, lon_max = lon_range
        lon_0360 = xr.where(lon < 0, lon + 360, lon)
        min_0360 = lon_min if lon_min >= 0 else lon_min + 360
        max_0360 = lon_max if lon_max >= 0 else lon_max + 360
        if min_0360 <= max_0360:
            return (lon_0360 >= min_0360) & (lon_0360 <= max_0360)
        return (lon_0360 >= min_0360) | (lon_0360 <= max_0360)

    @classmethod
    def _load_occci_regional(cls):
        import xarray as xr, os
        chl_file = pathlib.Path(os.path.expanduser("~/Observations")) / "OC-CCI/climatology/OC-CCI_climatology_1deg.nc"
        if not chl_file.exists():
            print_warning(f"OC-CCI file not found: {chl_file} — no obs overlay")
            return None
        try:
            with xr.open_dataset(chl_file, decode_times=False) as ds:
                if "chlor_a" not in ds:
                    return None
                chl = ds["chlor_a"]
                lat_name = next((n for n in ["lat", "latitude", "nav_lat"] if n in ds), None)
                lon_name = next((n for n in ["lon", "longitude", "nav_lon"] if n in ds), None)
                if lat_name is None or lon_name is None:
                    return None
                lat, lon = ds[lat_name], ds[lon_name]
                weights = np.cos(np.deg2rad(lat))
                time_dim = next((d for d in ["time", "month"] if d in chl.dims), None)
                obs_series = []
                for region in cls.TCHL_REGION_DEFS:
                    lat_min, lat_max = region["lat_range"]
                    lat_mask = (lat >= lat_min) & (lat <= lat_max)
                    lon_mask = cls._build_lon_mask(lon, region["lon_range"])
                    region_mask = lat_mask & lon_mask
                    if time_dim is not None:
                        monthly = []
                        for t in range(chl.sizes[time_dim]):
                            snap = chl.isel({time_dim: t})
                            valid_mask = region_mask & np.isfinite(snap)
                            w = weights.where(valid_mask, 0.0)
                            monthly.append(float(snap.where(valid_mask).weighted(w).mean()))
                        obs_series.append(np.array(monthly))
                    else:
                        obs_series.append(None)
                return obs_series
        except Exception as e:
            print_warning(f"Could not load OC-CCI data: {e}")
            return None

    def _plot_model(self, model, axes, color):
        import xarray as xr
        run_dir = pathlib.Path(model.model_dir) / model.name
        diad_files = sorted(run_dir.glob("ORCA2_1m_*_diad_T.nc"))
        if not diad_files:
            print_warning(f"No ORCA2_1m_*_diad_T.nc found in {run_dir}")
            return
        nc_file = max(diad_files, key=lambda p: p.name.split("_")[2] if len(p.name.split("_")) > 2 else "")
        try:
            with xr.open_dataset(nc_file, decode_times=False) as ds:
                if "TChl" not in ds:
                    print_warning(f"TChl not found in {nc_file.name}")
                    return
                tchl = ds["TChl"]
                depth_dim = next((d for d in ["deptht", "nav_lev", "z"] if d in tchl.dims), None)
                if depth_dim is not None:
                    tchl = tchl.isel({depth_dim: 0})
                lat_name = "nav_lat" if "nav_lat" in ds else "lat"
                lon_name = "nav_lon" if "nav_lon" in ds else "lon"
                if lat_name not in ds or lon_name not in ds:
                    return
                lat, lon = ds[lat_name], ds[lon_name]
                spatial_dims = lat.dims
                weights = xr.where(np.isfinite(lat), np.cos(np.deg2rad(lat)), 0.0)
                x = np.arange(12)
                month_names = [calendar.month_abbr[m] for m in range(1, 13)]
                for i, region in enumerate(self.TCHL_REGION_DEFS):
                    if i >= len(axes):
                        break
                    lat_min, lat_max = region["lat_range"]
                    lat_mask = (lat >= lat_min) & (lat <= lat_max)
                    lon_mask = self._build_lon_mask(lon, region["lon_range"])
                    region_mask = lat_mask & lon_mask & np.isfinite(tchl.isel(time_counter=0))
                    series = tchl.where(region_mask).weighted(weights.where(region_mask, 0.0)).mean(dim=spatial_dims)
                    vals = series.to_numpy().astype(float) * 1e6
                    if len(vals) == 12:
                        label = model.label if i == 0 else None
                        axes[i].plot(x, vals, color=color, linewidth=LINE_WIDTH, label=label)
                        axes[i].set_title(region["title"], fontsize=TITLE_FONTSIZE, fontweight='bold', pad=5)
                        axes[i].set_xticks(x)
                        axes[i].set_xticklabels(month_names, rotation=45, ha='right')
                        axes[i].set_ylabel("μg Chl/L")
        except Exception as e:
            print_warning(f"Could not load TChl from {nc_file.name}: {e}")

    def _add_obs_overlay(self, axes, obs_series):
        if obs_series is None:
            return
        x = np.arange(12)
        for i, obs in enumerate(obs_series):
            if i >= len(axes) or obs is None or len(obs) != 12:
                continue
            label = "OC-CCI" if i == 0 else None
            axes[i].plot(x, obs, label=label, **LINE_STYLE)

    def generate(self):
        n_regions = len(self.TCHL_REGION_DEFS)
        ncols = 3
        nrows = (n_regions + ncols - 1) // ncols
        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(ncols * SUBPLOT_WIDTH, nrows * SUBPLOT_HEIGHT),
            squeeze=False, sharex=True,
            constrained_layout=USE_CONSTRAINED_LAYOUT,
        )
        flat_axes = axes.flatten()
        setup_axes(flat_axes)
        obs_series = self._load_occci_regional()
        self.plot_all_models(fig, flat_axes, self._plot_model)
        self._add_obs_overlay(flat_axes, obs_series)
        for ax in flat_axes[n_regions:]:
            ax.set_visible(False)
        self.add_legend(fig)
        self.save_figure(fig, "mm_tchl.png")


class NutrientPlotter(PlotGenerator):
    """Generates nutrient summary plots"""

    def generate(self):
        for mode, filename, include_obs in [
            (None, 'mm_nutrients.png', True),
            ('anomaly', 'mm_nutrients_anom.png', False),
            ('anompct', 'mm_nutrients_anompct.png', False),
        ]:
            self._norm_mode = mode
            # Use 3x3 grid for 8 nutrients (matching single-model)
            fig, axes = plt.subplots(
                3, 3, figsize=(3 * SUBPLOT_WIDTH, 3 * SUBPLOT_HEIGHT), sharex=True,
                constrained_layout=USE_CONSTRAINED_LAYOUT
            )
            flat_axes = axes.flatten()

            setup_axes(flat_axes)

            self.plot_all_models(fig, flat_axes, self._plot_model)
            if include_obs:
                self._add_observational_data(flat_axes)

            self.add_legend(fig)
            self.save_figure(fig, filename)

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
            ("DIC", axes[6], "Surface DIC", "μmol/L", 1, False),
            ("AOU", axes[7], "AOU at 300m", "μmol/L", 1, False),
        ]

        for idx, (col_name, ax, title, ylabel, scale, add_label) in enumerate(plot_configs):
            values = DataLoader.safe_load_column(ave_data, col_name, indices)
            plot_year, plot_values = DataLoader.align_year_and_values(year, values)
            if plot_year is not None:
                label = model.label if add_label else None
                values_to_plot = plot_values * scale
                if self._norm_mode is not None:
                    values_to_plot = self._apply_norm(values_to_plot)
                ax.plot(
                    plot_year,
                    values_to_plot,
                    color=color,
                    label=label,
                    linewidth=LINE_WIDTH,
                )
                ax.set_title(title, fontsize=TITLE_FONTSIZE, fontweight='bold', pad=5)
                ax.set_ylabel(self._norm_ylabel(ylabel))
                # Add xlabel only to bottom row (last 3 in 3x3 grid)
                if idx >= 5:
                    ax.set_xlabel("Year", fontweight='bold')

    def _add_observational_data(self, axes):
        # Add observation lines for all 8 nutrients
        nutrient_keys = ["PO4", "NO3", "Fer", "Si", "O2", "Alkalini", "DIC", "AOU"]
        for i, key in enumerate(nutrient_keys):
            if i < len(axes) and key in ObservationData.get_nutrients():
                obs_value = ObservationData.get_nutrients()[key]
                if obs_value is not None:
                    axes[i].axhline(obs_value, **LINE_STYLE)

        for idx in range(len(nutrient_keys), len(axes)):
            axes[idx].set_visible(False)


class BenthicPlotter(PlotGenerator):
    """Generates deep-ocean (benthic) nutrient summary plots"""

    def generate(self):
        for mode, filename, include_obs in [
            (None, 'mm_benthic.png', True),
            ('anomaly', 'mm_benthic_anom.png', False),
            ('anompct', 'mm_benthic_anompct.png', False),
        ]:
            self._norm_mode = mode
            # 3x3 grid for 7 benthic variables
            fig, axes = plt.subplots(
                3, 3, figsize=(3 * SUBPLOT_WIDTH, 3 * SUBPLOT_HEIGHT), sharex=True,
                constrained_layout=USE_CONSTRAINED_LAYOUT
            )
            flat_axes = axes.flatten()

            setup_axes(flat_axes[:8])

            self.plot_all_models(fig, flat_axes, self._plot_model)
            if include_obs:
                self._add_observational_data(flat_axes)

            # Hide unused subplots
            for idx in range(8, len(flat_axes)):
                flat_axes[idx].set_visible(False)

            self.add_legend(fig)
            self.save_figure(fig, filename)

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
            ("bPO4", axes[0], "Deep Phosphate", "\u03bcmol/L", 1/122, False),
            ("bNO3", axes[1], "Deep Nitrate", "\u03bcmol/L", 1, True),
            ("bFer", axes[2], "Deep Iron", "nmol/L", 1000, False),
            ("bSi", axes[3], "Deep Silica", "\u03bcmol/L", 1, False),
            ("bO2", axes[4], "Deep Oxygen", "\u03bcmol/L", 1, False),
            ("bAlkalini", axes[5], "Deep Alkalinity", "\u03bcmol/L", 1, False),
            ("bDIC", axes[6], "Deep DIC", "\u03bcmol/L", 1, False),
            ("bDOC", axes[7], "Deep DOC", "\u03bcmol/L", 1, False),
        ]

        for idx, (col_name, ax, title, ylabel, scale, add_label) in enumerate(plot_configs):
            values = DataLoader.safe_load_column(ave_data, col_name, indices)
            plot_year, plot_values = DataLoader.align_year_and_values(year, values)
            if plot_year is not None:
                label = model.label if add_label else None
                values_to_plot = plot_values * scale
                if self._norm_mode is not None:
                    values_to_plot = self._apply_norm(values_to_plot)
                ax.plot(
                    plot_year,
                    values_to_plot,
                    color=color,
                    label=label,
                    linewidth=LINE_WIDTH,
                )
                ax.set_title(title, fontsize=TITLE_FONTSIZE, fontweight='bold', pad=5)
                ax.set_ylabel(self._norm_ylabel(ylabel))

    def _add_observational_data(self, axes):
        benthic_obs = ObservationData.get_benthic()
        benthic_keys = ["bPO4", "bNO3", "bFer", "bSi", "bO2", "bAlkalini", "bDIC", "bDOC"]
        for i, key in enumerate(benthic_keys):
            if i < len(axes) and key in benthic_obs:
                obs_value = benthic_obs[key]
                if obs_value is not None:
                    axes[i].axhline(obs_value, **LINE_STYLE)


class PCO2Plotter(RegionalPlotter):
    """Generates pCO2 summary plots"""

    def generate(self):
        for mode, filename in [
            (None, 'mm_pco2.png'),
            ('anomaly', 'mm_pco2_anom.png'),
            ('anompct', 'mm_pco2_anompct.png'),
        ]:
            self._norm_mode = mode
            fig, ax = plt.subplots(
                1, 1, figsize=(2 * SUBPLOT_WIDTH, 2 * SUBPLOT_HEIGHT),
                constrained_layout=USE_CONSTRAINED_LAYOUT
            )
            setup_axes([ax])
            self.plot_all_models(fig, [ax], self._plot_model)
            self.add_legend(fig)
            self.save_figure(fig, filename)

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
            values_to_plot = pco2_annual
            if self._norm_mode is not None:
                values_to_plot = self._apply_norm(pco2_annual)
            axes[0].plot(
                year, values_to_plot, color=color, linewidth=LINE_WIDTH
            )
            axes[0].set_title(
                "Avg Surface pCO2 (Global) [ppm]", fontsize=TITLE_FONTSIZE
            )
            axes[0].set_ylabel(self._norm_ylabel("ppm"))
            axes[0].set_xlabel("Year", fontweight='bold')


class PhysicsPlotter(PlotGenerator):
    """Generates physics summary plots"""

    def generate(self):
        for mode, filename in [
            (None, 'mm_physics.png'),
            ('anomaly', 'mm_physics_anom.png'),
            ('anompct', 'mm_physics_anompct.png'),
        ]:
            self._norm_mode = mode
            # Use 1x3 grid to match single-model (3 panels: SST, SSS, MLD)
            fig, axes = plt.subplots(
                1, 3, figsize=(3 * SUBPLOT_WIDTH, 1 * SUBPLOT_HEIGHT), sharex=True,
                constrained_layout=USE_CONSTRAINED_LAYOUT
            )

            setup_axes(axes)

            self.plot_all_models(fig, axes, self._plot_model)

            self.add_legend(fig)
            self.save_figure(fig, filename)

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
            sst_to_plot = self._apply_norm(sst) if self._norm_mode is not None else sst
            axes[0].plot(year, sst_to_plot, color=color, linewidth=LINE_WIDTH)
            axes[0].set_title("Sea Surface Temperature", fontsize=TITLE_FONTSIZE, fontweight='bold', pad=5)
            axes[0].set_ylabel(self._norm_ylabel("°C"))
            axes[0].set_xlabel("Year", fontweight='bold')

        if sss is not None:
            sss_to_plot = self._apply_norm(sss) if self._norm_mode is not None else sss
            axes[1].plot(
                year,
                sss_to_plot,
                color=color,
                label=model.label,
                linewidth=LINE_WIDTH,
            )
            axes[1].set_title("Sea Surface Salinity", fontsize=TITLE_FONTSIZE, fontweight='bold', pad=5)
            axes[1].set_ylabel(self._norm_ylabel("‰"))
            axes[1].set_xlabel("Year", fontweight='bold')

        if mld is not None:
            mld_to_plot = self._apply_norm(mld) if self._norm_mode is not None else mld
            axes[2].plot(year, mld_to_plot, color=color, linewidth=LINE_WIDTH)
            axes[2].set_title("Mixed Layer Depth", fontsize=TITLE_FONTSIZE, fontweight='bold', pad=5)
            axes[2].set_ylabel(self._norm_ylabel("m"))
            axes[2].set_xlabel("Year", fontweight='bold')


class DerivedSummaryPlotter(PlotGenerator):
    """Generates derived ecosystem variable summary plots"""

    def generate(self):
        self.ta_slopes = []
        self.ba_slopes = []
        fig, axes = plt.subplots(
            3, 3, figsize=(3 * SUBPLOT_WIDTH, 3 * SUBPLOT_HEIGHT), sharex=False,
            constrained_layout=USE_CONSTRAINED_LAYOUT
        )
        axes = axes.flatten()

        setup_axes(axes[:7])

        self.plot_all_models(fig, axes, self._plot_model)

        # Add observation line for ALK-DIC
        derived_obs = ObservationData.get_derived()
        if "ALK_DIC" in derived_obs:
            axes[6].axhline(derived_obs["ALK_DIC"], **LINE_STYLE)

        if self.ta_slopes:
            slope_text = "\n".join(f"{name}: {slope:.2f}" for name, slope in self.ta_slopes)
            axes[7].text(
                0.05, 0.95, slope_text, transform=axes[7].transAxes,
                va='top', ha='left', fontsize=7,
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.7)
            )

        if self.ba_slopes:
            slope_text = "\n".join(f"{name}: {slope:.2f}" for name, slope in self.ba_slopes)
            axes[8].text(
                0.05, 0.95, slope_text, transform=axes[8].transAxes,
                va='top', ha='left', fontsize=7,
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.7)
            )

        # Hide unused subplots
        for idx in range(9, len(axes)):
            axes[idx].set_visible(False)

        self.add_legend(fig)
        self.save_figure(fig, "mm_derived.png")

    def _plot_model(self, model, axes, color):
        vol_data = DataLoader.load_analyser_data(model, "vol", "annual")
        lev_data = DataLoader.load_analyser_data(model, "lev", "annual")
        int_data = DataLoader.load_analyser_data(model, "int", "annual")

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
            plot_year, plot_sp = DataLoader.align_year_and_values(year, sp)
            if plot_year is not None:
                axes[0].plot(plot_year, plot_sp, color=color, label=model.label, linewidth=LINE_WIDTH)
                axes[0].set_title("Secondary Production", fontsize=TITLE_FONTSIZE, fontweight='bold', pad=5)
                axes[0].set_ylabel("PgC/yr")

            if ppt is not None and exp is not None:
                # Align all arrays to minimum common length
                min_len = min(len(year), len(ppt), len(exp), len(sp))
                recycle = ppt[:min_len] - exp[:min_len] - sp[:min_len]
                axes[1].plot(year[:min_len], recycle, color=color, linewidth=LINE_WIDTH)
                axes[1].set_title("Residual Production", fontsize=TITLE_FONTSIZE, fontweight='bold', pad=5)
                axes[1].set_ylabel("PgC/yr")

            if ppt is not None:
                min_len = min(len(year), len(sp), len(ppt))
                spratio = sp[:min_len] / ppt[:min_len]
                axes[5].plot(year[:min_len], spratio, color=color, linewidth=LINE_WIDTH)
                axes[5].set_title("SP/NPP", fontsize=TITLE_FONTSIZE, fontweight='bold', pad=5)
                axes[5].set_ylabel("Dimensionless")

                x = DataLoader.compute_relative_change(ppt[:min_len])
                y = DataLoader.compute_relative_change(sp[:min_len])
                if x is not None and y is not None:
                    valid = np.isfinite(x) & np.isfinite(y)
                    if np.count_nonzero(valid) >= 2:
                        x_valid = x[valid]
                        y_valid = y[valid]
                        slope, intercept = np.polyfit(x_valid, y_valid, 1)
                        axes[7].scatter(x_valid, y_valid, color=color, s=16, alpha=0.75, label=model.label)
                        x_line = np.linspace(np.nanmin(x_valid), np.nanmax(x_valid), 100)
                        axes[7].plot(x_line, slope * x_line + intercept, color=color, linewidth=1.0)
                        axes[7].set_title("Trophic Amplification", fontsize=TITLE_FONTSIZE, fontweight='bold', pad=5)
                        axes[7].set_xlabel("Relative change in NPP", fontweight='bold')
                        axes[7].set_ylabel("Relative change in SP")
                        axes[7].axhline(0, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
                        axes[7].axvline(0, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
                        self.ta_slopes.append((model.name, slope))

        if exp is not None and ppt is not None:
            min_len = min(len(year), len(exp), len(ppt))
            eratio = exp[:min_len] / ppt[:min_len]
            axes[2].plot(year[:min_len], eratio, color=color, linewidth=LINE_WIDTH)
            axes[2].set_title("Export Ratio (e-ratio)", fontsize=TITLE_FONTSIZE, fontweight='bold', pad=5)
            axes[2].set_ylabel("Dimensionless")

        if exp1000 is not None and exp is not None:
            min_len = min(len(year), len(exp1000), len(exp))
            teff = exp1000[:min_len] / exp[:min_len]
            axes[3].plot(year[:min_len], teff, color=color, linewidth=LINE_WIDTH)
            axes[3].set_title("Transfer Efficiency", fontsize=TITLE_FONTSIZE, fontweight='bold', pad=5)
            axes[3].set_ylabel("Dimensionless")

        if int_data is not None:
            phy_cols = ["COC", "DIA", "FIX", "MIX", "PHA", "PIC"]
            zoo_cols = ["GEL", "CRU", "MES", "PRO", "PTE"]
            phy_parts = [DataLoader.safe_load_column(int_data, col, indices) for col in phy_cols]
            zoo_parts = [DataLoader.safe_load_column(int_data, col, indices) for col in zoo_cols]
            if all(part is not None for part in phy_parts + zoo_parts):
                phy = sum(phy_parts)
                zoo = sum(zoo_parts)
                min_len = min(len(phy), len(zoo))
                x = DataLoader.compute_relative_change(phy[:min_len])
                y = DataLoader.compute_relative_change(zoo[:min_len])
                if x is not None and y is not None:
                    valid = np.isfinite(x) & np.isfinite(y)
                    if np.count_nonzero(valid) >= 2:
                        x_valid = x[valid]
                        y_valid = y[valid]
                        slope, intercept = np.polyfit(x_valid, y_valid, 1)
                        axes[8].scatter(x_valid, y_valid, color=color, s=16, alpha=0.75, label=model.label)
                        x_line = np.linspace(np.nanmin(x_valid), np.nanmax(x_valid), 100)
                        axes[8].plot(x_line, slope * x_line + intercept, color=color, linewidth=1.0)
                        axes[8].set_title("Biomass Amplification", fontsize=TITLE_FONTSIZE, fontweight='bold', pad=5)
                        axes[8].set_xlabel("Relative change in PHY", fontweight='bold')
                        axes[8].set_ylabel("Relative change in ZOO")
                        axes[8].axhline(0, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
                        axes[8].axvline(0, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
                        self.ba_slopes.append((model.name, slope))

        # RLS and ALK-DIC from average file
        ave_data = DataLoader.load_analyser_data(model, "ave", "annual")
        if ave_data is not None:
            rls = DataLoader.safe_load_column(ave_data, "RLS", indices)
            plot_year, plot_rls = DataLoader.align_year_and_values(year, rls)
            if plot_year is not None:
                axes[4].plot(plot_year, plot_rls, color=color, linewidth=LINE_WIDTH)
                axes[4].set_title("RLS", fontsize=TITLE_FONTSIZE, fontweight='bold', pad=5)
                axes[4].set_ylabel("m")

            alk = DataLoader.safe_load_column(ave_data, "Alkalini", indices)
            dic = DataLoader.safe_load_column(ave_data, "DIC", indices)
            if alk is not None and dic is not None:
                min_len = min(len(year), len(alk), len(dic))
                alk_dic = alk[:min_len] - dic[:min_len]
                axes[6].plot(year[:min_len], alk_dic, color=color, linewidth=LINE_WIDTH)
                axes[6].set_title("ALK \u2212 DIC", fontsize=TITLE_FONTSIZE, fontweight='bold', pad=5)
                axes[6].set_ylabel("\u03bcmol/L")
                axes[6].set_xlabel("Year", fontweight='bold')


class DerivedSummaryNormalizedPlotter(PlotGenerator):
    """Generates normalized/anomaly plots for derived ecosystem variables"""

    def generate(self):
        for mode, filename in [
            ('anomaly', 'mm_derived_normalized.png'),
            ('anompct', 'mm_derived_anompct.png'),
        ]:
            self._norm_mode = mode
            self.ta_slopes = []
            self.ba_slopes = []
            fig, axes = plt.subplots(
                3, 3, figsize=(3 * SUBPLOT_WIDTH, 3 * SUBPLOT_HEIGHT), sharex=False,
                constrained_layout=USE_CONSTRAINED_LAYOUT
            )
            axes = axes.flatten()

            setup_axes(axes[:7])

            self.plot_all_models(fig, axes, self._plot_model)

            if self.ta_slopes:
                slope_text = "\n".join(f"{name}: {slope:.2f}" for name, slope in self.ta_slopes)
                axes[7].text(
                    0.05, 0.95, slope_text, transform=axes[7].transAxes,
                    va='top', ha='left', fontsize=7,
                    bbox=dict(facecolor='white', edgecolor='none', alpha=0.7)
                )

            if self.ba_slopes:
                slope_text = "\n".join(f"{name}: {slope:.2f}" for name, slope in self.ba_slopes)
                axes[8].text(
                    0.05, 0.95, slope_text, transform=axes[8].transAxes,
                    va='top', ha='left', fontsize=7,
                    bbox=dict(facecolor='white', edgecolor='none', alpha=0.7)
                )

            # Hide unused subplots
            for idx in range(9, len(axes)):
                axes[idx].set_visible(False)

            self.add_legend(fig)
            self.save_figure(fig, filename)

    def _plot_model(self, model, axes, color):
        vol_data = DataLoader.load_analyser_data(model, "vol", "annual")
        lev_data = DataLoader.load_analyser_data(model, "lev", "annual")
        int_data = DataLoader.load_analyser_data(model, "int", "annual")

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
            plot_year, plot_sp = DataLoader.align_year_and_values(year, sp)
            if plot_year is not None:
                sp_norm = self._apply_norm(plot_sp)
                axes[0].plot(plot_year, sp_norm, color=color, label=model.label, linewidth=LINE_WIDTH)
                axes[0].set_title("Secondary Production anomaly", fontsize=TITLE_FONTSIZE, fontweight='bold', pad=5)
                axes[0].set_ylabel(self._norm_ylabel("PgC/yr"))
                axes[0].axhline(0, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)

            if ppt is not None and exp is not None:
                min_len = min(len(year), len(ppt), len(exp), len(sp))
                recycle = ppt[:min_len] - exp[:min_len] - sp[:min_len]
                recycle_norm = self._apply_norm(recycle)
                axes[1].plot(year[:min_len], recycle_norm, color=color, linewidth=LINE_WIDTH)
                axes[1].set_title("Residual Production anomaly", fontsize=TITLE_FONTSIZE, fontweight='bold', pad=5)
                axes[1].set_ylabel(self._norm_ylabel("PgC/yr"))
                axes[1].axhline(0, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)

            if ppt is not None:
                min_len = min(len(year), len(sp), len(ppt))
                spratio = sp[:min_len] / ppt[:min_len]
                spratio_norm = self._apply_norm(spratio)
                axes[5].plot(year[:min_len], spratio_norm, color=color, linewidth=LINE_WIDTH)
                axes[5].set_title("SP/NPP anomaly", fontsize=TITLE_FONTSIZE, fontweight='bold', pad=5)
                axes[5].set_ylabel(self._norm_ylabel("Dimensionless"))
                axes[5].axhline(0, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)

                x = DataLoader.compute_relative_change(ppt[:min_len])
                y = DataLoader.compute_relative_change(sp[:min_len])
                if x is not None and y is not None:
                    valid = np.isfinite(x) & np.isfinite(y)
                    if np.count_nonzero(valid) >= 2:
                        x_valid = x[valid]
                        y_valid = y[valid]
                        slope, intercept = np.polyfit(x_valid, y_valid, 1)
                        axes[7].scatter(x_valid, y_valid, color=color, s=16, alpha=0.75, label=model.label)
                        x_line = np.linspace(np.nanmin(x_valid), np.nanmax(x_valid), 100)
                        axes[7].plot(x_line, slope * x_line + intercept, color=color, linewidth=1.0)
                        axes[7].set_title("Trophic Amplification", fontsize=TITLE_FONTSIZE, fontweight='bold', pad=5)
                        axes[7].set_xlabel("Relative change in NPP", fontweight='bold')
                        axes[7].set_ylabel("Relative change in SP")
                        axes[7].axhline(0, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
                        axes[7].axvline(0, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
                        self.ta_slopes.append((model.name, slope))

        if exp is not None and ppt is not None:
            min_len = min(len(year), len(exp), len(ppt))
            eratio = exp[:min_len] / ppt[:min_len]
            eratio_norm = self._apply_norm(eratio)
            axes[2].plot(year[:min_len], eratio_norm, color=color, linewidth=LINE_WIDTH)
            axes[2].set_title("Export Ratio anomaly", fontsize=TITLE_FONTSIZE, fontweight='bold', pad=5)
            axes[2].set_ylabel(self._norm_ylabel("Dimensionless"))
            axes[2].axhline(0, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)

        if exp1000 is not None and exp is not None:
            min_len = min(len(year), len(exp1000), len(exp))
            teff = exp1000[:min_len] / exp[:min_len]
            teff_norm = self._apply_norm(teff)
            axes[3].plot(year[:min_len], teff_norm, color=color, linewidth=LINE_WIDTH)
            axes[3].set_title("Transfer Efficiency anomaly", fontsize=TITLE_FONTSIZE, fontweight='bold', pad=5)
            axes[3].set_ylabel(self._norm_ylabel("Dimensionless"))
            axes[3].axhline(0, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)

        if int_data is not None:
            phy_cols = ["COC", "DIA", "FIX", "MIX", "PHA", "PIC"]
            zoo_cols = ["GEL", "CRU", "MES", "PRO", "PTE"]
            phy_parts = [DataLoader.safe_load_column(int_data, col, indices) for col in phy_cols]
            zoo_parts = [DataLoader.safe_load_column(int_data, col, indices) for col in zoo_cols]
            if all(part is not None for part in phy_parts + zoo_parts):
                phy = sum(phy_parts)
                zoo = sum(zoo_parts)
                min_len = min(len(phy), len(zoo))
                x = DataLoader.compute_relative_change(phy[:min_len])
                y = DataLoader.compute_relative_change(zoo[:min_len])
                if x is not None and y is not None:
                    valid = np.isfinite(x) & np.isfinite(y)
                    if np.count_nonzero(valid) >= 2:
                        x_valid = x[valid]
                        y_valid = y[valid]
                        slope, intercept = np.polyfit(x_valid, y_valid, 1)
                        axes[8].scatter(x_valid, y_valid, color=color, s=16, alpha=0.75, label=model.label)
                        x_line = np.linspace(np.nanmin(x_valid), np.nanmax(x_valid), 100)
                        axes[8].plot(x_line, slope * x_line + intercept, color=color, linewidth=1.0)
                        axes[8].set_title("Biomass Amplification", fontsize=TITLE_FONTSIZE, fontweight='bold', pad=5)
                        axes[8].set_xlabel("Relative change in PHY", fontweight='bold')
                        axes[8].set_ylabel("Relative change in ZOO")
                        axes[8].axhline(0, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
                        axes[8].axvline(0, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
                        self.ba_slopes.append((model.name, slope))

        # RLS and ALK-DIC from average file
        ave_data = DataLoader.load_analyser_data(model, "ave", "annual")
        if ave_data is not None:
            rls = DataLoader.safe_load_column(ave_data, "RLS", indices)
            plot_year, plot_rls = DataLoader.align_year_and_values(year, rls)
            if plot_year is not None:
                rls_norm = self._apply_norm(plot_rls)
                axes[4].plot(plot_year, rls_norm, color=color, linewidth=LINE_WIDTH)
                axes[4].set_title("RLS anomaly", fontsize=TITLE_FONTSIZE, fontweight='bold', pad=5)
                axes[4].set_ylabel(self._norm_ylabel("m"))
                axes[4].axhline(0, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)

            alk = DataLoader.safe_load_column(ave_data, "Alkalini", indices)
            dic = DataLoader.safe_load_column(ave_data, "DIC", indices)
            if alk is not None and dic is not None:
                min_len = min(len(year), len(alk), len(dic))
                alk_dic = alk[:min_len] - dic[:min_len]
                alk_dic_norm = self._apply_norm(alk_dic)
                axes[6].plot(year[:min_len], alk_dic_norm, color=color, linewidth=LINE_WIDTH)
                axes[6].set_title("ALK \u2212 DIC anomaly", fontsize=TITLE_FONTSIZE, fontweight='bold', pad=5)
                axes[6].set_ylabel(self._norm_ylabel("\u03bcmol/L"))
                axes[6].set_xlabel("Year", fontweight='bold')
                axes[6].axhline(0, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)


class OrganicMatterPlotter(PlotGenerator):
    """Generates organic matter (DOC, POC, GOC, HOC) summary plots"""

    def generate(self):
        # Use 2x2 grid for 4 organic matter pools
        fig, axes = plt.subplots(
            2, 2, figsize=(2 * SUBPLOT_WIDTH, 2 * SUBPLOT_HEIGHT), sharex=True,
            constrained_layout=USE_CONSTRAINED_LAYOUT
        )
        axes = axes.flatten()

        setup_axes(axes)

        self.plot_all_models(fig, axes, self._plot_model)

        self.add_legend(fig)
        self.save_figure(fig, "mm_organic_matter.png")

    def _plot_model(self, model, axes, color):
        int_data = DataLoader.load_analyser_data(model, "int", "annual")
        if int_data is None:
            return

        actual_years = DataLoader.get_actual_years(int_data)
        indices = model.get_year_range_indices(actual_years)
        if indices[0] is None:
            return

        year = DataLoader.safe_load_column(int_data, "year", indices)
        if year is None:
            return

        om_mapping = [
            ("DOC", "Dissolved Organic Carbon"),
            ("POC", "Particulate Organic Carbon"),
            ("GOC", "Large Particulate OC"),
            ("HOC", "Huge Particulate OC"),
        ]

        for i, (col_name, title) in enumerate(om_mapping):
            values = DataLoader.safe_load_column(int_data, col_name, indices)
            plot_year, plot_values = DataLoader.align_year_and_values(year, values)
            if plot_year is not None:
                label = model.label if i == 1 else None
                axes[i].plot(
                    plot_year,
                    plot_values,
                    color=color,
                    label=label,
                    linewidth=LINE_WIDTH,
                )
                axes[i].set_title(title, fontsize=TITLE_FONTSIZE, fontweight='bold', pad=5)
                axes[i].set_ylabel("PgC")
                # Add xlabel only to bottom row (indices 2, 3 in 2x2 grid)
                if i >= 2:
                    axes[i].set_xlabel("Year", fontweight='bold')


class OrganicMatterNormalizedPlotter(PlotGenerator):
    """Generates normalized/anomaly plots for organic matter pools"""

    def generate(self):
        for mode, filename in [
            ('anomaly', 'mm_organic_matter_normalized.png'),
            ('anompct', 'mm_organic_matter_anompct.png'),
        ]:
            self._norm_mode = mode
            fig, axes = plt.subplots(
                2, 2, figsize=(2 * SUBPLOT_WIDTH, 2 * SUBPLOT_HEIGHT), sharex=True,
                constrained_layout=USE_CONSTRAINED_LAYOUT
            )
            axes = axes.flatten()

            setup_axes(axes)

            self.plot_all_models(fig, axes, self._plot_model)

            self.add_legend(fig)
            self.save_figure(fig, filename)

    def _plot_model(self, model, axes, color):
        int_data = DataLoader.load_analyser_data(model, "int", "annual")
        if int_data is None:
            return

        actual_years = DataLoader.get_actual_years(int_data)
        indices = model.get_year_range_indices(actual_years)
        if indices[0] is None:
            return

        year = DataLoader.safe_load_column(int_data, "year", indices)
        if year is None:
            return

        om_mapping = [
            ("DOC", "DOC anomaly"),
            ("POC", "POC anomaly"),
            ("GOC", "GOC anomaly"),
            ("HOC", "HOC anomaly"),
        ]

        for i, (col_name, title) in enumerate(om_mapping):
            values = DataLoader.safe_load_column(int_data, col_name, indices)
            plot_year, plot_values = DataLoader.align_year_and_values(year, values)
            if plot_year is None:
                continue

            normalized = self._apply_norm(plot_values)
            label = model.label if i == 1 else None
            axes[i].plot(plot_year, normalized, color=color, label=label, linewidth=LINE_WIDTH)
            axes[i].set_title(title, fontsize=TITLE_FONTSIZE, fontweight='bold', pad=5)
            axes[i].set_ylabel(self._norm_ylabel("PgC anomaly"))
            # Add xlabel only to bottom row (indices 2, 3 in 2x2 grid)
            if i >= 2:
                axes[i].set_xlabel("Year", fontweight='bold')
            axes[i].axhline(0, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)


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
            TChlPlotter(self.models, self.save_dir),
            PFTPlotter(self.models, self.save_dir),
            PFTNormalizedPlotter(self.models, self.save_dir),
            PPTByPFTPlotter(self.models, self.save_dir),
            PPTByPFTNormalizedPlotter(self.models, self.save_dir),
            NutrientPlotter(self.models, self.save_dir),
            BenthicPlotter(self.models, self.save_dir),
            PCO2Plotter(self.models, self.save_dir),
            PhysicsPlotter(self.models, self.save_dir),
            DerivedSummaryPlotter(self.models, self.save_dir),
            DerivedSummaryNormalizedPlotter(self.models, self.save_dir),
            OrganicMatterPlotter(self.models, self.save_dir),
            OrganicMatterNormalizedPlotter(self.models, self.save_dir),
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
