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

# Load configuration
try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

# Load visualise_config.toml
def load_config():
    """Load configuration from visualise_config.toml"""
    # Try environment variable first (set by shell script)
    config_path_str = os.environ.get("VISUALISE_CONFIG", "")
    if config_path_str:
        config_path_env = pathlib.Path(config_path_str)
        if config_path_env.exists() and config_path_env.is_file():
            with open(config_path_env, "rb") as f:
                return tomllib.load(f)

    # Try current directory (for when script is copied to output dir)
    config_path = pathlib.Path("visualise_config.toml")
    if config_path.exists():
        with open(config_path, "rb") as f:
            return tomllib.load(f)

    # Try parent directory (visualise/) if not in current dir
    script_dir = pathlib.Path(__file__).parent
    config_path = script_dir.parent / "visualise_config.toml"
    if config_path.exists():
        with open(config_path, "rb") as f:
            return tomllib.load(f)

    print(f"Warning: Config file not found, using defaults")
    return None

CONFIG = load_config()


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
                print(f"Warning: Error calculating indices from actual years: {e}")

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


class PlotConfig:
    """Centralized configuration for plot styling and layout."""

    # Load from config or use defaults
    if CONFIG:
        COLORS = CONFIG.get("colors", {}).get("multimodel_palette", [
            "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
            "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
        ])
        RATIO = CONFIG.get("multimodel", {}).get("figure_ratio", 2.5)
        FONT_FAMILY = "sans-serif"
        TITLE_FONTSIZE = CONFIG.get("style", {}).get("fonts", {}).get("title", 8)
        LABEL_FONTSIZE = CONFIG.get("style", {}).get("fonts", {}).get("axis_label", 10)
        LINE_WIDTH = CONFIG.get("style", {}).get("linewidth", 1.5)

        # Grid style from config
        GRID_STYLE = {
            "linestyle": CONFIG.get("style", {}).get("grid_linestyle", "--"),
            "linewidth": CONFIG.get("style", {}).get("grid_linewidth", 0.5),
            "alpha": 0.7
        }

        # Hatch style for observations (preferred multimodel style)
        obs_config = CONFIG.get("style", {}).get("observations", {})
        HATCH_STYLE = {
            "color": obs_config.get("hatch_color", "k"),
            "alpha": obs_config.get("hatch_alpha", 0.15),
            "fill": obs_config.get("hatch_fill", False),
            "hatch": obs_config.get("hatch_pattern", "///")
        }

        # Line style for observation lines
        LINE_STYLE = {
            "color": obs_config.get("line_color", "k"),
            "linestyle": obs_config.get("line_linestyle", "--"),
            "alpha": obs_config.get("line_alpha", 0.8),
            "linewidth": obs_config.get("line_linewidth", 1.5)
        }
    else:
        # Fallback defaults if config not found
        COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
                  "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
        RATIO = 2.5
        FONT_FAMILY = "sans-serif"
        TITLE_FONTSIZE = 8
        LABEL_FONTSIZE = 10
        LINE_WIDTH = 1.5
        GRID_STYLE = {"linestyle": "--", "linewidth": 0.5, "alpha": 0.7}
        HATCH_STYLE = {"color": "k", "alpha": 0.15, "fill": False, "hatch": "///"}
        LINE_STYLE = {"color": "k", "linestyle": "dashed", "alpha": 0.8, "linewidth": 1.5}

    @staticmethod
    def setup_axes(axes):
        """Apply consistent styling to all axes in a figure."""
        for ax in axes:
            ax.grid(**PlotConfig.GRID_STYLE)
            ax.tick_params(
                axis="both", which="major", labelsize=PlotConfig.LABEL_FONTSIZE - 1
            )
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.xaxis.label.set_size(PlotConfig.LABEL_FONTSIZE)
            ax.yaxis.label.set_size(PlotConfig.LABEL_FONTSIZE)


class ObservationalData:
    """Container for observational data ranges"""

    GLOBAL_RANGES = {
        "TCHL": {"value": 0.2921, "type": "line"},
        "PPT": {"min": 51, "max": 65, "type": "span"},
        "EXP": {"min": 5., "max": 12., "type": "span"},
        "PROSIL": {"min": 203, "max": 307, "type": "span"},
        "SINKSIL": {"min": 89, "max": 135, "type": "span"},
    }

    PFT_RANGES = {
        "PIC": {"min": 0.28, "max": 0.52},
        "PHA": {"min": 0.11, "max": 0.69},
        "DIA": {"min": 0.013, "max": 0.75},
        "COC": {"min": 0.001, "max": 0.032},
        "FIX": {"min": 0.008, "max": 0.12},
        "GEL": {"min": 0.10, "max": 3.11},
        "PRO": {"min": 0.10, "max": 0.37},
        "MAC": {"min": 0.01, "max": 0.64},
        "MES": {"min": 0.21, "max": 0.34},
        "BAC": {"min": 0.25, "max": 0.26},
        "MIX": {"min": np.nan, "max": np.nan},
        "PTE": {"min": 0.048, "max":0.057}
    }

    NUTRIENT_VALUES = {
        "PO4": 0.530,
        "NO3": 5.152,
        "Fer": np.nan,
        "Si": 7.485,
        "O2": 251.1,
    }

    PCO2_MONTHLY = {
        "global": np.array(
            [
                374.5267,
                376.9849,
                378.6273,
                377.6980,
                374.4194,
                372.0030,
                372.8390,
                373.1397,
                373.7141,
                374.7667,
                375.3111,
                376.1471,
            ]
        ),
        "reg1": np.array(
            [
                367.6449,
                374.6633,
                376.1620,
                367.9230,
                348.2934,
                327.9850,
                319.9174,
                315.5227,
                320.6641,
                336.6024,
                353.5276,
                366.8002,
            ]
        ),
        "reg2": np.array(
            [
                360.3107,
                359.7239,
                360.9054,
                364.4423,
                372.4738,
                384.9409,
                398.0022,
                403.2199,
                398.5489,
                386.5432,
                374.1430,
                366.2098,
            ]
        ),
        "reg3": np.array(
            [
                399.8266,
                401.4656,
                403.5920,
                404.5923,
                404.1028,
                402.5178,
                401.7164,
                401.2903,
                401.1736,
                401.0325,
                400.8897,
                401.5201,
            ]
        ),
        "reg4": np.array(
            [
                383.5775,
                385.4783,
                381.7295,
                373.6719,
                366.9351,
                362.8020,
                361.2315,
                360.9230,
                361.6792,
                363.8154,
                368.7272,
                377.6358,
            ]
        ),
        "reg5": np.array(
            [
                360.1319,
                361.5651,
                368.3551,
                376.4319,
                381.8729,
                387.2647,
                391.9770,
                394.6131,
                394.8327,
                390.6098,
                380.8251,
                368.3117,
            ]
        ),
    }

    TCHL_MONTHLY = {
        "global": np.array(
            [
                0.0,
                0.00132746,
                -0.00287947,
                0.05538714,
                0.19660503,
                0.25430918,
                0.25188863,
                0.2225644,
                0.20896989,
                0.0965862,
                0.00146702,
                0.00250745,
            ]
        ),
        "reg1": np.array(
            [
                0.0000000e00,
                -9.6781135e-02,
                -3.8862228e-05,
                1.1393067e00,
                2.7156515e00,
                3.1558909e00,
                2.8647695e00,
                2.6039510e00,
                3.1784720e00,
                2.7834001e00,
                1.4468776e00,
                -1.5452981e-02,
            ]
        ),
        "reg2": np.array(
            [
                0.0,
                0.05065274,
                0.0375762,
                -0.08406973,
                -0.12498319,
                -0.22558725,
                -0.23778045,
                -0.22738922,
                0.11215413,
                0.08687639,
                0.2218734,
                0.09384,
            ]
        ),
        "reg3": np.array(
            [
                0.0,
                -0.04654008,
                -0.10164082,
                -0.10021466,
                0.000534,
                0.01825154,
                0.04680908,
                -0.00357324,
                -0.06979609,
                -0.09147316,
                -0.02505821,
                -0.04246014,
            ]
        ),
        "reg4": np.array(
            [
                0.0,
                0.03992385,
                0.06925935,
                0.09288526,
                0.10391548,
                0.10491946,
                0.12549761,
                0.11763752,
                0.10455954,
                0.10054898,
                0.05817935,
                0.03453296,
            ]
        ),
        "reg5": np.array(
            [
                0.0,
                -0.06191447,
                -0.13221624,
                0.01676586,
                0.1082058,
                0.04452744,
                0.02129921,
                0.10751635,
                0.1628111,
                0.15176588,
                0.16497672,
                0.03976542,
            ]
        ),
    }


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
    def load_breakdown_data(model_config, data_type, frequency="annual"):
        # Try CSV format first (new format), fall back to TSV if not found
        csv_path = f"/{model_config.model_dir}/{model_config.name}/breakdown.{data_type}.{frequency}.csv"
        dat_path = f"/{model_config.model_dir}/{model_config.name}/breakdown.{data_type}.{frequency}.dat"

        try:
            # New CSV format - single header row, comma-separated
            df = pd.read_csv(csv_path)
            if df.empty:
                print(f"Warning: Empty dataframe loaded from {csv_path}")
                return None
            print(f"Loaded CSV {csv_path}")
        except FileNotFoundError:
            try:
                # Legacy TSV format - 3 header rows, tab-separated
                # Row 0: variable names, Row 1: units, Row 2: keys
                # Use header=0 to get the variable names
                df = pd.read_csv(dat_path, sep="\t", header=0, skiprows=[1, 2])
                if df.empty:
                    print(f"Warning: Empty dataframe loaded from {dat_path}")
                    return None
                print(f"Loaded TSV {dat_path}")
            except FileNotFoundError:
                print(f"Warning: File not found - tried both {csv_path} and {dat_path}")
                return None
            except Exception as e:
                print(f"Warning: Error loading {dat_path}: {e}")
                return None
        except Exception as e:
            print(f"Warning: Error loading {csv_path}: {e}")
            return None

        # Sort by year if year column exists
        if 'year' in df.columns:
            df = df.sort_values('year').reset_index(drop=True)
        # For monthly data, sort by year and month if both exist
        elif 'month' in df.columns or 'Month' in df.columns:
            month_col = 'month' if 'month' in df.columns else 'Month'
            if 'year' in df.columns:
                df = df.sort_values(['year', month_col]).reset_index(drop=True)
            else:
                df = df.sort_values(month_col).reset_index(drop=True)

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
        plt.rcParams["font.family"] = PlotConfig.FONT_FAMILY

    def add_legend(self, fig, **kwargs):
        """
        MODIFIED: Add a legend to the bottom of the plot area.
        """
        legend_kwargs = {
            "loc": "upper center",  # Anchor the legend at its upper center
            "bbox_to_anchor": (
                0.5,
                0.05,
            ),  # Position the anchor at the bottom center of the figure
            "fontsize": PlotConfig.LABEL_FONTSIZE - 2,
            "ncol": 5,  # Arrange legend items horizontally, adjust as needed
            "frameon": True,
        }
        legend_kwargs.update(kwargs)
        # We get handles and labels from the figure's axes
        handles, labels = [], []
        for ax in fig.get_axes():
            h, l = ax.get_legend_handles_labels()
            handles.extend(h)
            labels.extend(l)

        # Remove duplicate labels/handles
        by_label = dict(zip(labels, handles))
        if by_label:  # Only add legend if there are items to show
            fig.legend(by_label.values(), by_label.keys(), **legend_kwargs)

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
        print(f"Created {filename}")
        plt.close(fig)

    def plot_all_models(self, fig, axes, plot_func):
        successful_plots = 0
        for i, model in enumerate(self.models):
            color = PlotConfig.COLORS[i % len(PlotConfig.COLORS)]
            try:
                plot_func(model, axes, color)
                successful_plots += 1
            except Exception as e:
                print(f"Error plotting {model.name}: {e}")
                import traceback

                if hasattr(self, "debug") and self.debug:
                    traceback.print_exc()

        if successful_plots == 0:
            print(
                f"Warning: No models were successfully plotted for {self.__class__.__name__}"
            )
        else:
            print(f"Successfully plotted {successful_plots}/{len(self.models)} models")


class GlobalSummaryPlotter(PlotGenerator):
    """Generates global summary plots"""

    def generate(self):
        # REWRITTEN: Use plt.subplots() to create the figure and axes grid directly.
        fig, axes = plt.subplots(
            2, 3, figsize=(3 * PlotConfig.RATIO, 2 * PlotConfig.RATIO), sharex=True
        )
        axes = axes.flatten()  # Flatten the 2D axes array for easy iteration.

        PlotConfig.setup_axes(axes)

        self.plot_all_models(fig, axes, self._plot_model)
        self._add_observational_data(axes)

        self.add_legend(fig)
        self.save_figure(fig, "multimodel_summary_global.png")

    def _plot_model(self, model, axes, color):
        sur_data = DataLoader.load_breakdown_data(model, "sur", "annual")
        if sur_data is None:
            return

        actual_years = DataLoader.get_actual_years(sur_data)
        indices = model.get_year_range_indices(actual_years)
        if indices[0] is None:
            return

        vol_data = DataLoader.load_breakdown_data(model, "vol", "annual")
        lev_data = DataLoader.load_breakdown_data(model, "lev", "annual")
        ave_data = DataLoader.load_breakdown_data(model, "ave", "annual")

        year = DataLoader.safe_load_column(sur_data, "year", indices)
        if year is None:
            return

        plot_configs = [
            (axes[0], sur_data, "Cflx", "Cflx [PgC/yr]", False),
            (axes[1], ave_data, "TChl", "TChl [ug/L]", True),
            (axes[2], vol_data, "PPT", "PPT [PgC/yr]", False),
            (axes[3], lev_data, "EXP", "EXP@100 [PgC/yr]", False),
            (axes[4], vol_data, "probsi", "PROSi [Tmol/yr]", False),
            (axes[5], lev_data, "sinksil", "SNKSi [Tmol/yr]", False),
        ]

        for ax, data_df, column, title, add_label in plot_configs:
            if data_df is not None:
                values = DataLoader.safe_load_column(data_df, column, indices)
                if values is not None and len(values) == len(year):
                    label = model.label if add_label else None
                    ax.plot(
                        year,
                        values,
                        color=color,
                        label=label,
                        linewidth=PlotConfig.LINE_WIDTH,
                    )
                    ax.set_title(title, fontsize=PlotConfig.TITLE_FONTSIZE)

    def _add_observational_data(self, axes):
        obs_mapping = {1: "TCHL", 2: "PPT", 3: "EXP", 4: "PROSIL", 5: "SINKSIL"}
        for i, (key, ax) in enumerate(
            zip(obs_mapping, [axes[1], axes[2], axes[3], axes[4], axes[5]])
        ):
            obs_key = obs_mapping[i + 1]
            if obs_key in ObservationalData.GLOBAL_RANGES:
                obs = ObservationalData.GLOBAL_RANGES[obs_key]
                if obs["type"] == "line":
                    ax.axhline(obs["value"], **PlotConfig.LINE_STYLE)
                else:
                    ax.axhspan(obs["min"], obs["max"], **PlotConfig.HATCH_STYLE)


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
                linewidth=PlotConfig.LINE_WIDTH,
            )


class CflxPlotter(RegionalPlotter):
    """Generates Cflx summary plots"""

    def generate(self):
        """
        Generates and saves the Cflx summary plot with a balanced, nested grid layout.
        """
        fig = plt.figure(figsize=(3.5 * PlotConfig.RATIO, 2 * PlotConfig.RATIO))

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

        PlotConfig.setup_axes(axes)
        # --- MODIFICATION END ---

        self.plot_all_models(fig, axes, self._plot_model)

        self.add_legend(fig)
        self.save_figure(fig, "multimodel_summary_cflx.png")

    def _plot_model(self, model, axes, color):
        sur_annual = DataLoader.load_breakdown_data(model, "sur", "annual")
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
                year, cflx_annual, color=color, linewidth=PlotConfig.LINE_WIDTH
            )
            axes[0].set_title(
                "Surface Cflx (Global) [PgC/yr]", fontsize=PlotConfig.TITLE_FONTSIZE
            )

        sur_monthly = DataLoader.load_breakdown_data(model, "sur", "monthly")
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
                fontsize=PlotConfig.TITLE_FONTSIZE - 2,
                transform=ax.transAxes,
            )


class PFTPlotter(PlotGenerator):
    """Generates PFT summary plots"""

    def generate(self):
        # REWRITTEN: Use plt.subplots() to create the grid and handle unused axes.
        fig, axes = plt.subplots(
            3,
            4,
            figsize=(4.5 * PlotConfig.RATIO, 2.5 * PlotConfig.RATIO),
            sharex=True,
        )
        flat_axes = axes.flatten()

        self._setup_pft_axes(flat_axes)

        self.plot_all_models(fig, flat_axes, self._plot_model)
        self._add_observational_ranges(flat_axes)

        self.add_legend(fig)
        self.save_figure(fig, "multimodel_summary_pfts.png")

    def _setup_pft_axes(self, axes):
        PlotConfig.setup_axes(axes)
        for i, ax in enumerate(axes):
            if i != 9:
                ax.set_ylim(0, 0.5)
        axes[9].set_ylim(bottom=0)

    def _plot_model(self, model, axes, color):
        int_data = DataLoader.load_breakdown_data(model, "int", "annual")
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
            ("MIX", "Mixed Phytoplankton"),
            ("DIA", "Diatoms"),
            ("COC", "Coccolithophores"),
            ("FIX", "N2-fixers"),
            ("BAC", "Bacteria"),
            ("GEL", "Jellyfish"),
            ("PRO", "Microzooplankton"),
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
                    linewidth=PlotConfig.LINE_WIDTH,
                )
                axes[i].set_title(f"{title} [PgC]", fontsize=PlotConfig.TITLE_FONTSIZE)

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
            9: "MAC",
            10: "PTE",
            11: "MES",
        }
        for idx, pft_name in obs_indices.items():
            if pft_name in ObservationalData.PFT_RANGES:
                ranges = ObservationalData.PFT_RANGES[pft_name]
                axes[idx].axhspan(
                    ranges["min"], ranges["max"], **PlotConfig.HATCH_STYLE
                )


class TChlPlotter(RegionalPlotter):
    """Generates TChl summary plots"""

    def generate(self):
        """
        Generates and saves the TChl summary plot with a balanced, nested grid layout.
        """
        fig = plt.figure(figsize=(3.5 * PlotConfig.RATIO, 2 * PlotConfig.RATIO))

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

        PlotConfig.setup_axes(axes)

        self.plot_all_models(fig, axes, self._plot_model)
        self._add_observational_data(axes)  # MODIFICATION: This line was added

        self.add_legend(fig)
        self.save_figure(fig, "multimodel_summary_tchl.png")

    def _plot_model(self, model, axes, color):
        ave_annual = DataLoader.load_breakdown_data(model, "ave", "annual")
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
                year, tchl_annual, color=color, linewidth=PlotConfig.LINE_WIDTH
            )
            axes[0].set_title(
                "Average TChl (Global) [ug Chl/L]", fontsize=PlotConfig.TITLE_FONTSIZE
            )

        ave_monthly = DataLoader.load_breakdown_data(model, "ave", "monthly")
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
                fontsize=PlotConfig.TITLE_FONTSIZE - 2,
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
            data = ObservationalData.TCHL_MONTHLY[region]
            normalized = data - data[0]
            label = "data-products" if i == 0 else None
            ax.plot(
                month_names,
                normalized,
                color="k",
                linestyle="dashed",
                label=label,
                linewidth=PlotConfig.LINE_WIDTH,
            )
            ax.text(
                0.8,
                0.9,
                f"{title}",
                fontsize=PlotConfig.TITLE_FONTSIZE - 2,
                transform=ax.transAxes,
            )


class NutrientPlotter(PlotGenerator):
    """Generates nutrient summary plots"""

    def generate(self):
        # REWRITTEN: Use plt.subplots() to create the grid and handle unused axes.
        fig, axes = plt.subplots(
            2, 3, figsize=(3.5 * PlotConfig.RATIO, 2 * PlotConfig.RATIO), sharex=True
        )
        flat_axes = axes.flatten()

        # Turn off the last unused subplot in the 2x3 grid
        flat_axes[-1].axis("off")

        flat_axes = flat_axes[:5]  # We only need the first 5 axes

        PlotConfig.setup_axes(flat_axes)

        self.plot_all_models(fig, flat_axes, self._plot_model)
        self._add_observational_data(flat_axes)

        self.add_legend(fig)
        self.save_figure(fig, "multimodel_summary_nutrients.png")

    def _plot_model(self, model, axes, color):
        ave_data = DataLoader.load_breakdown_data(model, "ave", "annual")
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
            ("PO4", axes[0], "Avg Surface Phosphate [umol/L]", 1 / 122, False),
            ("NO3", axes[1], "Avg Surface Nitrates [umol/L]", 1, True),
            ("Fer", axes[2], "Avg Surface Iron [nmol/L]", 1000, False),
            ("Si", axes[3], "Avg Surface Silica [umol/L]", 1, False),
            ("O2", axes[4], "Avg Surface Oxygen [umol/L]", 1, False),
        ]

        for col_name, ax, title, scale, add_label in plot_configs:
            values = DataLoader.safe_load_column(ave_data, col_name, indices)
            if values is not None and len(values) == len(year):
                label = model.label if add_label else None
                ax.plot(
                    year,
                    values * scale,
                    color=color,
                    label=label,
                    linewidth=PlotConfig.LINE_WIDTH,
                )
                ax.set_title(title, fontsize=PlotConfig.TITLE_FONTSIZE)

    def _add_observational_data(self, axes):
        for i, key in enumerate(ObservationalData.NUTRIENT_VALUES.keys()):
            if i < len(axes):
                axes[i].axhline(
                    ObservationalData.NUTRIENT_VALUES[key], **PlotConfig.LINE_STYLE
                )


class PCO2Plotter(RegionalPlotter):
    """Generates pCO2 summary plots"""

    def generate(self):
        """
        Generates and saves the pCO2 summary plot with a balanced, nested grid layout.
        """
        fig = plt.figure(figsize=(3.5 * PlotConfig.RATIO, 2 * PlotConfig.RATIO))

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

        PlotConfig.setup_axes(axes)
        # --- MODIFICATION END ---

        self.plot_all_models(fig, axes, self._plot_model)
        self._add_observational_data(axes)

        self.add_legend(fig)
        self.save_figure(fig, "multimodel_summary_pco2.png")

    def _plot_model(self, model, axes, color):
        ave_annual = DataLoader.load_breakdown_data(model, "ave", "annual")
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
                year, pco2_annual, color=color, linewidth=PlotConfig.LINE_WIDTH
            )
            axes[0].set_title(
                "Avg Surface pCO2 (Global) [ppm]", fontsize=PlotConfig.TITLE_FONTSIZE
            )

        ave_monthly = DataLoader.load_breakdown_data(model, "ave", "monthly")
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
            data = ObservationalData.PCO2_MONTHLY[region]
            normalized = data - data[0]
            label = "data-products" if i == 0 else None
            ax.plot(
                month_names,
                normalized,
                color="k",
                linestyle="dashed",
                label=label,
                linewidth=PlotConfig.LINE_WIDTH,
            )
            # Use ax.text to prevent titles from overlapping in the new layout
            ax.text(
                0.8,
                0.9,
                f"{title}",
                fontsize=PlotConfig.TITLE_FONTSIZE - 2,
                transform=ax.transAxes,
            )


class PhysicsPlotter(PlotGenerator):
    """Generates physics summary plots"""

    def generate(self):
        # REWRITTEN: Use plt.subplots() to create the figure and axes grid directly.
        fig, axes = plt.subplots(
            2, 2, figsize=(3 * PlotConfig.RATIO, 1.5 * PlotConfig.RATIO), sharex=True
        )
        axes = axes.flatten()  # Flatten for consistent iteration pattern

        PlotConfig.setup_axes(axes)

        self.plot_all_models(fig, axes, self._plot_model)

        self.add_legend(fig)
        self.save_figure(fig, "multimodel_summary_physics.png")

    def _plot_model(self, model, axes, color):
        ave_data = DataLoader.load_breakdown_data(model, "ave", "annual")
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
            axes[0].plot(year, sst, color=color, linewidth=PlotConfig.LINE_WIDTH)
            axes[0].set_title(
                "Avg Sea Surface Temp [degC]", fontsize=PlotConfig.TITLE_FONTSIZE
            )
            axes[2].plot(
                year, sst - sst[0], color=color, linewidth=PlotConfig.LINE_WIDTH
            )
            axes[2].set_title(
                "Normalised Avg SST [degC]", fontsize=PlotConfig.TITLE_FONTSIZE
            )

        if sss is not None:
            axes[1].plot(
                year,
                sss,
                color=color,
                label=model.label,
                linewidth=PlotConfig.LINE_WIDTH,
            )
            axes[1].set_title(
                "Avg Sea Surface Salinity [1e-3]", fontsize=PlotConfig.TITLE_FONTSIZE
            )

        if mld is not None:
            axes[3].plot(year, mld, color=color, linewidth=PlotConfig.LINE_WIDTH)
            axes[3].set_title(
                "Avg Mixed Layer Depth [m]", fontsize=PlotConfig.TITLE_FONTSIZE
            )


class MultiModelPlotter:
    """Main class coordinating all plot generation"""

    def __init__(self, model_csv_path: str, save_dir: str, debug: bool = False):
        self.models = DataLoader.load_model_configs(model_csv_path)
        self.save_dir = save_dir
        self.debug = debug

        print(f"Loaded {len(self.models)} models:")
        for model in self.models:
            print(f"  - {model.name}: {model.description}")

    def generate_all_plots(self):
        plotters = [
            GlobalSummaryPlotter(self.models, self.save_dir),
            CflxPlotter(self.models, self.save_dir),
            TChlPlotter(self.models, self.save_dir),
            PFTPlotter(self.models, self.save_dir),
            NutrientPlotter(self.models, self.save_dir),
            PCO2Plotter(self.models, self.save_dir),
            PhysicsPlotter(self.models, self.save_dir),
        ]

        for plotter in plotters:
            if self.debug:
                plotter.debug = True
            try:
                print(f"\nGenerating {plotter.__class__.__name__}...")
                plotter.generate()
            except Exception as e:
                print(f"Error generating {plotter.__class__.__name__}: {e}")
                if self.debug:
                    import traceback

                    traceback.print_exc()


def main():
    if len(sys.argv) < 2:
        print("Usage: python script.py <save_directory> [--debug]")
        sys.exit(1)

    model_csv = "modelsToPlot.csv"
    save_dir = sys.argv[1]
    debug = "--debug" in sys.argv

    plotter = MultiModelPlotter(model_csv, save_dir, debug=debug)
    plotter.generate_all_plots()


if __name__ == "__main__":
    main()

