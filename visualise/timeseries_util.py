#!/usr/bin/env python3
"""
Common utilities for timeseries visualization scripts.
Shared between single-model and multi-model timeseries plotting.
"""

import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple

import config_utils


@dataclass
class ObservationRange:
    """Represents an observational data range to plot as a shaded region."""
    min_val: float
    max_val: float
    label: str = "Obs. Range"


@dataclass
class ObservationLine:
    """Represents an observational data point to plot as a horizontal line."""
    value: float
    label: str = "Observation"


# Regions for the seasonal TChl summary, shared by the single- and multi-model
# timeseries plotters so both stay in step. lon_range is (min, max) in -180..180
# (None = all longitudes); a min > max wraps across the dateline.
TCHL_REGION_DEFS = [
    {"title": "N. Pacific subtropical",      "lat_range": (15.0, 35.0),   "lon_range": (160.0, -130.0)},
    {"title": "N. Pacific subpolar",         "lat_range": (50.0, 62.0),   "lon_range": (160.0, -140.0)},
    {"title": "N. Atlantic subtropical",     "lat_range": (20.0, 35.0),   "lon_range": (-80.0, -20.0)},
    {"title": "N. Atlantic subpolar",        "lat_range": (45.0, 65.0),   "lon_range": (-60.0, -10.0)},
    {"title": "Equatorial Pacific",          "lat_range": (-5.0, 5.0),    "lon_range": (160.0, -90.0)},
    {"title": "S. Pacific subtropical",      "lat_range": (-35.0, -15.0), "lon_range": (170.0, -100.0)},
    {"title": "S. Atlantic subtropical",     "lat_range": (-30.0, -10.0), "lon_range": (-50.0, 10.0)},
    {"title": "Sub-Antarctic Zone (Indian)", "lat_range": (-50.0, -40.0), "lon_range": (20.0, 120.0)},
    {"title": "Antarctic Zone",              "lat_range": (-65.0, -55.0), "lon_range": None},
]


class ObservationData:
    """
    Centralized container for all observational data values.
    These values are literature-based constraints for model validation.
    Loads from visualise_config.toml [observations] section.
    """

    # Class-level cache for loaded data
    _loaded = False
    _config = None

    # Will be populated from config
    GLOBAL = {}
    PFT = {}
    NUTRIENTS = {}
    DERIVED = {}
    BENTHIC = {}
    PHYSICS = {}
    ORGANIC_CARBON = {}

    @classmethod
    def configure(cls, config):
        """Inject the per-run visualise_config and (re)load observation refs.

        Must be called once with the run's config before any get_* accessor;
        there is no ambient config discovery.
        """
        cls._config = config
        cls._loaded = False
        cls._ensure_loaded()

    @classmethod
    def _ensure_loaded(cls):
        """Populate the observation reference tables from the injected config."""
        if cls._loaded:
            return

        if cls._config is None:
            raise RuntimeError(
                "ObservationData is not configured; call "
                "ObservationData.configure(config) with the per-run visualise_config "
                "before reading observation references."
            )
        obs_config = cls._config.get('observations', {})

        # Load global ecosystem observations
        global_config = obs_config.get('global', {})
        cls.GLOBAL = {}
        for key, value in global_config.items():
            if isinstance(value, list) and len(value) == 2:
                cls.GLOBAL[key] = {"min": value[0], "max": value[1], "type": "span"}
            else:
                cls.GLOBAL[key] = {"value": value, "type": "line"}

        # Load PFT observations
        pft_config = obs_config.get('pfts', {})
        cls.PFT = {}
        for key, value in pft_config.items():
            if isinstance(value, list) and len(value) == 2:
                cls.PFT[key] = {"min": value[0], "max": value[1], "type": "range"}
            elif isinstance(value, (int, float)):
                cls.PFT[key] = {"value": value, "type": "line"}
            else:
                cls.PFT[key] = {"min": None, "max": None, "type": None}
        # Add MIX with no observations if not in config
        if "MIX" not in cls.PFT:
            cls.PFT["MIX"] = {"min": None, "max": None, "type": None}

        # Load nutrient observations
        cls.NUTRIENTS = obs_config.get('nutrients', {})

        # Load derived variable observations
        cls.DERIVED = obs_config.get('derived', {})

        # Load benthic (deep-ocean) observations
        cls.BENTHIC = obs_config.get('benthic', {})

        # Load physics observations
        physics_config = obs_config.get('physics', {})
        cls.PHYSICS = {}
        for key, value in physics_config.items():
            if isinstance(value, list) and len(value) == 2:
                cls.PHYSICS[key] = {"min": value[0], "max": value[1], "type": "range"}
            elif isinstance(value, (int, float)):
                cls.PHYSICS[key] = {"value": value, "type": "line"}

        # Load organic carbon observations
        oc_config = obs_config.get('organic_carbon', {})
        cls.ORGANIC_CARBON = {}
        for key, value in oc_config.items():
            if isinstance(value, list) and len(value) == 2:
                cls.ORGANIC_CARBON[key] = {"min": value[0], "max": value[1], "type": "span"}
            elif isinstance(value, (int, float)):
                cls.ORGANIC_CARBON[key] = {"value": value, "type": "line"}

        cls._loaded = True

    @classmethod
    def get_global(cls) -> Dict[str, Any]:
        """Get global ecosystem observation data."""
        cls._ensure_loaded()
        return cls.GLOBAL

    @classmethod
    def get_pft(cls) -> Dict[str, Any]:
        """Get PFT observation data."""
        cls._ensure_loaded()
        return cls.PFT

    @classmethod
    def get_nutrients(cls) -> Dict[str, Any]:
        """Get nutrient observation data."""
        cls._ensure_loaded()
        return cls.NUTRIENTS

    @classmethod
    def get_derived(cls) -> Dict[str, Any]:
        """Get derived variable observation data."""
        cls._ensure_loaded()
        return cls.DERIVED

    @classmethod
    def get_benthic(cls) -> Dict[str, Any]:
        """Get benthic (deep-ocean) observation data."""
        cls._ensure_loaded()
        return cls.BENTHIC

    @classmethod
    def get_physics(cls) -> Dict[str, Any]:
        """Get physics observation data (e.g., AMOC from RAPID)."""
        cls._ensure_loaded()
        return cls.PHYSICS

    @classmethod
    def get_organic_carbon(cls) -> Dict[str, Any]:
        """Get organic carbon observation data."""
        cls._ensure_loaded()
        return cls.ORGANIC_CARBON

    # Monthly pCO2 data by region (from data products)
    PCO2_MONTHLY = {
        "global": np.array([
            374.5267, 376.9849, 378.6273, 377.6980, 374.4194, 372.0030,
            372.8390, 373.1397, 373.7141, 374.7667, 375.3111, 376.1471
        ]),
        "reg1": np.array([  # 45N-90N
            367.6449, 374.6633, 376.1620, 367.9230, 348.2934, 327.9850,
            319.9174, 315.5227, 320.6641, 336.6024, 353.5276, 366.8002
        ]),
        "reg2": np.array([  # 15N-45N
            360.3107, 359.7239, 360.9054, 364.4423, 372.4738, 384.9409,
            398.0022, 403.2199, 398.5489, 386.5432, 374.1430, 366.2098
        ]),
        "reg3": np.array([  # 15S-15N
            399.8266, 401.4656, 403.5920, 404.5923, 404.1028, 402.5178,
            401.7164, 401.2903, 401.1736, 401.0325, 400.8897, 401.5201
        ]),
        "reg4": np.array([  # 45S-15S
            383.5775, 385.4783, 381.7295, 373.6719, 366.9351, 362.8020,
            361.2315, 360.9230, 361.6792, 363.8154, 368.7272, 377.6358
        ]),
        "reg5": np.array([  # 90S-45S
            360.1319, 361.5651, 368.3551, 376.4319, 381.8729, 387.2647,
            391.9770, 394.6131, 394.8327, 390.6098, 380.8251, 368.3117
        ]),
    }

    # Monthly TChl anomaly data by region (from data products)
    TCHL_MONTHLY = {
        "global": np.array([
            0.0, 0.00132746, -0.00287947, 0.05538714, 0.19660503, 0.25430918,
            0.25188863, 0.2225644, 0.20896989, 0.0965862, 0.00146702, 0.00250745
        ]),
        "reg1": np.array([  # 45N-90N
            0.0, -9.6781135e-02, -3.8862228e-05, 1.1393067, 2.7156515, 3.1558909,
            2.8647695, 2.6039510, 3.1784720, 2.7834001, 1.4468776, -1.5452981e-02
        ]),
        "reg2": np.array([  # 15N-45N
            0.0, 0.05065274, 0.0375762, -0.08406973, -0.12498319, -0.22558725,
            -0.23778045, -0.22738922, 0.11215413, 0.08687639, 0.2218734, 0.09384
        ]),
        "reg3": np.array([  # 15S-15N
            0.0, -0.04654008, -0.10164082, -0.10021466, 0.000534, 0.01825154,
            0.04680908, -0.00357324, -0.06979609, -0.09147316, -0.02505821, -0.04246014
        ]),
        "reg4": np.array([  # 45S-15S
            0.0, 0.03992385, 0.06925935, 0.09288526, 0.10391548, 0.10491946,
            0.12549761, 0.11763752, 0.10455954, 0.10054898, 0.05817935, 0.03453296
        ]),
        "reg5": np.array([  # 90S-45S
            0.0, -0.06191447, -0.13221624, 0.01676586, 0.1082058, 0.04452744,
            0.02129921, 0.10751635, 0.1628111, 0.15176588, 0.16497672, 0.03976542
        ]),
    }


class ConfigLoader:
    """Loads the per-run visualization config (delegates to config_utils)."""

    @staticmethod
    def load_config(run_dir=None, config_path=None) -> Dict[str, Any]:
        """Parse the per-run visualise_config.

        Provide ``run_dir`` (resolved from its setUpData) or an explicit
        ``config_path``. There is no environment-variable or cwd discovery, so a
        misconfigured run fails loudly rather than guessing the grid/obs.
        """
        return config_utils.load_config(run_dir=run_dir, config_path=config_path)


class DataFileLoader:
    """Handles loading analyser data files in CSV or legacy TSV format."""

    @staticmethod
    def read_analyser_file(base_dir: pathlib.Path, model_name: str,
                           file_type: str, frequency: str = "annual") -> Optional[pd.DataFrame]:
        """
        Read an analyser file, trying multiple formats for backwards compatibility.

        Args:
            base_dir: Base directory containing model output
            model_name: Name of the model run
            file_type: Type of file (sur, vol, lev, ave, int)
            frequency: Data frequency (annual or monthly)

        Returns:
            DataFrame with the data, or None if file not found
        """
        # Try analyser files first (higher priority), then breakdown files
        # Within each, try CSV format before DAT format
        analyser_paths = [
            (base_dir / model_name / f"analyser.{file_type}.{frequency}.csv", "csv"),
            (base_dir / model_name / f"analyser.{file_type}.{frequency}.dat", "dat"),
        ]
        breakdown_paths = [
            (base_dir / model_name / f"breakdown.{file_type}.{frequency}.csv", "csv"),
            (base_dir / model_name / f"breakdown.{file_type}.{frequency}.dat", "dat"),
        ]

        # Try all analyser files first, then breakdown files
        for path, fmt in analyser_paths + breakdown_paths:
            try:
                if fmt == "csv":
                    try:
                        df = pd.read_csv(path)
                    except pd.errors.ParserError:
                        import warnings
                        warnings.warn(
                            f"Column mismatch in '{path}' (analyser config may have changed). "
                            f"Rows with extra columns skipped. Re-run the analyser to fix."
                        )
                        df = pd.read_csv(path, on_bad_lines='skip')
                else:
                    # Legacy TSV format - 3 header rows, tab-separated
                    # Row 0: variable names, Row 1: units, Row 2: keys
                    # Use header=0 to get variable names, skip rows 1 and 2
                    df = pd.read_csv(path, sep="\t", header=0, skiprows=[1, 2])
                return df
            except FileNotFoundError:
                continue

        # No file found
        return None

    @staticmethod
    def extract_columns(df: pd.DataFrame, columns: list,
                       skip_rows: int = 0) -> Dict[str, np.ndarray]:
        """
        Extract columns as numpy arrays.

        Args:
            df: DataFrame to extract from
            columns: List of column names to extract
            skip_rows: Number of rows to skip (default 0, legacy parameter)

        Returns:
            Dictionary mapping column names to numpy arrays
        """
        data = {}
        for col in columns:
            if col in df.columns:
                data[col] = df[col][skip_rows:].to_numpy().astype(float)
        return data


class PlotStyler:
    """Handles matplotlib styling and configuration."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize plot styler with configuration.

        Args:
            config: Configuration dictionary from TOML file
        """
        self.config = config

        # Extract commonly used config values with defaults
        self.dpi = config.get('figure', {}).get('dpi', 300)
        self.format = config.get('figure', {}).get('format', 'png')
        self.style = config.get('figure', {}).get('style', 'seaborn-v0_8-darkgrid')

        # Color palette
        color_palette_name = config.get('colors', {}).get('palette', 'tab10')
        self.color_palette = plt.get_cmap(color_palette_name)
        self.colors = [self.color_palette(i) for i in np.linspace(0, 1, 12)]

        # Style configuration
        style_cfg = config.get('style', {})
        fonts_cfg = style_cfg.get('fonts', {})
        axes_cfg = style_cfg.get('axes', {})

        # Store style parameters
        self.alpha = style_cfg.get('alpha', 0.8)
        self.linewidth = style_cfg.get('linewidth', 1.5)
        self.grid_linewidth = style_cfg.get('grid_linewidth', 0.5)
        self.grid_color = style_cfg.get('grid_color', 'gray')
        self.grid_linestyle = style_cfg.get('grid_linestyle', '--')

        # Font sizes
        self.font_base = fonts_cfg.get('base', 10)
        self.font_title = fonts_cfg.get('title', 8)
        self.font_axis_label = fonts_cfg.get('axis_label', 10)
        self.font_tick_label = fonts_cfg.get('tick_label', 9)
        self.font_legend = fonts_cfg.get('legend', 8)
        self.font_figure_title = fonts_cfg.get('figure_title', 12)

        # Axes parameters
        self.axes_linewidth = axes_cfg.get('linewidth', 0.8)
        self.tick_major_width = axes_cfg.get('tick_major_width', 0.8)
        self.tick_major_size = axes_cfg.get('tick_major_size', 3.5)

        # Observation styling - using hatch style for consistency
        obs_cfg = style_cfg.get('observations', {})
        self.obs_hatch_color = obs_cfg.get('hatch_color', 'k')
        self.obs_hatch_alpha = obs_cfg.get('hatch_alpha', 0.15)
        self.obs_hatch_fill = obs_cfg.get('hatch_fill', False)
        self.obs_hatch_pattern = obs_cfg.get('hatch_pattern', '///')
        self.obs_line_color = obs_cfg.get('line_color', 'k')
        self.obs_line_linestyle = obs_cfg.get('line_linestyle', '--')
        self.obs_line_alpha = obs_cfg.get('line_alpha', 0.8)
        self.obs_line_linewidth = obs_cfg.get('line_linewidth', 1.5)

    def apply_style(self):
        """Apply matplotlib style configuration."""
        plt.style.use(self.style)
        plt.rcParams.update({
            'font.family': 'sans-serif',
            'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
            'font.size': self.font_base,
            'axes.titlesize': self.font_title,
            'axes.labelsize': self.font_axis_label,
            'xtick.labelsize': self.font_tick_label,
            'ytick.labelsize': self.font_tick_label,
            'legend.fontsize': self.font_legend,
            'figure.titlesize': self.font_figure_title,
            'lines.linewidth': self.linewidth,
            'axes.linewidth': self.axes_linewidth,
            'grid.linewidth': self.grid_linewidth,
            'grid.color': self.grid_color,
            'grid.linestyle': self.grid_linestyle,
            'xtick.major.width': self.tick_major_width,
            'ytick.major.width': self.tick_major_width,
            'xtick.major.size': self.tick_major_size,
            'ytick.major.size': self.tick_major_size,
            # Explicitly keep all 4 spines visible (full frame border)
            'axes.spines.left': True,
            'axes.spines.bottom': True,
            'axes.spines.top': True,
            'axes.spines.right': True,
        })


class FigureSaver:
    """Handles saving figures with consistent settings."""

    def __init__(self, save_dir: pathlib.Path, dpi: int, format: str):
        """
        Initialize figure saver.

        Args:
            save_dir: Directory to save figures
            dpi: DPI for raster formats
            format: File format (png, svg, etc.)
        """
        self.save_dir = pathlib.Path(save_dir)
        self.dpi = dpi
        self.format = format

    def save(self, fig: plt.Figure, filename_base: str) -> pathlib.Path:
        """
        Save figure with configured settings.

        Args:
            fig: Figure to save
            filename_base: Base filename (extension will be replaced)

        Returns:
            Path to saved file
        """
        # Replace extension with configured format
        filename = pathlib.Path(filename_base).stem + f".{self.format}"
        path = self.save_dir / filename

        # For SVG, don't use DPI (it's vector-based)
        if self.format == 'svg':
            fig.savefig(path, format='svg', bbox_inches='tight', facecolor='white')
        elif self.format == 'png':
            fig.savefig(path, dpi=self.dpi, bbox_inches='tight', facecolor='white',
                        pil_kwargs={'optimize': True, 'compress_level': 9})
        else:
            fig.savefig(path, dpi=self.dpi, bbox_inches='tight', facecolor='white')

        plt.close(fig)
        return path


class AxisSetup:
    """Utilities for setting up plot axes with consistent styling."""

    @staticmethod
    def setup_axis(ax: plt.Axes, year: np.ndarray, data: np.ndarray,
                   color: str, title: str, ylabel: str,
                   obs_range: Optional[ObservationRange] = None,
                   obs_line: Optional[ObservationLine] = None,
                   year_limits: Optional[Tuple[float, float]] = None,
                   add_xlabel: bool = True,
                   styler: Optional[PlotStyler] = None):
        """
        Set up a single axis with data, styling, and observations.

        Args:
            ax: Matplotlib axis to configure
            year: Year data for x-axis
            data: Data values for y-axis
            color: Line color
            title: Plot title
            ylabel: Y-axis label
            obs_range: Optional observational range to plot (uses hatch style)
            obs_line: Optional observational line to plot
            year_limits: Optional (min, max) tuple for x-axis limits
            add_xlabel: Whether to add x-axis label
            styler: PlotStyler instance for styling parameters
        """
        # Use default values if styler not provided
        alpha = styler.alpha if styler else 0.8
        obs_hatch_color = styler.obs_hatch_color if styler else 'k'
        obs_hatch_alpha = styler.obs_hatch_alpha if styler else 0.15
        obs_hatch_fill = styler.obs_hatch_fill if styler else False
        obs_hatch_pattern = styler.obs_hatch_pattern if styler else '///'
        obs_line_color = styler.obs_line_color if styler else 'k'
        obs_line_linestyle = styler.obs_line_linestyle if styler else '--'
        obs_line_alpha = styler.obs_line_alpha if styler else 0.8
        obs_line_linewidth = styler.obs_line_linewidth if styler else 1.5

        # Plot main data
        ax.plot(year, data, color=color, alpha=alpha)
        ax.set_title(title, fontweight='bold', pad=5)
        ax.set_ylabel(ylabel)

        # Configure x-axis
        if add_xlabel:
            ax.set_xlabel('Year', fontweight='bold')
        else:
            ax.tick_params(labelbottom=False)

        ax.grid(True)

        # Calculate y-axis limits
        min_val, max_val = np.min(data), np.max(data)

        # Add observation range using hatch style
        if obs_range:
            ax.axhspan(obs_range.min_val, obs_range.max_val,
                      color=obs_hatch_color,
                      alpha=obs_hatch_alpha,
                      fill=obs_hatch_fill,
                      hatch=obs_hatch_pattern,
                      zorder=1)
            min_val = min(min_val, obs_range.min_val)
            max_val = max(max_val, obs_range.max_val)

        # Add observation line
        if obs_line:
            ax.axhline(obs_line.value,
                      color=obs_line_color,
                      linestyle=obs_line_linestyle,
                      alpha=obs_line_alpha,
                      linewidth=obs_line_linewidth, zorder=2)
            min_val = min(min_val, obs_line.value)
            max_val = max(max_val, obs_line.value)

        # Set axis limits
        buffer = (max_val - min_val) * 0.15
        ax.set_ylim(min_val - buffer, max_val + buffer)
        ax.ticklabel_format(style='scientific', axis='y', scilimits=(-2, 3), useMathText=True)

        if year_limits:
            ax.set_xlim(year_limits[0], year_limits[1])
        else:
            ax.set_xlim(year.min(), year.max())
