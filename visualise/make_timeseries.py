#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pathlib
import sys
import argparse
from dataclasses import dataclass

# Import TOML parser (tomllib in Python 3.11+, tomli for earlier versions)
try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

@dataclass
class ObservationRange:
    min_val: float
    max_val: float
    label: str = "Obs. Range"

@dataclass
class ObservationLine:
    value: float
    label: str = "Observation"

class ModelDataLoader:
    def __init__(self, base_dir, model_name):
        self.base_dir = base_dir
        self.model_name = model_name
        
    def _read_breakdown_file(self, file_type):
        # Try CSV format first (new format), fall back to TSV if not found
        base_path = pathlib.Path(self.base_dir)
        csv_path = base_path / self.model_name / f"breakdown.{file_type}.annual.csv"
        dat_path = base_path / self.model_name / f"breakdown.{file_type}.annual.dat"

        try:
            # New CSV format - single header row, comma-separated
            df = pd.read_csv(csv_path)
        except FileNotFoundError:
            # Legacy TSV format - 3 header rows, tab-separated
            # Read with header=2 to skip first 2 rows (units and keys)
            df = pd.read_csv(dat_path, sep="\t", header=2)

        # Sort by year if year column exists
        if 'year' in df.columns:
            df = df.sort_values('year').reset_index(drop=True)

        return df

    def _extract_arrays(self, df, columns, skip_rows=0):
        """Extract columns as numpy arrays.

        Note: skip_rows is now 0 by default since CSV format has clean headers.
        Legacy TSV files are handled in _read_breakdown_file by using header=2.
        """
        data = {}
        for col in columns:
            if col in df.columns:
                data[col] = df[col][skip_rows:].to_numpy().astype(float)
        return data
        
    def load_all_data(self):
        data = {}
        
        surface_df = self._read_breakdown_file("sur")
        surface_data = self._extract_arrays(surface_df, ["year", "Cflx"])
        data.update(surface_data)
        
        volume_df = self._read_breakdown_file("vol")
        volume_cols = ["PPT", "proara", "prococ", "probsi", "GRAGEL", "GRACRU", "GRAMES", "GRAPRO", "GRAPTE"]
        volume_data = self._extract_arrays(volume_df, volume_cols)
        volume_data["PROCACO3"] = volume_data["proara"] + volume_data["prococ"]
        # SP (secondary production) = sum of all grazing terms
        volume_data["SP"] = sum(volume_data[col] for col in ["GRAGEL", "GRACRU", "GRAMES", "GRAPRO", "GRAPTE"])
        volume_data["SPT"] = sum(volume_data[col] for col in ["GRAGEL", "GRACRU", "GRAMES", "GRAPRO", "GRAPTE"])
        data.update(volume_data)
        
        level_df = self._read_breakdown_file("lev")
        level_cols = ["EXP", "ExpARA", "ExpCO3", "sinksil", "EXP1000"]
        level_data = self._extract_arrays(level_df, level_cols)
        level_data["EXPCACO3"] = level_data["ExpARA"] + level_data["ExpCO3"]
        level_data["SI_FLX"] = level_data["sinksil"]
        # Derived variables: Teff (transfer efficiency), e-ratio (export ratio), and recycle
        level_data["Teff"] = level_data["EXP1000"] / level_data["EXP"]  # EXP1000/EXP
        level_data["eratio"] = level_data["EXP"] / volume_data["PPT"]  # export100/NPP
        level_data["recycle"] = volume_data["PPT"] - level_data["EXP"] - volume_data["SP"]  # NPP - EXP100 - SP
        data.update(level_data)
        
        average_df = self._read_breakdown_file("ave")
        avg_cols = ["TChl", "PO4", "NO3", "Fer", "Si", "O2", "tos", "sos", "mldr10_1", "Alkalini"]
        avg_data = self._extract_arrays(average_df, avg_cols)
        avg_data["nFer"] = avg_data["Fer"] * 1000
        avg_data["nPO4"] = avg_data["PO4"] / 122
        avg_data["SST"] = avg_data["tos"]
        avg_data["SSS"] = avg_data["sos"]
        avg_data["MLD"] = avg_data["mldr10_1"]
        data.update(avg_data)
        
        int_df = self._read_breakdown_file("int")
        int_cols = ["BAC", "COC", "DIA", "FIX", "GEL", "CRU", "MES", "MIX", "PHA", "PIC", "PRO", "PTE"]
        int_data = self._extract_arrays(int_df, int_cols)
        int_data["PHY"] = sum(int_data[col] for col in ["COC", "DIA", "FIX", "MIX", "PHA", "PIC"])
        int_data["ZOO"] = sum(int_data[col] for col in ["BAC", "GEL", "CRU", "MES", "PRO", "PTE"])
        int_data["TOT"] = int_data["PHY"] + int_data["ZOO"]
        data.update(int_data)
        
        print("✓ Data loading completed successfully")
        return data

class FigureCreator:
    def __init__(self, save_dir, model_name, config_path=None):
        """
        Initialize figure creator with configurable settings.

        Args:
            save_dir: Directory to save figures
            model_name: Name of the model run
            config_path: Path to TOML configuration file (optional)
        """
        self.save_dir = save_dir
        self.model_name = model_name

        # Load configuration
        if config_path is None:
            # Default to config file in the same directory as this script
            script_dir = pathlib.Path(__file__).parent
            config_path = script_dir / 'visualise_config.toml'

        with open(config_path, 'rb') as f:
            self.config = tomllib.load(f)

        # Extract commonly used config values
        self.dpi = self.config['figure']['dpi']
        self.format = self.config['figure']['format']
        color_palette = self.config['colors']['palette']

        # Use automatic color palette
        self.color_palette = plt.get_cmap(color_palette)
        self.colors = [self.color_palette(i) for i in np.linspace(0, 1, 10)]

        # Apply matplotlib style from config
        style_cfg = self.config['style']
        fonts_cfg = style_cfg['fonts']
        axes_cfg = style_cfg['axes']

        plt.style.use(self.config['figure']['style'])
        plt.rcParams.update({
            'font.family': 'sans-serif',
            'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
            'font.size': fonts_cfg['base'],
            'axes.titlesize': fonts_cfg['title'],
            'axes.labelsize': fonts_cfg['axis_label'],
            'xtick.labelsize': fonts_cfg['tick_label'],
            'ytick.labelsize': fonts_cfg['tick_label'],
            'legend.fontsize': fonts_cfg['legend'],
            'figure.titlesize': fonts_cfg['figure_title'],
            'lines.linewidth': style_cfg['linewidth'],
            'axes.linewidth': axes_cfg['linewidth'],
            'grid.linewidth': style_cfg['grid_linewidth'],
            'grid.color': style_cfg['grid_color'],
            'grid.linestyle': style_cfg['grid_linestyle'],
            'xtick.major.width': axes_cfg['tick_major_width'],
            'ytick.major.width': axes_cfg['tick_major_width'],
            'xtick.major.size': axes_cfg['tick_major_size'],
            'ytick.major.size': axes_cfg['tick_major_size'],
        })

    def _get_global_year_limits(self, data):
        return data["year"].min(), data["year"].max()

    def _setup_axis(self, ax, year, data, color, title, ylabel, obs_range=None, obs_line=None, year_limits=None, add_xlabel=True):
        obs_cfg = self.config['style']['observations']

        ax.plot(year, data, color=color, alpha=self.config['style']['alpha'])
        ax.set_title(title, fontweight='bold', pad=5)
        ax.set_ylabel(ylabel)

        # Conditionally add the x-axis label
        if add_xlabel:
            ax.set_xlabel('Year', fontweight='bold')
        else:
            ax.tick_params(labelbottom=False)

        ax.grid(True)

        min_val, max_val = np.min(data), np.max(data)

        if obs_range:
            ax.axhspan(obs_range.min_val, obs_range.max_val,
                       color=obs_cfg['range_color'],
                       alpha=obs_cfg['range_alpha'], zorder=1)
            min_val = min(min_val, obs_range.min_val)
            max_val = max(max_val, obs_range.max_val)

        if obs_line:
            ax.axhline(obs_line.value,
                       color=obs_cfg['line_color'],
                       linestyle=obs_cfg['line_linestyle'],
                       alpha=obs_cfg['line_alpha'],
                       linewidth=obs_cfg['line_linewidth'], zorder=2)
            min_val = min(min_val, obs_line.value)
            max_val = max(max_val, obs_line.value)
        
        buffer = (max_val - min_val) * 0.15
        ax.set_ylim(min_val - buffer, max_val + buffer)
        ax.ticklabel_format(style='scientific', axis='y', scilimits=(-2, 3), useMathText=True)
        
        if year_limits:
            ax.set_xlim(year_limits[0], year_limits[1])
        else:
            ax.set_xlim(year.min(), year.max())

    def _save_figure(self, fig, filename_base):
        # Replace extension with configured format
        filename = pathlib.Path(filename_base).stem + f".{self.format}"
        path = pathlib.Path(self.save_dir) / filename

        # For SVG, don't use DPI (it's vector-based)
        if self.format == 'svg':
            fig.savefig(path, format='svg', bbox_inches='tight', facecolor='white')
        else:
            fig.savefig(path, dpi=self.dpi, bbox_inches='tight', facecolor='white')

        print(f"✓ Saved: {path}")
        plt.close(fig)

    def create_global_summary(self, data):
        year_limits = self._get_global_year_limits(data)
        layout = self.config['layout']['global_summary']
        subplot_width = self.config['layout']['subplot_width']
        subplot_height = self.config['layout']['subplot_height']

        plot_configs = [
            (data["Cflx"], self.colors[0], "Surface Carbon Flux", "PgC/yr", None, None),
            (data["TChl"], self.colors[1], "Surface Chlorophyll", "μg Chl/L", None, ObservationLine(0.2921)),
            (data["PPT"], self.colors[2], "Primary Production", "PgC/yr", ObservationRange(51, 65), None),
            (data["EXP"], self.colors[3], "Export at 100m", "PgC/yr", ObservationRange(7.8, 12.2), None),
            (data["EXP1000"], self.colors[4], "Export at 1000m", "PgC/yr", None, None),
            (data["PROCACO3"], self.colors[5], "CaCO₃ Production", "PgC/yr", ObservationRange(1.04, 3.34), None),
            (data["EXPCACO3"], self.colors[6], "CaCO₃ Export at 100m", "PgC/yr", ObservationRange(0.68, 0.9), None),
            (data["probsi"], self.colors[7], "Silica Production", "Tmol/yr", ObservationRange(203, 307), None),
            (data["SI_FLX"], self.colors[8], "Silica Export at 100m", "Tmol/yr", ObservationRange(89, 135), None),
        ]

        fig, axes = plt.subplots(
            layout['rows'], layout['cols'],
            figsize=(layout['cols'] * subplot_width, layout['rows'] * subplot_height),
            sharex=True, constrained_layout=self.config['layout']['use_constrained_layout']
        )
        axes = axes.flatten()

        for i, (plot_data, color, title, ylabel, obs_range, obs_line) in enumerate(plot_configs):
            is_bottom_row = i >= (layout['rows'] - 1) * layout['cols']
            self._setup_axis(axes[i], data["year"], plot_data, color, title, ylabel,
                           obs_range, obs_line, year_limits, add_xlabel=is_bottom_row)

        self._save_figure(fig, f"{self.model_name}_summary_global.png")

    def create_pft_summary(self, data):
        year_limits = self._get_global_year_limits(data)
        layout = self.config['layout']['pft_summary']
        subplot_width = self.config['layout']['subplot_width']
        subplot_height = self.config['layout']['subplot_height']

        pft_configs = [
            ("PIC", "Picophytoplankton", "PgC", ObservationRange(0.28, 0.52)),
            ("PHA", "Phaeocystis", "PgC", ObservationRange(0.11, 0.69)),
            ("MIX", "Mixotrophs", "PgC", None),
            ("DIA", "Diatoms", "PgC", ObservationRange(0.013, 0.75)),
            ("COC", "Coccolithophores", "PgC", ObservationRange(0.001, 0.032)),
            ("FIX", "Nitrogen Fixers", "PgC", ObservationRange(0.008, 0.12)),
            ("GEL", "Gelatinous Zooplankton", "PgC", ObservationRange(0.10, 3.11)),
            ("PRO", "Protozooplankton", "PgC", ObservationRange(0.10, 0.37)),
            ("BAC", "Bacteria", "PgC", ObservationRange(0.25, 0.26)),
            ("CRU", "Crustaceans", "PgC", ObservationRange(0.01, 0.64)),
            ("PTE", "Pteropods", "PgC", ObservationRange(0.048, 0.057)),
            ("MES", "Mesozooplankton", "PgC", ObservationRange(0.21, 0.34))
        ]

        fig, axes = plt.subplots(
            layout['rows'], layout['cols'],
            figsize=(layout['cols'] * subplot_width, layout['rows'] * subplot_height),
            sharex=True, constrained_layout=self.config['layout']['use_constrained_layout']
        )
        axes = axes.flatten()

        for i, (var_name, title, ylabel, obs_range) in enumerate(pft_configs):
            color = self.colors[i % len(self.colors)]
            is_bottom_row = i >= (layout['rows'] - 1) * layout['cols']
            self._setup_axis(axes[i], data["year"], data[var_name], color, title, ylabel,
                           obs_range, None, year_limits, add_xlabel=is_bottom_row)

        self._save_figure(fig, f"{self.model_name}_summary_pfts.png")

    def create_nutrient_summary(self, data):
        year_limits = self._get_global_year_limits(data)
        layout = self.config['layout']['nutrient_summary']
        subplot_width = self.config['layout']['subplot_width']
        subplot_height = self.config['layout']['subplot_height']

        nutrient_configs = [
            (data["nPO4"], self.colors[0], "Surface Phosphate", "μmol/L", None, ObservationLine(0.517)),
            (data["NO3"], self.colors[1], "Surface Nitrate", "μmol/L", None, ObservationLine(5.044)),
            (data["nFer"], self.colors[2], "Surface Iron", "nmol/L", None, None),
            (data["Si"], self.colors[3], "Surface Silica", "μmol/L", None, ObservationLine(7.227)),
            (data["O2"], self.colors[4], "Oxygen at 300m", "μmol/L", None, ObservationLine(168.3)),
            # Alkalinity: GLODAP value converted from μmol/kg to μmol/L (2295.11 * 1.025)
            (data["Alkalini"], self.colors[5], "Surface Alkalinity", "μmol/L", None, ObservationLine(2352.49)),
        ]

        fig, axes = plt.subplots(
            layout['rows'], layout['cols'],
            figsize=(layout['cols'] * subplot_width, layout['rows'] * subplot_height),
            sharex=True, constrained_layout=self.config['layout']['use_constrained_layout']
        )
        axes = axes.flatten()

        for i, (plot_data, color, title, ylabel, obs_range, obs_line) in enumerate(nutrient_configs):
            is_bottom_row = i >= (layout['rows'] - 1) * layout['cols']
            self._setup_axis(axes[i], data["year"], plot_data, color, title, ylabel,
                           obs_range, obs_line, year_limits, add_xlabel=is_bottom_row)

        self._save_figure(fig, f"{self.model_name}_summary_nutrients.png")

    def create_physics_summary(self, data):
        year_limits = self._get_global_year_limits(data)
        layout = self.config['layout']['physics_summary']
        subplot_width = self.config['layout']['subplot_width']
        subplot_height = self.config['layout']['subplot_height']

        physics_configs = [
            (data["SST"], self.colors[0], "SST", "°C", None, None),
            (data["SSS"], self.colors[1], "SSS", "‰", None, None),
            (data["MLD"], self.colors[2], "MLD", "m", None, None),
        ]

        fig, axes = plt.subplots(
            layout['rows'], layout['cols'],
            figsize=(layout['cols'] * subplot_width, layout['rows'] * subplot_height),
            sharex=True, constrained_layout=self.config['layout']['use_constrained_layout']
        )

        for i, (plot_data, color, title, ylabel, obs_range, obs_line) in enumerate(physics_configs):
            self._setup_axis(axes[i], data["year"], plot_data, color, title, ylabel,
                           obs_range, obs_line, year_limits, add_xlabel=True)

        self._save_figure(fig, f"{self.model_name}_summary_physics.png")

    def create_derived_summary(self, data):
        """Create summary plots for derived ecosystem variables."""
        year_limits = self._get_global_year_limits(data)
        layout = {'rows': 2, 'cols': 2}
        subplot_width = self.config['layout']['subplot_width']
        subplot_height = self.config['layout']['subplot_height']

        derived_configs = [
            (data["SP"], self.colors[0], "Secondary Production", "PgC/yr", None, None),
            (data["recycle"], self.colors[1], "Recycled Production", "PgC/yr", None, None),
            (data["eratio"], self.colors[2], "Export Ratio (e-ratio)", "Dimensionless", None, None),
            (data["Teff"], self.colors[3], "Transfer Efficiency", "Dimensionless", None, None),
        ]

        fig, axes = plt.subplots(
            layout['rows'], layout['cols'],
            figsize=(layout['cols'] * subplot_width, layout['rows'] * subplot_height),
            sharex=True, constrained_layout=self.config['layout']['use_constrained_layout']
        )
        axes = axes.flatten()

        for i, (plot_data, color, title, ylabel, obs_range, obs_line) in enumerate(derived_configs):
            is_bottom_row = i >= layout['cols']
            self._setup_axis(axes[i], data["year"], plot_data, color, title, ylabel,
                           obs_range, obs_line, year_limits, add_xlabel=is_bottom_row)

        self._save_figure(fig, f"{self.model_name}_summary_derived.png")

def main():
    parser = argparse.ArgumentParser(
        description='Create annual summary visualizations for ocean model output',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--model-id',
        required=True,
        help='Model run identifier'
    )
    parser.add_argument(
        '--model-dir',
        default='~/scratch/ModelRuns',
        help='Base directory for model outputs (default: %(default)s)'
    )

    args = parser.parse_args()

    model_name = args.model_id
    model_output_dir = pathlib.Path(args.model_dir).expanduser()

    print(f"\n🌊 Ocean Model Visualization Tool")
    print(f"📊 Processing model: {model_name}")
    print(f"📁 Model output directory: {model_output_dir}")
    print("="*50)

    save_dir = model_output_dir / "monitor" / model_name
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"✓ Output directory ready: {save_dir}")

    try:
        loader = ModelDataLoader(model_output_dir, model_name)
        data = loader.load_all_data()
        
        creator = FigureCreator(save_dir, model_name)
        
        print("\n📈 Creating visualizations...")
        creator.create_global_summary(data)
        creator.create_pft_summary(data)
        creator.create_nutrient_summary(data)
        creator.create_physics_summary(data)
        creator.create_derived_summary(data)

        print("\n🎉 All visualizations completed successfully!")
        print(f"📂 Files saved to: {save_dir}")
        
    except Exception as e:
        print(f"\n❌ Error occurred: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
