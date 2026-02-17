#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pathlib
import sys
import argparse

# Import shared utilities
from timeseries_util import (
    ObservationRange,
    ObservationLine,
    ObservationData,
    ConfigLoader,
    DataFileLoader,
    PlotStyler,
    FigureSaver,
    AxisSetup
)

class ModelDataLoader:
    def __init__(self, base_dir, model_name):
        self.base_dir = pathlib.Path(base_dir)
        self.model_name = model_name

    def _read_analyser_file(self, file_type):
        """Read analyser file using shared utility."""
        return DataFileLoader.read_analyser_file(self.base_dir, self.model_name, file_type, "annual")

    def _extract_arrays(self, df, columns, skip_rows=0):
        """Extract columns as numpy arrays using shared utility."""
        return DataFileLoader.extract_columns(df, columns, skip_rows)
        
    def load_all_data(self):
        data = {}

        surface_df = self._read_analyser_file("sur")
        surface_data = self._extract_arrays(surface_df, ["year", "Cflx"])
        data.update(surface_data)

        volume_df = self._read_analyser_file("vol")
        volume_cols = ["PPT", "PPT_Trop", "proara", "prococ", "probsi", "GRAGEL", "GRACRU", "GRAMES", "GRAPRO", "GRAPTE",
                       "PPT_DIA", "PPT_MIX", "PPT_COC", "PPT_PIC", "PPT_PHA", "PPT_FIX"]
        volume_data = self._extract_arrays(volume_df, volume_cols)
        volume_data["PROCACO3"] = volume_data["proara"] + volume_data["prococ"]
        volume_data["PPT_ExtTrop"] = volume_data["PPT"] - volume_data["PPT_Trop"]
        # SP (secondary production) = sum of all grazing terms
        volume_data["SP"] = sum(volume_data[col] for col in ["GRAGEL", "GRACRU", "GRAMES", "GRAPRO", "GRAPTE"])
        volume_data["SPT"] = sum(volume_data[col] for col in ["GRAGEL", "GRACRU", "GRAMES", "GRAPRO", "GRAPTE"])
        data.update(volume_data)

        level_df = self._read_analyser_file("lev")
        level_cols = ["EXP", "ExpARA", "ExpCO3", "sinksil", "EXP1000"]
        level_data = self._extract_arrays(level_df, level_cols)
        level_data["EXPCACO3"] = level_data["ExpARA"] + level_data["ExpCO3"]
        level_data["SI_FLX"] = level_data["sinksil"]
        # Derived variables: Teff (transfer efficiency), e-ratio (export ratio), and recycle
        level_data["Teff"] = level_data["EXP1000"] / level_data["EXP"]  # EXP1000/EXP
        level_data["eratio"] = level_data["EXP"] / volume_data["PPT"]  # export100/NPP
        level_data["recycle"] = volume_data["PPT"] - level_data["EXP"] - volume_data["SP"]  # NPP - EXP100 - SP
        level_data["spratio"] = volume_data["SP"] / volume_data["PPT"]  # SP/NPP
        data.update(level_data)

        average_df = self._read_analyser_file("ave")
        avg_cols = ["TChl", "PO4", "NO3", "Fer", "Si", "O2", "tos", "sos", "mldr10_1", "Alkalini", "DIC"]
        avg_data = self._extract_arrays(average_df, avg_cols)
        avg_data["nFer"] = avg_data["Fer"] * 1000
        avg_data["nPO4"] = avg_data["PO4"] / 122
        avg_data["ALK_DIC"] = avg_data["Alkalini"] - avg_data["DIC"]
        avg_data["SST"] = avg_data["tos"]
        avg_data["SSS"] = avg_data["sos"]
        avg_data["MLD"] = avg_data["mldr10_1"]
        # Try to load AOU if available in analyser output
        if average_df is not None and "AOU" in average_df.columns:
            aou_data = self._extract_arrays(average_df, ["AOU"])
            avg_data.update(aou_data)
        # Try to load RLS if available in analyser output
        if average_df is not None and "RLS" in average_df.columns:
            rls_data = self._extract_arrays(average_df, ["RLS"])
            avg_data["rls"] = rls_data["RLS"]
        # Try to load AMOC if available in analyser output
        if average_df is not None and "AMOC" in average_df.columns:
            amoc_data = self._extract_arrays(average_df, ["AMOC"])
            avg_data.update(amoc_data)
        data.update(avg_data)

        int_df = self._read_analyser_file("int")
        int_cols = ["BAC", "COC", "DIA", "FIX", "GEL", "CRU", "MES", "MIX", "PHA", "PIC", "PRO", "PTE"]
        int_data = self._extract_arrays(int_df, int_cols)
        int_data["PHY"] = sum(int_data[col] for col in ["COC", "DIA", "FIX", "MIX", "PHA", "PIC"])
        int_data["ZOO"] = sum(int_data[col] for col in ["BAC", "GEL", "CRU", "MES", "PRO", "PTE"])
        int_data["TOT"] = int_data["PHY"] + int_data["ZOO"]
        data.update(int_data)

        # Organic carbon pools (POC, DOC, GOC, HOC)
        oc_cols = ["POC", "DOC", "GOC", "HOC"]
        oc_data = self._extract_arrays(int_df, oc_cols)
        data.update(oc_data)
        
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
        self.save_dir = pathlib.Path(save_dir)
        self.model_name = model_name

        # Load configuration using shared utility
        self.config = ConfigLoader.load_config(config_path)

        # Initialize styler and apply style
        self.styler = PlotStyler(self.config)
        self.styler.apply_style()

        # Store commonly used values
        self.colors = self.styler.colors

        # Initialize figure saver
        self.saver = FigureSaver(self.save_dir, self.styler.dpi, self.styler.format)

    def _get_global_year_limits(self, data):
        return data["year"].min(), data["year"].max()

    def _setup_axis(self, ax, year, data, color, title, ylabel, obs_range=None, obs_line=None, year_limits=None, add_xlabel=True):
        """Set up axis using shared utility."""
        AxisSetup.setup_axis(ax, year, data, color, title, ylabel,
                           obs_range, obs_line, year_limits, add_xlabel, self.styler)

    def _save_figure(self, fig, filename_base):
        """Save figure using shared utility."""
        path = self.saver.save(fig, filename_base)
        print(f"✓ Saved: {path}")

    def _create_summary_from_config(self, data, plot_info, obs_source, layout_key, filename_suffix):
        """Config-driven summary plot: reads plot_info dict, looks up observations, creates subplots.

        Args:
            data: Dict of variable name -> numpy array
            plot_info: Dict from config, e.g. {var_name: {title, color_index, obs_key?}, ...}
            obs_source: Observation dict (from ObservationData), or None if no observations
            layout_key: Key into self.config['layout'], e.g. 'global_summary'
            filename_suffix: Suffix for output filename, e.g. 'summary_global'
        """
        year_limits = self._get_global_year_limits(data)
        layout = self.config['layout'][layout_key]
        subplot_width = self.config['layout']['subplot_width']
        subplot_height = self.config['layout']['subplot_height']

        plot_configs = []
        for var_name, var_info in plot_info.items():
            title = var_info['title']
            unit = var_info.get('unit', '')
            color = self.colors[var_info['color_index'] % len(self.colors)]
            obs_range, obs_line = None, None
            if obs_source is not None:
                obs_key = var_info.get('obs_key', var_name)
                if obs_key in obs_source:
                    o = obs_source[obs_key]
                    if isinstance(o, dict):
                        if o.get("type") in ("span", "range"):
                            obs_range = ObservationRange(o["min"], o["max"])
                        elif o.get("type") == "line":
                            obs_line = ObservationLine(o["value"])
                    elif isinstance(o, (int, float)):
                        obs_line = ObservationLine(o)
            plot_configs.append((var_name, color, title, unit, obs_range, obs_line))

        fig, axes = plt.subplots(
            layout['rows'], layout['cols'],
            figsize=(layout['cols'] * subplot_width, layout['rows'] * subplot_height),
            sharex=True, squeeze=False,
            constrained_layout=self.config['layout']['use_constrained_layout']
        )
        axes = axes.flatten()

        for i, (var_name, color, title, unit, obs_range, obs_line) in enumerate(plot_configs):
            is_bottom_row = i >= (layout['rows'] - 1) * layout['cols']
            if var_name in data and data[var_name] is not None and len(data[var_name]) > 0 and not np.all(data[var_name] == -1):
                self._setup_axis(axes[i], data["year"], data[var_name], color, title, unit,
                               obs_range, obs_line, year_limits, add_xlabel=is_bottom_row)
            else:
                axes[i].text(0.5, 0.5, f'{title}\nnot available',
                           ha='center', va='center', transform=axes[i].transAxes, fontsize=9, color='gray')
                axes[i].set_title(title, fontweight='bold', pad=5)
                if is_bottom_row:
                    axes[i].set_xlabel("Year", fontweight='bold')

        for idx in range(len(plot_configs), len(axes)):
            axes[idx].set_visible(False)

        self._save_figure(fig, f"{self.model_name}_{filename_suffix}.png")

    def create_global_summary(self, data):
        self._create_summary_from_config(
            data, self.config['plot_info']['ecosystem'],
            ObservationData.get_global(), 'global_summary', 'summary_global')

    def create_pft_summary(self, data):
        plot_info = dict(self.config['plot_info']['pfts']['phytoplankton'])
        plot_info.update(self.config['plot_info']['pfts']['zooplankton'])
        self._create_summary_from_config(
            data, plot_info,
            ObservationData.get_pft(), 'pft_summary', 'summary_pfts')

    def create_nutrient_summary(self, data):
        self._create_summary_from_config(
            data, self.config['plot_info']['nutrients'],
            ObservationData.get_nutrients(), 'nutrient_summary', 'summary_nutrients')

    def create_physics_summary(self, data):
        self._create_summary_from_config(
            data, self.config['plot_info']['physics'],
            ObservationData.get_physics(), 'physics_summary', 'summary_physics')

    def create_derived_summary(self, data):
        self._create_summary_from_config(
            data, self.config['plot_info']['derived'],
            ObservationData.get_derived(), 'derived_summary', 'summary_derived')

    def create_organic_carbon_summary(self, data):
        self._create_summary_from_config(
            data, self.config['plot_info']['organic_carbon'],
            None, 'organic_carbon_summary', 'summary_organic_carbon')

    def create_ppt_by_pft_summary(self, data):
        self._create_summary_from_config(
            data, self.config['plot_info']['ppt_by_pft'],
            None, 'ppt_by_pft_summary', 'summary_ppt_by_pft')

def main():
    parser = argparse.ArgumentParser(
        description='Create annual summary visualizations for ocean model output',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        'model_id',
        help='Model run identifier (e.g., TOM12_RY_SPE2)'
    )
    parser.add_argument(
        '--model-run-dir',
        default='~/scratch/ModelRuns',
        help='Base directory for model outputs (default: %(default)s)'
    )

    args = parser.parse_args()

    model_name = args.model_id
    model_output_dir = pathlib.Path(args.model_run_dir).expanduser()

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
        creator.create_organic_carbon_summary(data)
        creator.create_ppt_by_pft_summary(data)

        print("\n🎉 All visualizations completed successfully!")
        print(f"📂 Files saved to: {save_dir}")
        
    except Exception as e:
        print(f"\n❌ Error occurred: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
