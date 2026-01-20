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
        volume_cols = ["PPT", "proara", "prococ", "probsi", "GRAGEL", "GRACRU", "GRAMES", "GRAPRO", "GRAPTE",
                       "PPT_DIA", "PPT_MIX", "PPT_COC", "PPT_PIC", "PPT_PHA", "PPT_FIX"]
        volume_data = self._extract_arrays(volume_df, volume_cols)
        volume_data["PROCACO3"] = volume_data["proara"] + volume_data["prococ"]
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
        data.update(level_data)

        average_df = self._read_analyser_file("ave")
        avg_cols = ["TChl", "PO4", "NO3", "Fer", "Si", "O2", "tos", "sos", "mldr10_1", "Alkalini"]
        avg_data = self._extract_arrays(average_df, avg_cols)
        avg_data["nFer"] = avg_data["Fer"] * 1000
        avg_data["nPO4"] = avg_data["PO4"] / 122
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

    def create_global_summary(self, data):
        year_limits = self._get_global_year_limits(data)
        layout = self.config['layout']['global_summary']
        subplot_width = self.config['layout']['subplot_width']
        subplot_height = self.config['layout']['subplot_height']

        # Use shared observation data (loaded from config)
        obs = ObservationData.get_global()

        plot_configs = [
            (data["Cflx"], self.colors[0], "Surface Carbon Flux", "PgC/yr", None, None),
            (data["TChl"], self.colors[1], "Surface Chlorophyll", "μg Chl/L", None,
             ObservationLine(obs["TChl"]["value"])),
            (data["PPT"], self.colors[2], "Primary Production", "PgC/yr",
             ObservationRange(obs["PPT"]["min"], obs["PPT"]["max"]), None),
            (data["EXP"], self.colors[3], "Export at 100m", "PgC/yr",
             ObservationRange(obs["EXP"]["min"], obs["EXP"]["max"]), None),
            (data["EXP1000"], self.colors[4], "Export at 1000m", "PgC/yr", None, None),
            (data["PROCACO3"], self.colors[5], "CaCO₃ Production", "PgC/yr",
             ObservationRange(obs["PROCACO3"]["min"], obs["PROCACO3"]["max"]), None),
            (data["EXPCACO3"], self.colors[6], "CaCO₃ Export at 100m", "PgC/yr",
             ObservationRange(obs["EXPCACO3"]["min"], obs["EXPCACO3"]["max"]), None),
            (data["probsi"], self.colors[7], "Silica Production", "Tmol/yr",
             ObservationRange(obs["probsi"]["min"], obs["probsi"]["max"]), None),
            (data["SI_FLX"], self.colors[8], "Silica Export at 100m", "Tmol/yr",
             ObservationRange(obs["SI_FLX"]["min"], obs["SI_FLX"]["max"]), None),
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

        # Use shared observation data (loaded from config)
        pft_obs = ObservationData.get_pft()

        pft_configs = [
            ("PIC", "Picophytoplankton", "PgC",
             ObservationRange(pft_obs["PIC"]["min"], pft_obs["PIC"]["max"]) if pft_obs["PIC"]["min"] else None),
            ("PHA", "Phaeocystis", "PgC",
             ObservationRange(pft_obs["PHA"]["min"], pft_obs["PHA"]["max"]) if pft_obs["PHA"]["min"] else None),
            ("MIX", "Mixotrophs", "PgC", None),
            ("DIA", "Diatoms", "PgC",
             ObservationRange(pft_obs["DIA"]["min"], pft_obs["DIA"]["max"]) if pft_obs["DIA"]["min"] else None),
            ("COC", "Coccolithophores", "PgC",
             ObservationRange(pft_obs["COC"]["min"], pft_obs["COC"]["max"]) if pft_obs["COC"]["min"] else None),
            ("FIX", "Nitrogen Fixers", "PgC",
             ObservationRange(pft_obs["FIX"]["min"], pft_obs["FIX"]["max"]) if pft_obs["FIX"]["min"] else None),
            ("GEL", "Gelatinous Zooplankton", "PgC",
             ObservationRange(pft_obs["GEL"]["min"], pft_obs["GEL"]["max"]) if pft_obs["GEL"]["min"] else None),
            ("PRO", "Protozooplankton", "PgC",
             ObservationRange(pft_obs["PRO"]["min"], pft_obs["PRO"]["max"]) if pft_obs["PRO"]["min"] else None),
            ("BAC", "Bacteria", "PgC",
             ObservationRange(pft_obs["BAC"]["min"], pft_obs["BAC"]["max"]) if pft_obs["BAC"]["min"] else None),
            ("CRU", "Crustaceans", "PgC",
             ObservationRange(pft_obs["CRU"]["min"], pft_obs["CRU"]["max"]) if pft_obs["CRU"]["min"] else None),
            ("PTE", "Pteropods", "PgC",
             ObservationRange(pft_obs["PTE"]["min"], pft_obs["PTE"]["max"]) if pft_obs["PTE"]["min"] else None),
            ("MES", "Mesozooplankton", "PgC",
             ObservationRange(pft_obs["MES"]["min"], pft_obs["MES"]["max"]) if pft_obs["MES"]["min"] else None)
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

        # Use shared observation data (loaded from config)
        nut_obs = ObservationData.get_nutrients()

        nutrient_configs = [
            ("nPO4", self.colors[0], "Surface Phosphate", "μmol/L",
             ObservationLine(nut_obs["PO4"]) if nut_obs.get("PO4") else None),
            ("NO3", self.colors[1], "Surface Nitrate", "μmol/L",
             ObservationLine(nut_obs["NO3"]) if nut_obs.get("NO3") else None),
            ("nFer", self.colors[2], "Surface Iron", "nmol/L",
             ObservationLine(nut_obs["Fer"]) if nut_obs.get("Fer") else None),
            ("Si", self.colors[3], "Surface Silica", "μmol/L",
             ObservationLine(nut_obs["Si"]) if nut_obs.get("Si") else None),
            ("O2", self.colors[4], "Oxygen at 300m", "μmol/L",
             ObservationLine(nut_obs["O2"]) if nut_obs.get("O2") else None),
            ("Alkalini", self.colors[5], "Surface Alkalinity", "μmol/L",
             ObservationLine(nut_obs["Alkalini"]) if nut_obs.get("Alkalini") else None),
            ("AOU", self.colors[6] if len(self.colors) > 6 else self.colors[0],
             "AOU at 300m", "μmol/L",
             ObservationLine(nut_obs["AOU"]) if nut_obs.get("AOU") else None),
        ]

        fig, axes = plt.subplots(
            layout['rows'], layout['cols'],
            figsize=(layout['cols'] * subplot_width, layout['rows'] * subplot_height),
            sharex=True, constrained_layout=self.config['layout']['use_constrained_layout']
        )
        axes = axes.flatten()

        for i, (var_name, color, title, ylabel, obs_line) in enumerate(nutrient_configs):
            is_bottom_row = i >= (layout['rows'] - 1) * layout['cols']
            if var_name in data and data[var_name] is not None and len(data[var_name]) > 0:
                self._setup_axis(axes[i], data["year"], data[var_name], color, title, ylabel,
                               None, obs_line, year_limits, add_xlabel=is_bottom_row)
            else:
                axes[i].text(0.5, 0.5, f'{title}\nnot available',
                           ha='center', va='center', transform=axes[i].transAxes, fontsize=9, color='gray')
                axes[i].set_title(title, fontweight='bold', pad=5)
                if is_bottom_row:
                    axes[i].set_xlabel("Year", fontweight='bold')

        for idx in range(len(nutrient_configs), len(axes)):
            axes[idx].set_visible(False)

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
        layout = {'rows': 2, 'cols': 3}
        subplot_width = self.config['layout']['subplot_width']
        subplot_height = self.config['layout']['subplot_height']

        # Build derived_configs with available data
        derived_configs = [
            ("SP", self.colors[0], "Secondary Production", "PgC/yr", None, None),
            ("recycle", self.colors[1], "Residual Production", "PgC/yr", None, None),
            ("eratio", self.colors[2], "Export Ratio (e-ratio)", "Dimensionless", None, None),
            ("Teff", self.colors[3], "Transfer Efficiency", "Dimensionless", None, None),
            ("rls", self.colors[4], "RLS", "m", None, None),
        ]

        fig, axes = plt.subplots(
            layout['rows'], layout['cols'],
            figsize=(layout['cols'] * subplot_width, layout['rows'] * subplot_height),
            sharex=True, constrained_layout=self.config['layout']['use_constrained_layout']
        )
        axes = axes.flatten()

        for i, (var_name, color, title, ylabel, obs_range, obs_line) in enumerate(derived_configs):
            is_bottom_row = i >= layout['cols']
            # Check if data is available
            if var_name in data and data[var_name] is not None and len(data[var_name]) > 0:
                self._setup_axis(axes[i], data["year"], data[var_name], color, title, ylabel,
                               obs_range, obs_line, year_limits, add_xlabel=is_bottom_row)
            else:
                # Variable not available - show placeholder
                axes[i].text(0.5, 0.5, f'{title}\nnot available',
                           ha='center', va='center', transform=axes[i].transAxes, fontsize=9, color='gray')
                axes[i].set_title(title, fontweight='bold', pad=5)
                if is_bottom_row:
                    axes[i].set_xlabel("Year", fontweight='bold')

        # Hide unused subplots
        for idx in range(len(derived_configs), len(axes)):
            axes[idx].set_visible(False)

        self._save_figure(fig, f"{self.model_name}_summary_derived.png")

    def create_organic_carbon_summary(self, data):
        """Create summary plots for organic carbon pools (POC, DOC, GOC, HOC)."""
        year_limits = self._get_global_year_limits(data)
        layout = {'rows': 2, 'cols': 2}
        subplot_width = self.config['layout']['subplot_width']
        subplot_height = self.config['layout']['subplot_height']

        oc_configs = [
            ("POC", "POC", "PgC", None),
            ("DOC", "DOC", "PgC", None),
            ("GOC", "GOC", "PgC", None),
            ("HOC", "HOC", "PgC", None),
        ]

        fig, axes = plt.subplots(
            layout['rows'], layout['cols'],
            figsize=(layout['cols'] * subplot_width, layout['rows'] * subplot_height),
            sharex=True, constrained_layout=self.config['layout']['use_constrained_layout']
        )
        axes = axes.flatten()

        for i, (var_name, title, ylabel, obs_range) in enumerate(oc_configs):
            if var_name not in data or data[var_name] is None or len(data[var_name]) == 0:
                axes[i].text(0.5, 0.5, f'{var_name} not available',
                           ha='center', va='center', transform=axes[i].transAxes)
                axes[i].set_title(title, fontweight='bold', pad=5)
                continue
            color = self.colors[i % len(self.colors)]
            is_bottom_row = i >= layout['cols']
            self._setup_axis(axes[i], data["year"], data[var_name], color, title, ylabel,
                           obs_range, None, year_limits, add_xlabel=is_bottom_row)

        self._save_figure(fig, f"{self.model_name}_summary_organic_carbon.png")

    def create_ppt_by_pft_summary(self, data):
        """Create summary plots for primary production by phytoplankton PFT."""
        year_limits = self._get_global_year_limits(data)
        layout = {'rows': 2, 'cols': 3}
        subplot_width = self.config['layout']['subplot_width']
        subplot_height = self.config['layout']['subplot_height']

        # Primary production by PFT configs (matching the phytoplankton order in PFT summary)
        ppt_configs = [
            ("PPT_PIC", "Picophytoplankton PP", "PgC/yr"),
            ("PPT_PHA", "Phaeocystis PP", "PgC/yr"),
            ("PPT_MIX", "Mixotrophs PP", "PgC/yr"),
            ("PPT_DIA", "Diatoms PP", "PgC/yr"),
            ("PPT_COC", "Coccolithophores PP", "PgC/yr"),
            ("PPT_FIX", "Nitrogen Fixers PP", "PgC/yr"),
        ]

        fig, axes = plt.subplots(
            layout['rows'], layout['cols'],
            figsize=(layout['cols'] * subplot_width, layout['rows'] * subplot_height),
            sharex=True, constrained_layout=self.config['layout']['use_constrained_layout']
        )
        axes = axes.flatten()

        for i, (var_name, title, ylabel) in enumerate(ppt_configs):
            color = self.colors[i % len(self.colors)]
            is_bottom_row = i >= layout['cols']
            # Check if data is available
            if var_name in data and data[var_name] is not None and len(data[var_name]) > 0:
                self._setup_axis(axes[i], data["year"], data[var_name], color, title, ylabel,
                               None, None, year_limits, add_xlabel=is_bottom_row)
            else:
                # Variable not available - show placeholder
                axes[i].text(0.5, 0.5, f'{title}\nnot available',
                           ha='center', va='center', transform=axes[i].transAxes, fontsize=9, color='gray')
                axes[i].set_title(title, fontweight='bold', pad=5)
                if is_bottom_row:
                    axes[i].set_xlabel("Year", fontweight='bold')

        self._save_figure(fig, f"{self.model_name}_summary_ppt_by_pft.png")

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
