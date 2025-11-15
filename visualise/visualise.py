#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pathlib
import sys
from dataclasses import dataclass

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
        csv_path = f"{self.base_dir}/{self.model_name}/breakdown.{file_type}.annual.csv"
        dat_path = f"{self.base_dir}/{self.model_name}/breakdown.{file_type}.annual.dat"

        try:
            # New CSV format - single header row, comma-separated
            return pd.read_csv(csv_path)
        except FileNotFoundError:
            # Legacy TSV format - 3 header rows, tab-separated
            # Read with header=2 to skip first 2 rows (units and keys)
            return pd.read_csv(dat_path, sep="\t", header=2)

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
        volume_data["SPT"] = sum(volume_data[col] for col in ["GRAGEL", "GRACRU", "GRAMES", "GRAPRO", "GRAPTE"])
        data.update(volume_data)
        
        level_df = self._read_breakdown_file("lev")
        level_cols = ["EXP", "ExpARA", "ExpCO3", "sinksil", "EXP1000"]
        level_data = self._extract_arrays(level_df, level_cols)
        level_data["EXPCACO3"] = level_data["ExpARA"] + level_data["ExpCO3"]
        level_data["SI_FLX"] = level_data["sinksil"]
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
    def __init__(self, save_dir, model_name):
        self.save_dir = save_dir
        self.model_name = model_name
        self.ratio = 2.5
        
        plt.style.use('seaborn-v0_8-poster')
        plt.rcParams.update({
            'font.family': 'sans-serif',
            'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
            'font.size': 20,
            'axes.titlesize': 24,
            'axes.labelsize': 22,
            'xtick.labelsize': 18,
            'ytick.labelsize': 18,
            'legend.fontsize': 18,
            'figure.titlesize': 30,
            'lines.linewidth': 4.5,
            'axes.linewidth': 2,
            'grid.linewidth': 1.5,
            'grid.color': '#cccccc',
            'grid.linestyle': '--',
            'xtick.major.width': 2,
            'ytick.major.width': 2,
            'xtick.major.size': 8,
            'ytick.major.size': 8,
        })
        
        self.colors = {
            'blue': '#0077b6', 'cyan': '#00b4d8', 'red': '#d00000', 
            'orange': '#f48c06', 'purple': '#5a189a', 'green': '#1b998b',
            'pink': '#e56b6f', 'yellow': '#fde428'
        }

    def _get_global_year_limits(self, data):
        return data["year"].min(), data["year"].max()

    def _setup_axis(self, ax, year, data, color, title, obs_range=None, obs_line=None, year_limits=None, add_xlabel=True):
        ax.plot(year, data, color=color, alpha=0.9)
        ax.set_title(title, fontweight='bold', pad=20)
        
        # Conditionally add the x-axis label
        if add_xlabel:
            ax.set_xlabel('Year', fontweight='bold')
        else:
            ax.tick_params(labelbottom=False)
            
        ax.grid(True)
        
        min_val, max_val = np.min(data), np.max(data)

        if obs_range:
            ax.axhspan(obs_range.min_val, obs_range.max_val, color="#6c757d", 
                       alpha=0.1, zorder=1)
            min_val = min(min_val, obs_range.min_val)
            max_val = max(max_val, obs_range.max_val)

        if obs_line:
            ax.axhline(obs_line.value, color="#d00000", linestyle="--", 
                       alpha=0.9, linewidth=3.5, zorder=2)
            min_val = min(min_val, obs_line.value)
            max_val = max(max_val, obs_line.value)
        
        buffer = (max_val - min_val) * 0.15
        ax.set_ylim(min_val - buffer, max_val + buffer)
        ax.ticklabel_format(style='scientific', axis='y', scilimits=(-2, 3), useMathText=True)
        
        if year_limits:
            ax.set_xlim(year_limits[0], year_limits[1])
        else:
            ax.set_xlim(year.min(), year.max())

    def _create_subplot_grid(self, configs, data, year_limits, figsize, grid_shape, filename, ylabel=None):
        fig, axes = plt.subplots(*grid_shape, figsize=figsize, sharex=True) # Use sharex
        if len(grid_shape) > 1:
            axes = axes.flatten()
        elif grid_shape[0] == 1 and grid_shape[1] == 1:
            axes = [axes]
        
        for i, config in enumerate(configs):
            if i >= len(axes):
                break
            ax = axes[i]
            plot_data, color, title = config[:3]
            obs_range = config[3] if len(config) > 3 else None
            obs_line = config[4] if len(config) > 4 else None
            
            # Check if subplot is in the bottom row to add the xlabel
            rows, cols = grid_shape
            is_bottom_row = i >= (rows - 1) * cols
            
            self._setup_axis(ax, data["year"], plot_data, color, title, obs_range, obs_line, year_limits, add_xlabel=is_bottom_row)
            
            if ylabel:
                ax.set_ylabel(ylabel, fontweight='bold')
        
        if hasattr(axes, '__len__') and len(axes) > len(configs):
            for i in range(len(configs), len(axes)):
                axes[i].set_visible(False)        

        self._save_figure(fig, filename)

    def _save_figure(self, fig, filename):
        path = pathlib.Path(self.save_dir) / filename
        fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Saved: {path}")
        plt.close(fig)

    def create_global_summary(self, data):
        year_limits = self._get_global_year_limits(data)
        
        plot_configs = [
            (data["Cflx"], self.colors['blue'], "Surface Carbon Flux [PgC/yr]"),
            (data["TChl"], self.colors['green'], "Surface Chlorophyll [μg Chl/L]", None, ObservationLine(0.2921)),
            (data["PPT"], self.colors['orange'], "Primary Production [PgC/yr]", ObservationRange(51, 65)),
            (data["EXP"], self.colors['red'], "Export at 100m [PgC/yr]", ObservationRange(7.8, 12.2)),
            (data["EXP1000"], self.colors['purple'], "Export at 1000m [PgC/yr]"),
            (data["PROCACO3"], self.colors['cyan'], "CaCO₃ Production [PgC/yr]", ObservationRange(1.04, 3.34)),
            (data["EXPCACO3"], self.colors['blue'], "CaCO₃ Export at 100m [PgC/yr]", ObservationRange(0.68, 0.9)),
            (data["probsi"], self.colors['pink'], "Silica Production [Tmol/yr]", ObservationRange(203, 307)),
            (data["SI_FLX"], self.colors['green'], "Silica Export at 100m [Tmol/yr]", ObservationRange(89, 135)),
        ]

        self._create_subplot_grid(
            plot_configs, data, year_limits,
            (13.5 * self.ratio, 6.5 * self.ratio), (2, 5),
            f"{self.model_name}_summary_global.jpg"
        )

    def create_pft_summary(self, data):
        year_limits = self._get_global_year_limits(data)
        
        fig = plt.figure(figsize=(14 * self.ratio, 7.5 * self.ratio))
        gs = gridspec.GridSpec(3, 4, figure=fig)
        
        phyto_configs = [
            ("PIC", "Picophytoplankton", ObservationRange(0.28, 0.52)), 
            ("PHA", "Phaeocystis", ObservationRange(0.11, 0.69)),
            ("MIX", "Mixotrophs", None), 
            ("DIA", "Diatoms", ObservationRange(0.013, 0.75)),
            ("COC", "Coccolithophores", ObservationRange(0.001, 0.032)), 
            ("FIX", "Nitrogen Fixers", ObservationRange(0.008, 0.12))
        ]
        
        zoo_configs = [
            ("GEL", "Gelatinous Zooplankton", ObservationRange(0.10, 3.11)), 
            ("PRO", "Protozooplankton", ObservationRange(0.10, 0.37)),
            ("BAC", "Bacteria", ObservationRange(0.25, 0.26)), 
            ("CRU", "Crustaceans", ObservationRange(0.01, 0.64)),
            ("PTE", "Pteropods", ObservationRange(0.048, 0.057)), 
            ("MES", "Mesozooplankton", ObservationRange(0.21, 0.34))
        ]
        
        all_configs = phyto_configs + zoo_configs
        all_colors = [self.colors[c] for c in ['blue', 'green', 'cyan', 'orange', 'red', 'purple', 
                                              'pink', 'blue', 'orange', 'green', 'red', 'purple']]

        for i, (config, color) in enumerate(zip(all_configs, all_colors)):
            row, col = i // 4, i % 4
            ax = fig.add_subplot(gs[row, col])
            
            # Check if subplot is in the bottom row (row index 2)
            add_xlabel = (row == 2)
            
            obs_range = config[2] if len(config) > 2 else None
            self._setup_axis(ax, data["year"], data[config[0]], color, config[1], obs_range, year_limits=year_limits, add_xlabel=add_xlabel)
#        ax.set_ylabel('Biomass [PgC]', fontweight='bold')

        # This ensures all subplots share the same x-axis
        fig.align_xlabels(fig.axes)        

        self._save_figure(fig, f"{self.model_name}_summary_pfts.jpg")

    def create_nutrient_summary(self, data):
        year_limits = self._get_global_year_limits(data)
        
        nutrient_configs = [
            (data["nPO4"], self.colors['blue'], "Surface Phosphate [μmol/L]", None, ObservationLine(0.530)),
            (data["NO3"], self.colors['orange'], "Surface Nitrate [μmol/L]", None, ObservationLine(5.152)),
            (data["nFer"], self.colors['red'], "Surface Iron [nmol/L]"),
            (data["Si"], self.colors['purple'], "Surface Silica [μmol/L]", None, ObservationLine(7.485)),
            (data["O2"], self.colors['cyan'], "Surface Oxygen [μmol/L]", None, ObservationLine(251.1)),
        ]
        
        self._create_subplot_grid(
            nutrient_configs, data, year_limits,
            (14 * self.ratio, 7.5 * self.ratio), (2, 3),
            f"{self.model_name}_summary_nutrients.jpg"
        )

    def create_physics_summary(self, data):
        year_limits = self._get_global_year_limits(data)
        
        physics_configs = [
            (data["SST"], self.colors['red'], "SST [°C]"),
            (data["SSS"], self.colors['blue'], "SSS [‰]"),
            (data["MLD"], self.colors['purple'], "MLD [m]"),
        ]
        
        self._create_subplot_grid(
            physics_configs, data, year_limits,
            (6 * self.ratio, 2 * self.ratio), (1, 3),
            f"{self.model_name}_summary_physics.jpg"
        )

def main():
    if len(sys.argv) != 3:
        print("Usage: visualize.py <model_name> <model_dir>")
        sys.exit(1)
        
    model_name = sys.argv[1]
    base_dir = sys.argv[2]
    
    print(f"\n🌊 Ocean Model Visualization Tool")
    print(f"📊 Processing model: {model_name}")
    print(f"📁 Base directory: {base_dir}")
    print("="*50)
    
    save_dir = f"{base_dir}/monitor/{model_name}/"
    pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)
    print(f"✓ Output directory ready: {save_dir}")
    
    try:
        loader = ModelDataLoader(base_dir, model_name)
        data = loader.load_all_data()
        
        creator = FigureCreator(save_dir, model_name)
        
        print("\n📈 Creating visualizations...")
        creator.create_global_summary(data)
        creator.create_pft_summary(data)
        creator.create_nutrient_summary(data)
        creator.create_physics_summary(data)
        
        print("\n🎉 All visualizations completed successfully!")
        print(f"📂 Files saved to: {save_dir}")
        
    except Exception as e:
        print(f"\n❌ Error occurred: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
