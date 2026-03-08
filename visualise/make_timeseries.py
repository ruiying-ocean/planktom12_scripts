#!/usr/bin/env python3

import calendar
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pathlib
import sys
import argparse
import xarray as xr

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

    @staticmethod
    def _compute_relative_change(values, baseline_years=10):
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
        
    def load_all_data(self):
        data = {}

        surface_df = self._read_analyser_file("sur")
        surface_data = self._extract_arrays(surface_df, ["year", "Cflx"])
        data.update(surface_data)

        volume_df = self._read_analyser_file("vol")
        volume_cols = ["PPT", "PPT_Trop", "proara", "prococ", "probsi", "GRAGEL", "GRACRU", "GRAMES", "GRAPRO", "GRAPTE",
                       "PPT_DIA", "PPT_MIX", "PPT_COC", "PPT_PIC", "PPT_PHA", "PPT_FIX"]
        volume_data = self._extract_arrays(volume_df, volume_cols)
        if "proara" in volume_data and "prococ" in volume_data:
            volume_data["PROCACO3"] = volume_data["proara"] + volume_data["prococ"]
        if "PPT" in volume_data and "PPT_Trop" in volume_data:
            volume_data["PPT_ExtTrop"] = volume_data["PPT"] - volume_data["PPT_Trop"]
        # SP (secondary production) = sum of all grazing terms
        grazing_cols = ["GRAGEL", "GRACRU", "GRAMES", "GRAPRO", "GRAPTE"]
        if all(col in volume_data for col in grazing_cols):
            volume_data["SP"] = sum(volume_data[col] for col in grazing_cols)
            volume_data["SPT"] = volume_data["SP"]
        data.update(volume_data)

        level_df = self._read_analyser_file("lev")
        level_cols = ["EXP", "ExpARA", "ExpCO3", "sinksil", "EXP1000"]
        level_data = self._extract_arrays(level_df, level_cols)
        if "ExpARA" in level_data and "ExpCO3" in level_data:
            level_data["EXPCACO3"] = level_data["ExpARA"] + level_data["ExpCO3"]
        if "sinksil" in level_data:
            level_data["SI_FLX"] = level_data["sinksil"]
        # Derived variables: Teff (transfer efficiency), e-ratio (export ratio), and recycle
        if "EXP1000" in level_data and "EXP" in level_data:
            level_data["Teff"] = level_data["EXP1000"] / level_data["EXP"]  # EXP1000/EXP
        if "EXP" in level_data and "PPT" in volume_data:
            level_data["eratio"] = level_data["EXP"] / volume_data["PPT"]  # export100/NPP
        if "PPT" in volume_data and "EXP" in level_data and "SP" in volume_data:
            level_data["recycle"] = volume_data["PPT"] - level_data["EXP"] - volume_data["SP"]  # NPP - EXP100 - SP
        if "SP" in volume_data and "PPT" in volume_data:
            level_data["spratio"] = volume_data["SP"] / volume_data["PPT"]  # SP/NPP
        # Trophic coupling indices following Xue et al. (2022)
        # Ingestion ratio at each trophic level relative to NPP
        # TL2: microzooplankton (PRO)
        # TL3: middle predators (MES + PTE)
        # FCE: top predators (GEL + CRU) / NPP — Food Chain Efficiency
        if "GRAPRO" in volume_data and "PPT" in volume_data:
            level_data["ratio_TL2"] = volume_data["GRAPRO"] / volume_data["PPT"]
        if all(c in volume_data for c in ["GRAMES", "GRAPTE"]) and "PPT" in volume_data:
            level_data["ratio_TL3"] = (volume_data["GRAMES"] + volume_data["GRAPTE"]) / volume_data["PPT"]
        if all(c in volume_data for c in ["GRAGEL", "GRACRU"]) and "PPT" in volume_data:
            level_data["FCE"] = (volume_data["GRAGEL"] + volume_data["GRACRU"]) / volume_data["PPT"]
        data.update(level_data)

        average_df = self._read_analyser_file("ave")
        avg_cols = [
            "TChl", "PO4", "NO3", "Fer", "Si", "O2",
            "tos", "sos", "mldr10_1", "pCO2", "Alkalini", "DIC",
            "AOU", "RLS", "AMOC",
            "bPO4", "bNO3", "bFer", "bSi", "bO2", "bAlkalini", "bDIC", "bDOC",
        ]
        avg_data = self._extract_arrays(average_df, avg_cols)
        # Derived variables: (output_key, [dependencies], transform)
        # Skipped automatically when dependencies are missing from the CSV
        derived = [
            ("nFer",    ["Fer"],              lambda d: d["Fer"] * 1000),
            ("nPO4",    ["PO4"],              lambda d: d["PO4"] / 122),
            ("ALK_DIC", ["Alkalini", "DIC"],  lambda d: d["Alkalini"] - d["DIC"]),
            ("SST",     ["tos"],              lambda d: d["tos"]),
            ("SSS",     ["sos"],              lambda d: d["sos"]),
            ("MLD",     ["mldr10_1"],         lambda d: d["mldr10_1"]),
            ("rls",     ["RLS"],              lambda d: d["RLS"]),
            ("bPO4",    ["bPO4"],             lambda d: d["bPO4"] / 122),
            ("bFer",    ["bFer"],             lambda d: d["bFer"] * 1000),
        ]
        for key, deps, func in derived:
            if all(d in avg_data for d in deps):
                avg_data[key] = func(avg_data)
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
    TCHL_REGION_DEFS = [
        {
            "title": "N. Pacific gyres",
            "lat_range": (15.0, 35.0),
            "lon_range": (140.0, -120.0),
        },
        {
            "title": "N. subpolar Pacific",
            "lat_range": (40.0, 60.0),
            "lon_range": (140.0, -120.0),
        },
        {
            "title": "North Atlantic",
            "lat_range": (40.0, 65.0),
            "lon_range": (-80.0, 0.0),
        },
        {
            "title": "Equatorial Pacific",
            "lat_range": (-5.0, 5.0),
            "lon_range": (160.0, -90.0),
        },
        {
            "title": "Subantarctic",
            "lat_range": (-55.0, -40.0),
            "lon_range": None,
        },
    ]

    def __init__(self, save_dir, model_name, model_output_dir, config_path=None):
        """
        Initialize figure creator with configurable settings.

        Args:
            save_dir: Directory to save figures
            model_name: Name of the model run
            config_path: Path to TOML configuration file (optional)
        """
        self.save_dir = pathlib.Path(save_dir)
        self.model_name = model_name
        self.model_output_dir = pathlib.Path(model_output_dir)

        # Load configuration using shared utility
        self.config = ConfigLoader.load_config(config_path)

        # Initialize styler and apply style
        self.styler = PlotStyler(self.config)
        self.styler.apply_style()

        # Store commonly used values
        self.colors = self.styler.colors

        # Initialize figure saver
        self.saver = FigureSaver(self.save_dir, self.styler.dpi, self.styler.format)

    @staticmethod
    def _build_lon_mask(lon, lon_range):
        if lon_range is None:
            return xr.ones_like(lon, dtype=bool)

        lon_min, lon_max = lon_range
        lon_0360 = xr.where(lon < 0, lon + 360, lon)
        min_0360 = lon_min if lon_min >= 0 else lon_min + 360
        max_0360 = lon_max if lon_max >= 0 else lon_max + 360

        if min_0360 <= max_0360:
            return (lon_0360 >= min_0360) & (lon_0360 <= max_0360)
        return (lon_0360 >= min_0360) | (lon_0360 <= max_0360)

    def _load_tchl_seasonal_regions(self):
        run_dir = self.model_output_dir / self.model_name
        diad_files = sorted(run_dir.glob("ORCA2_1m_*_diad_T.nc"))
        if not diad_files:
            print(f"  ⚠ No ORCA2_1m_*_diad_T.nc files found in {run_dir} — skipping tchl seasonal plot")
            return None, None

        latest_file = max(
            diad_files,
            key=lambda p: p.name.split("_")[2] if len(p.name.split("_")) > 2 else ""
        )
        nc_file = latest_file
        if not nc_file.exists():
            print(f"  ⚠ File not found: {nc_file} — skipping tchl seasonal plot")
            return None, None

        try:
            with xr.open_dataset(nc_file, decode_times=False) as ds:
                if "TChl" not in ds:
                    print(f"  ⚠ 'TChl' variable not found in {nc_file.name} — skipping tchl seasonal plot")
                    return None, None

                tchl = ds["TChl"]
                depth_dim = next((d for d in ["deptht", "nav_lev", "z"] if d in tchl.dims), None)
                if depth_dim is not None:
                    tchl = tchl.isel({depth_dim: 0})

                lat_name = "nav_lat" if "nav_lat" in ds else "lat"
                lon_name = "nav_lon" if "nav_lon" in ds else "lon"
                if lat_name not in ds or lon_name not in ds:
                    return None, None

                lat = ds[lat_name]
                lon = ds[lon_name]
                spatial_dims = lat.dims
                weights = xr.where(np.isfinite(lat), np.cos(np.deg2rad(lat)), 0.0)

                region_series = []
                for region in self.TCHL_REGION_DEFS:
                    lat_min, lat_max = region["lat_range"]
                    lat_mask = (lat >= lat_min) & (lat <= lat_max)
                    lon_mask = self._build_lon_mask(lon, region["lon_range"])
                    region_mask = lat_mask & lon_mask & np.isfinite(tchl.isel(time_counter=0))
                    weighted_mean = tchl.where(region_mask).weighted(weights.where(region_mask)).mean(dim=spatial_dims)
                    region_series.append(weighted_mean.to_numpy().astype(float))

                month_names = list(calendar.month_abbr)[1:len(region_series[0]) + 1]
                return month_names, region_series
        except Exception as e:
            print(f"  ⚠ Could not load tchl seasonal data: {e} — skipping tchl seasonal plot")
            return None, None

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
                year = data["year"]
                y = data[var_name]
                n = min(len(year), len(y))
                self._setup_axis(axes[i], year[:n], y[:n], color, title, unit,
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
            ObservationData.get_global(), 'global_summary', 'ts_global')

    def create_pft_summary(self, data):
        plot_info = dict(self.config['plot_info']['pfts']['phytoplankton'])
        plot_info.update(self.config['plot_info']['pfts']['zooplankton'])
        self._create_summary_from_config(
            data, plot_info,
            ObservationData.get_pft(), 'pft_summary', 'ts_pfts')

    def create_nutrient_summary(self, data):
        self._create_summary_from_config(
            data, self.config['plot_info']['nutrients'],
            ObservationData.get_nutrients(), 'nutrient_summary', 'ts_nutrients')

    def create_physics_summary(self, data):
        self._create_summary_from_config(
            data, self.config['plot_info']['physics'],
            ObservationData.get_physics(), 'physics_summary', 'ts_physics')

    def create_derived_summary(self, data):
        self._create_summary_from_config(
            data, self.config['plot_info']['derived'],
            ObservationData.get_derived(), 'derived_summary', 'ts_derived')

    def create_benthic_summary(self, data):
        self._create_summary_from_config(
            data, self.config['plot_info']['benthic'],
            ObservationData.get_benthic(), 'benthic_summary', 'ts_benthic')

    def create_organic_carbon_summary(self, data):
        self._create_summary_from_config(
            data, self.config['plot_info']['organic_carbon'],
            ObservationData.get_organic_carbon(), 'organic_carbon_summary', 'ts_organic_carbon')

    def create_ppt_by_pft_summary(self, data):
        self._create_summary_from_config(
            data, self.config['plot_info']['ppt_by_pft'],
            None, 'ppt_by_pft_summary', 'ts_ppt_by_pft')

    def create_trophic_summary(self, data):
        layout = self.config['layout']['trophic_summary']
        subplot_width = self.config['layout']['subplot_width']
        subplot_height = self.config['layout']['subplot_height']
        plot_info = self.config['plot_info']['trophic']

        fig, axes = plt.subplots(
            layout['rows'], layout['cols'],
            figsize=(layout['cols'] * subplot_width, layout['rows'] * subplot_height),
            squeeze=False,
            constrained_layout=self.config['layout']['use_constrained_layout']
        )
        axes = axes.flatten()

        year_limits = self._get_global_year_limits(data)
        line_vars = ["SP", "recycle", "spratio", "ratio_TL2", "ratio_TL3", "FCE"]

        for i, var_name in enumerate(line_vars):
            var_info = plot_info[var_name]
            color = self.colors[var_info['color_index'] % len(self.colors)]
            title = var_info['title']
            unit = var_info.get('unit', '')
            is_bottom_row = i >= (layout['rows'] - 1) * layout['cols']

            if var_name in data and data[var_name] is not None and len(data[var_name]) > 0:
                year = data["year"]
                values = data[var_name]
                n = min(len(year), len(values))
                self._setup_axis(
                    axes[i], year[:n], values[:n], color, title, unit,
                    None, None, year_limits, add_xlabel=is_bottom_row
                )
            else:
                axes[i].text(0.5, 0.5, f'{title}\nnot available',
                             ha='center', va='center', transform=axes[i].transAxes,
                             fontsize=9, color='gray')
                axes[i].set_title(title, fontweight='bold', pad=5)
                if is_bottom_row:
                    axes[i].set_xlabel("Year", fontweight='bold')

        ta_ax = axes[6]
        ta_info = plot_info['TA']
        ta_color = self.colors[ta_info['color_index'] % len(self.colors)]
        sp = data.get("SP")
        ppt = data.get("PPT")
        if sp is not None and ppt is not None:
            x = ModelDataLoader._compute_relative_change(ppt)
            y = ModelDataLoader._compute_relative_change(sp)
            if x is not None and y is not None:
                min_len = min(len(x), len(y))
                x = x[:min_len]
                y = y[:min_len]
                valid = np.isfinite(x) & np.isfinite(y)
                if np.count_nonzero(valid) >= 2:
                    x_valid = x[valid]
                    y_valid = y[valid]
                    slope, intercept = np.polyfit(x_valid, y_valid, 1)

                    ta_ax.scatter(x_valid, y_valid, color=ta_color, s=18, alpha=0.8)

                    x_line = np.linspace(np.nanmin(x_valid), np.nanmax(x_valid), 100)
                    ta_ax.plot(x_line, slope * x_line + intercept, color=ta_color, linewidth=1.2)
                    ta_ax.axhline(0, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
                    ta_ax.axvline(0, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
                    ta_ax.set_title(ta_info['title'], fontweight='bold', pad=5)
                    ta_ax.set_xlabel("Relative change in NPP", fontweight='bold')
                    ta_ax.set_ylabel("Relative change in SP")
                    ta_ax.text(
                        0.05, 0.95, f"slope = {slope:.2f}",
                        transform=ta_ax.transAxes, va='top', ha='left', fontsize=8,
                        bbox=dict(facecolor='white', edgecolor='none', alpha=0.7)
                    )
                else:
                    ta_ax.text(0.5, 0.5, 'Trophic Amplification\nnot available',
                               ha='center', va='center', transform=ta_ax.transAxes,
                               fontsize=9, color='gray')
            else:
                ta_ax.text(0.5, 0.5, 'Trophic Amplification\nnot available',
                           ha='center', va='center', transform=ta_ax.transAxes,
                           fontsize=9, color='gray')
        else:
            ta_ax.text(0.5, 0.5, 'Trophic Amplification\nnot available',
                       ha='center', va='center', transform=ta_ax.transAxes,
                       fontsize=9, color='gray')

        ta_ax.grid(True)

        ba_ax = axes[7]
        ba_info = plot_info['BA']
        ba_color = self.colors[ba_info['color_index'] % len(self.colors)]
        zoo = data.get("ZOO")
        phy = data.get("PHY")
        if zoo is not None and phy is not None:
            x = ModelDataLoader._compute_relative_change(phy)
            y = ModelDataLoader._compute_relative_change(zoo)
            if x is not None and y is not None:
                min_len = min(len(x), len(y))
                x = x[:min_len]
                y = y[:min_len]
                valid = np.isfinite(x) & np.isfinite(y)
                if np.count_nonzero(valid) >= 2:
                    x_valid = x[valid]
                    y_valid = y[valid]
                    slope, intercept = np.polyfit(x_valid, y_valid, 1)

                    ba_ax.scatter(x_valid, y_valid, color=ba_color, s=18, alpha=0.8)
                    x_line = np.linspace(np.nanmin(x_valid), np.nanmax(x_valid), 100)
                    ba_ax.plot(x_line, slope * x_line + intercept, color=ba_color, linewidth=1.2)
                    ba_ax.axhline(0, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
                    ba_ax.axvline(0, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
                    ba_ax.set_title(ba_info['title'], fontweight='bold', pad=5)
                    ba_ax.set_xlabel("Relative change in PHY", fontweight='bold')
                    ba_ax.set_ylabel("Relative change in ZOO")
                    ba_ax.text(
                        0.05, 0.95, f"slope = {slope:.2f}",
                        transform=ba_ax.transAxes, va='top', ha='left', fontsize=8,
                        bbox=dict(facecolor='white', edgecolor='none', alpha=0.7)
                    )
                else:
                    ba_ax.text(0.5, 0.5, 'Biomass Amplification\nnot available',
                               ha='center', va='center', transform=ba_ax.transAxes,
                               fontsize=9, color='gray')
            else:
                ba_ax.text(0.5, 0.5, 'Biomass Amplification\nnot available',
                           ha='center', va='center', transform=ba_ax.transAxes,
                           fontsize=9, color='gray')
        else:
            ba_ax.text(0.5, 0.5, 'Biomass Amplification\nnot available',
                       ha='center', va='center', transform=ba_ax.transAxes,
                       fontsize=9, color='gray')

        ba_ax.grid(True)

        for idx in range(8, len(axes)):
            axes[idx].set_visible(False)

        self._save_figure(fig, f"{self.model_name}_ts_trophic.png")

    def create_tchl_seasonal_regions(self):
        month_names, region_series = self._load_tchl_seasonal_regions()
        if month_names is None or region_series is None:
            return

        fig, axes = plt.subplots(
            3, 2,
            figsize=(2 * self.config['layout']['subplot_width'], 3 * self.config['layout']['subplot_height']),
            squeeze=False,
            constrained_layout=self.config['layout']['use_constrained_layout']
        )
        axes = axes.flatten()

        for ax, region, series in zip(axes, self.TCHL_REGION_DEFS, region_series):
            ax.plot(month_names, series, color=self.colors[1], linewidth=self.styler.linewidth)
            ax.set_title(region["title"], fontweight='bold', pad=5)
            ax.set_ylabel("μg Chl/L")
            ax.set_xlabel("Month", fontweight='bold')
            ax.grid(True)

        for ax in axes[len(self.TCHL_REGION_DEFS):]:
            ax.set_visible(False)

        self._save_figure(fig, f"{self.model_name}_ts_tchl_seasonal.png")

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
        
        creator = FigureCreator(save_dir, model_name, model_output_dir)
        
        print("\n📈 Creating visualizations...")
        creator.create_global_summary(data)
        creator.create_tchl_seasonal_regions()
        creator.create_pft_summary(data)
        creator.create_nutrient_summary(data)
        creator.create_physics_summary(data)
        creator.create_derived_summary(data)
        creator.create_trophic_summary(data)
        creator.create_benthic_summary(data)
        creator.create_organic_carbon_summary(data)
        creator.create_ppt_by_pft_summary(data)

        print("\n🎉 All visualizations completed successfully!")
        print(f"📂 Files saved to: {save_dir}")
        
    except Exception as e:
        print(f"\n❌ Error occurred: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
