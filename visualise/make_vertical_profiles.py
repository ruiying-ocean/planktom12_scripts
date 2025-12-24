#!/usr/bin/env python3

import numpy as np
import xarray as xr
from pathlib import Path
import matplotlib.pyplot as plt
import gsw
import argparse


def get_ocn_mask(path, domcfg):
    orca_basin_mask = xr.open_dataset(path)
    

    orca_basin_mask['P3N'] = orca_basin_mask['P3'].where(orca_basin_mask['P3'].Y>=0) .fillna(0)
    orca_basin_mask['P3S'] = orca_basin_mask['P3'].where(orca_basin_mask['P3'].Y<=0).fillna(0)

    orca_basin_mask['A3N'] = orca_basin_mask['A3'].where(orca_basin_mask['A3'].Y>=0).fillna(0)
    orca_basin_mask['A3S'] = orca_basin_mask['A3'].where(orca_basin_mask['A3'].Y<=0).fillna(0)


    regdict = {'ARCTIC' : 1,           

               'A1' : 2.1,
               'A2' : 2.2,
                'A3N' : 2.3,
                'A3S': 2.4,
               'A4' : 2.5,
               'A5' : 5.1,

               'P1' : 3.1,
               'P2' : 3.2,
                'P3N' : 3.3,
                'P3S' : 3.4,
               'P4' : 3.5,
               'P5' : 5.2,           

               'I3' : 4.1,
               'I4' : 4.2,
               'I5' : 5.3
              }
    regs = list(regdict.keys())

    maskno = np.zeros([149,182])*np.nan
    for i in range(0, len(regdict)):
        sub_basin_mask = orca_basin_mask[regs[i]][:] == 1
        maskno[sub_basin_mask] = regdict[regs[i]]

    maskno = xr.DataArray(maskno, dims=('y', 'x'), coords={'y': domcfg.y, 'x': domcfg.x})

    atl = (maskno >= 2.1) & (maskno <= 2.5)
    pac = (maskno >= 3.1) & (maskno <= 3.5)
    arc = maskno == 1
    so = (maskno == 5.1) | (maskno == 5.2) | (maskno == 5.3)
    ind = (maskno == 4.1) | (maskno == 4.2)
    alt_n =  (maskno >= 2.1) & (maskno <= 2.3)
    alt_s =  (maskno >= 2.4) & (maskno <= 2.5)

    pac_n = (maskno >= 3.1) & (maskno <= 3.3)
    pac_s = (maskno >= 3.4) & (maskno <= 3.5)

    glob_mask = (maskno >= 1) & (maskno <= 5.3)


    ocn_mask = xr.Dataset({'atl': atl, 'pac': pac, 'arc': arc, 'so': so, 'ind': ind, 'alt_n': alt_n,
                           'alt_s': alt_s, 'pac_n': pac_n, 'pac_s': pac_s, 'global': glob_mask})

    return ocn_mask

def get_dom_cfg(path):
    return xr.open_dataset(path)


# Default paths - will be overridden by command line args
DEFAULT_OBS_DIR = Path("/gpfs/home/vhf24tbu/Observations")
DEFAULT_MASK_DIR = Path("/gpfs/home/vhf24tbu/masks")
DEFAULT_MODEL_DIR = Path("~/scratch/ModelRuns").expanduser()

# These will be initialized in main()
woa_data = None
glodap_data = None
huang2022 = None
area_mask = None
domcfg = None
dom_confg = None
ocn_mask = None
area = None
volume = None
land_mask_2d = None
land_mask_3d = None


def load_reference_data(obs_dir, mask_dir):
    """Load observational data and masks"""
    woa_data = xr.open_dataset(obs_dir / "woa_orca_bil.nc", decode_times=False)
    glodap_data = xr.open_dataset(obs_dir / "glodap_orca_bil.nc")
    huang2022 = xr.open_dataset(obs_dir / "Huang2022_orca.nc")
    # nmol/L -> mol/L
    huang2022['fe'] = huang2022['fe'] * 1E-9

    area_mask = xr.open_dataset(mask_dir / "basin_mask.lowRes.nc")
    domcfg = xr.open_dataset(mask_dir / "meshmask.lowRes.nc")
    dom_confg = get_dom_cfg(mask_dir / "mesh_mask3_6.nc")
    ocn_mask = get_ocn_mask(mask_dir / "clq_basin_masks_ORCA.nc", dom_confg)

    land_mask_2d = domcfg['tmaskutil'].isel(t=0)
    land_mask_3d = domcfg['tmask'].isel(t=0)
    land_mask_3d = land_mask_3d.rename({'z':'deptht'})
    area = area_mask['AREA']
    volume = area_mask['VOLUME']

    return woa_data, glodap_data, huang2022, ocn_mask, area, volume, land_mask_2d, land_mask_3d


def process_and_convert_data(ptrcT_path, gridT_path, land_mask_3d):
    """
    Process and convert model data from given paths.

    Parameters:
    - ptrcT_path: Path to the ptrcT dataset (e.g., "~Downloads/ORCA2_1m_19700101_19701231_ptrc_T.nc")
    - gridT_path: Path to the gridT dataset (e.g., "~Downloads/ORCA2_1m_19700101_19701231_grid_T.nc")
    - land_mask_3d: A 3D land mask array where ocean cells are marked with 1
    
    Returns:
    A dictionary containing processed and converted variables:
    - 'alk': Alkalinity in µmol/kg
    - 'dic': Dissolved Inorganic Carbon in µmol/kg
    - 'no3': Nitrate in µmol/kg
    - 'po4': Phosphate in µmol/kg
    - 'si': Silicate in µmol/kg
    - 'o2': Oxygen in µmol/kg
    - 'temp': Temperature (°C)
    - 'sal': Salinity (PSU)
    - 'rho': Density (kg/m³)
    """
    # Load datasets
    model_ptrcT = xr.load_dataset(ptrcT_path)
    model_gridT = xr.load_dataset(gridT_path)

    # Extract relevant variables
    model_alk = model_ptrcT['Alkalini']
    model_o2 = model_ptrcT['O2']
    model_dic = model_ptrcT['DIC']
    model_no3 = model_ptrcT['NO3']
    model_po4 = model_ptrcT['PO4']
    model_si = model_ptrcT['Si']
    model_fer = model_ptrcT['Fer']
    model_temp = model_gridT['votemper']
    model_sal = model_gridT['vosaline']
    model_bac = model_ptrcT['BAC']
    model_poc = model_ptrcT['POC']
    model_goc = model_ptrcT['GOC']

    # Calculate annual mean and apply land mask
    model_alk_am = model_alk.mean(dim='time_counter').where(land_mask_3d == 1)
    model_o2_am = model_o2.mean(dim='time_counter').where(land_mask_3d == 1)
    model_dic_am = model_dic.mean(dim='time_counter').where(land_mask_3d == 1)
    model_no3_am = model_no3.mean(dim='time_counter').where(land_mask_3d == 1)
    model_po4_am = (model_po4.mean(dim='time_counter').where(land_mask_3d == 1)) / 122  # Convert to P units
    model_si_am = model_si.mean(dim='time_counter').where(land_mask_3d == 1)
    model_temp_am = model_temp.mean(dim='time_counter').where(land_mask_3d == 1)
    model_sal_am = model_sal.mean(dim='time_counter').where(land_mask_3d == 1)
    model_bac_am = model_bac.mean(dim='time_counter').where(land_mask_3d == 1)
    model_poc_am = model_poc.mean(dim='time_counter').where(land_mask_3d == 1)
    model_goc_am = model_goc.mean(dim='time_counter').where(land_mask_3d == 1)
    model_fer_am = model_fer.mean(dim='time_counter').where(land_mask_3d == 1)

    # Calculate density
    model_rho = gsw.density.sigma0(model_sal_am, model_temp_am) + 1000

    # Conversion function
    def convert_unit(data, rho):
        "convert mol/L to µmol/kg"
        data = data * 1000  # Convert mol/L to mol/m³
        data = data / rho  # Convert mol/m³ to mol/kg
        data = data * 1E6 # Convert mol/kg to µmol/kg
        return data

    # Convert and scale units
    converted_data = {
        'alk': convert_unit(model_alk_am, model_rho),
        'dic': convert_unit(model_dic_am, model_rho),
        'no3': convert_unit(model_no3_am, model_rho),
        'po4': convert_unit(model_po4_am, model_rho),
        'si': convert_unit(model_si_am, model_rho),
        'o2': convert_unit(model_o2_am, model_rho),
        'fe': model_fer_am,    # Iron (mol/L)


        'temp': model_temp_am,  # Temperature (°C)
        'sal': model_sal_am,    # Salinity (PSU)
        'rho': model_rho,        # Density (kg/m³)

        'bac': model_bac_am,
        'poc': model_poc_am,
        'goc': model_goc_am,

    }

    return converted_data

def get_model_data(dir, model_id, year, land_mask_3d):
    ptrcT_path = f"{dir}/{model_id}/ORCA2_1m_{year}0101_{year}1231_ptrc_T.nc"
    gridT_path = f"{dir}/{model_id}/ORCA2_1m_{year}0101_{year}1231_grid_T.nc"

    return process_and_convert_data(ptrcT_path, gridT_path, land_mask_3d)


# Variable display info for plot titles
VAR_INFO = {
    'no3': 'Nitrate [µmol/kg]',
    'po4': 'Phosphate [µmol/kg]',
    'si': 'Silicate [µmol/kg]',
    'fe': 'Iron [mol/L]',
    'o2': 'Oxygen [µmol/kg]',
    'alk': 'Alkalinity [µmol/kg]',
    'dic': 'DIC [µmol/kg]',
    'temp': 'Temperature [°C]',
    'sal': 'Salinity [PSU]',
}


def plot_vertical_profiles(
    model_ids,
    year,
    variables,
    model_dir,
    output_dir,
    obs_dir=None,
    mask_dir=None,
    run_name=None
):
    """
    Plot vertical profiles for multiple variables.

    Args:
        model_ids: List of model IDs to plot
        year: Year to process
        variables: List of variables to plot (e.g., ['no3', 'po4', 'si', 'o2'])
        model_dir: Path to model run directory
        output_dir: Path to output directory
        obs_dir: Path to observations directory (default: DEFAULT_OBS_DIR)
        mask_dir: Path to mask files directory (default: DEFAULT_MASK_DIR)
        run_name: Run name for output file naming (default: join model_ids)

    Returns:
        List of paths to generated figures
    """
    model_dir = Path(model_dir).expanduser()
    output_dir = Path(output_dir).expanduser()
    obs_dir = Path(obs_dir).expanduser() if obs_dir else DEFAULT_OBS_DIR
    mask_dir = Path(mask_dir).expanduser() if mask_dir else DEFAULT_MASK_DIR

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load reference data
    print("  Loading reference data for vertical profiles...")
    woa_data, glodap_data, huang2022, ocn_mask, area, volume, land_mask_2d, land_mask_3d = \
        load_reference_data(obs_dir, mask_dir)

    basins = ['atl', 'ind', 'pac', 'arc', 'so', 'global']
    output_files = []

    for var in variables:
        print(f"  Plotting vertical profile for {var}...")

        fig, axs = plt.subplots(2, 3, tight_layout=True, figsize=(6, 6), sharey=True)

        for i, basin in enumerate(basins):
            ax = axs.flatten()[i]

            # Plot reference data
            if var == 'fe':
                huang2022['fe'].where(ocn_mask[basin] == 1).weighted(area) \
                    .mean(dim=['x','y']).plot(
                        ax=ax, label='Huang2022', y='depth', color='k', linestyle='-'
                    )
            elif var not in ['alk', 'dic']:
                if var in ['temp', 'sal', 'no3', 'po4', 'o2', 'si']:
                    woa_data[var].where(ocn_mask[basin] == 1).weighted(area) \
                        .mean(dim=['x','y']).plot(
                            ax=ax, label='WOA', y='depth', color='k', linestyle='-'
                        )
            else:
                glodap_data[var].where(ocn_mask[basin] == 1).weighted(area) \
                    .mean(dim=['x','y']).plot(
                        ax=ax, label='GLODAP', y='depth_surface', color='k', linestyle='-'
                    )

            # Plot each model
            for mid in model_ids:
                data_model = get_model_data(
                    str(model_dir),
                    mid, year, land_mask_3d
                )
                data_model[var].where(ocn_mask[basin] == 1).weighted(area) \
                    .mean(dim=['x','y']).plot(
                        ax=ax, label=mid, y='deptht'
                    )

            ax.set_title(basin)
            ax.set_ylabel('Depth (m)')

        axs[0, 0].invert_yaxis()

        # Add legend
        handles, labels = axs.flatten()[0].get_legend_handles_labels()
        fig.legend(
            handles, labels,
            loc='lower center',
            ncol=len(model_ids) + 1,
            frameon=False,
            bbox_to_anchor=(0.5, -0.05)
        )
        fig.subplots_adjust(bottom=0.15)

        # Add suptitle with variable info
        var_title = VAR_INFO.get(var, var)
        fig.suptitle(var_title, fontsize=12, y=1.02)

        # Save figure
        if run_name:
            file_name = output_dir / f"{run_name}_{year}_profile_{var}.png"
        else:
            file_name = output_dir / f"{'_'.join(model_ids)}_{var}_vertical_profile.png"
        fig.savefig(file_name, bbox_inches='tight', dpi=300)
        plt.close(fig)
        output_files.append(file_name)
        print(f"    Saved: {file_name.name}")

    return output_files


def main():
    """Main function"""
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Plot vertical depth profiles for one or more model runs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single model for NO3
  %(prog)s TOM12_RY_SPE2 --year 1790 --var no3

  # Compare two models for temperature
  %(prog)s TOM12_RY_SPE2 TOM12_RY_SPE5 --year 1790 --var temp

  # Custom directories
  %(prog)s TOM12_RY_SPE2 --year 1790 --var fe --model-dir ~/custom/path --output-dir ./plots
        """
    )
    parser.add_argument(
        'model_ids',
        nargs='+',
        help="One or more model IDs (e.g. TOM12_RY_SPE2 TOM12_RY_SPE5)"
    )
    parser.add_argument(
        '--year',
        required=True,
        help="Year of the model run"
    )
    parser.add_argument(
        '--var',
        required=True,
        help="Variable to plot (e.g. temp, sal, fe, no3, po4, si, o2, alk, dic)"
    )
    parser.add_argument(
        '--model-dir',
        default=str(DEFAULT_MODEL_DIR),
        help="Base directory for model output (default: %(default)s)"
    )
    parser.add_argument(
        '--output-dir',
        default='.',
        help="Output directory for plots (default: current directory)"
    )
    parser.add_argument(
        '--obs-dir',
        default=str(DEFAULT_OBS_DIR),
        help="Directory containing observational data (default: %(default)s)"
    )
    parser.add_argument(
        '--mask-dir',
        default=str(DEFAULT_MASK_DIR),
        help="Directory containing mask files (default: %(default)s)"
    )

    args = parser.parse_args()

    # Convert to Paths
    model_dir = Path(args.model_dir).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    obs_dir = Path(args.obs_dir).expanduser()
    mask_dir = Path(args.mask_dir).expanduser()

    # Create output directory if needed
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load reference data
    print("Loading reference data...")
    woa_data, glodap_data, huang2022, ocn_mask, area, volume, land_mask_2d, land_mask_3d = \
        load_reference_data(obs_dir, mask_dir)

    # Create figure
    fig, axs = plt.subplots(2, 3, tight_layout=True, figsize=(6, 6), sharey=True)
    basins = ['atl', 'ind', 'pac', 'arc', 'so', 'global']

    # Loop over basins
    for i, basin in enumerate(basins):
        ax = axs.flatten()[i]

        # Plot reference data
        if args.var == 'fe':
            huang2022['fe'].where(ocn_mask[basin] == 1).weighted(area) \
                .mean(dim=['x','y']).plot(
                    ax=ax, label='Huang2022', y='depth', color='k', linestyle='-'
                )
        elif args.var not in ['alk','dic']:
            if args.var in ['temp','sal', 'no3','po4','o2','si']:
                woa_data[args.var].where(ocn_mask[basin] == 1).weighted(area) \
                    .mean(dim=['x','y']).plot(
                        ax=ax, label='WOA', y='depth', color='k', linestyle='-'
                    )
        else:
            glodap_data[args.var].where(ocn_mask[basin] == 1).weighted(area) \
                .mean(dim=['x','y']).plot(
                    ax=ax, label='GLODAP', y='depth_surface', color='k', linestyle='-'
                )

        # Plot each model
        for mid in args.model_ids:
            print(f"Processing {mid}...")
            data_model = get_model_data(
                str(model_dir),
                mid, args.year, land_mask_3d
            )
            data_model[args.var].where(ocn_mask[basin] == 1).weighted(area) \
                .mean(dim=['x','y']).plot(
                    ax=ax, label=mid, y='deptht'
                )

        ax.set_title(basin)
        ax.set_ylabel('Depth (m)')

    axs[0,0].invert_yaxis()

    # Add legend
    handles, labels = axs.flatten()[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc='lower center',
        ncol=len(args.model_ids) + 1,  # +1 for the reference
        frameon=False,
        bbox_to_anchor=(0.5, -0.05)
    )
    fig.subplots_adjust(bottom=0.15)

    # Save figure
    file_name = output_dir / f"{'_'.join(args.model_ids)}_{args.var}_vertical_profile.png"
    fig.savefig(file_name, bbox_inches='tight', dpi=300)
    print(f"Saved: {file_name}")


if __name__ == "__main__":
    main()
