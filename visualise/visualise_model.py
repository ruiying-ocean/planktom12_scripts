#!/usr/bin/env python3
"""
Complete visualization workflow for a single NEMO/PlankTom model run.
Generates maps, transects, and all diagnostic plots.

This is the main entry point for generating all visualizations for a model run.
It orchestrates preprocessing and calls individual plotting modules.

Usage:
    python visualise_model.py <run_name> <year> [OPTIONS]
"""

import argparse
import sys
from pathlib import Path

from map_utils import OceanMapPlotter, PHYTOS, ZOOS
from preprocess_data import (
    load_and_preprocess_ptrc,
    load_and_preprocess_diad,
    load_observations,
    get_nav_coordinates
)
from make_maps import (
    plot_pft_maps,
    plot_ecosystem_diagnostics,
    plot_nutrient_comparison,
    plot_carbon_chemistry
)
from make_transects import plot_basin_transects, plot_pft_transects
from difference_utils import plot_comparison_panel
from logging_utils import print_header, print_step, print_success, print_warning


def main():
    """Main entry point for complete model visualization."""
    parser = argparse.ArgumentParser(
        description='Generate complete visualization suite for NEMO/PlankTom model output'
    )
    parser.add_argument('run_name', help='Model run name (e.g., TOM12_RY_SPE2)')
    parser.add_argument('year', help='Year to process (YYYY)')
    parser.add_argument('--model-run-dir', default='~/scratch/ModelRuns',
                       help='Directory containing model runs (default: %(default)s)')
    parser.add_argument('--output-dir', default=None,
                       help='Output directory (default: <model-run-dir>/monitor/<run_name>)')
    parser.add_argument('--mask-path',
                       default='/gpfs/data/greenocean/software/resources/breakdown/basin_mask.nc',
                       help='Path to basin mask file')
    parser.add_argument('--obs-dir',
                       default='/gpfs/home/vhf24tbu/Observations',
                       help='Directory containing observational data files')
    parser.add_argument('--skip-maps', action='store_true',
                       help='Skip generating spatial maps')
    parser.add_argument('--skip-transects', action='store_true',
                       help='Skip generating transects')
    parser.add_argument('--skip-observations', action='store_true',
                       help='Skip loading and comparing with observations')
    parser.add_argument('--with-difference-maps', action='store_true',
                       help='Generate detailed model-observation difference maps')
    parser.add_argument('--max-depth', type=float, default=500.0,
                       help='Maximum depth for PFT transects in meters (default: 500)')

    args = parser.parse_args()

    # Setup paths
    model_run_dir = Path(args.model_run_dir).expanduser()
    run_dir = model_run_dir / args.run_name

    # Set default output directory
    if args.output_dir is None:
        output_dir = model_run_dir / "monitor" / args.run_name
    else:
        output_dir = Path(args.output_dir).expanduser()

    output_dir.mkdir(parents=True, exist_ok=True)

    # Construct file paths
    date_str = f"{args.year}0101_{args.year}1231"
    ptrc_file = run_dir / f"ORCA2_1m_{date_str}_ptrc_T.nc"
    diad_file = run_dir / f"ORCA2_1m_{date_str}_diad_T.nc"

    # Check files exist
    if not ptrc_file.exists():
        print(f"Error: {ptrc_file} not found")
        sys.exit(1)
    if not diad_file.exists():
        print(f"Error: {diad_file} not found")
        sys.exit(1)

    print_header("NEMO/PlankTom Model Visualization")
    print(f"  Run:         {args.run_name}")
    print(f"  Year:        {args.year}")
    print(f"  Data dir:    {run_dir}")
    print(f"  Output dir:  {output_dir}")
    print()

    # ========================================================================
    # STEP 1: Data Preprocessing (load once, use everywhere)
    # ========================================================================
    print_header("Step 1: Data Preprocessing")

    # Initialize plotter
    plotter = OceanMapPlotter(mask_path=args.mask_path)

    # Load and preprocess model datasets
    need_transects = not args.skip_transects
    ptrc_ds = load_and_preprocess_ptrc(
        ptrc_file=ptrc_file,
        plotter=plotter,
        compute_integrated=True,  # For PFT maps
        compute_concentrations=need_transects  # For transects (skip if not needed)
    )

    diad_ds = load_and_preprocess_diad(
        diad_file=diad_file,
        plotter=plotter
    )

    print_success("Data preprocessing complete\n")

    # ========================================================================
    # STEP 2: Generate Spatial Maps
    # ========================================================================
    if not args.skip_maps:
        print_header("Step 2: Generating Spatial Maps")

        # 2.1 Ecosystem diagnostics with satellite chlorophyll
        print_step(1, 5, "Ecosystem diagnostics (TChl, EXP, PPINT)")
        obs_dir = Path(args.obs_dir)
        chl_obs_file = obs_dir / 'occi_chla_monthly_climatology.nc'
        if not chl_obs_file.exists():
            chl_obs_file = obs_dir / 'modis_chla_climatology_orca.nc'
        if not chl_obs_file.exists():
            chl_obs_file = obs_dir / 'merged_chla_climatology_orca.nc'

        plot_ecosystem_diagnostics(
            plotter=plotter,
            diad_ds=diad_ds,
            sat_chl_path=chl_obs_file,
            output_path=output_dir / f"{args.run_name}_{args.year}_diagnostics.png"
        )

        # 2.2 Phytoplankton maps
        print_step(2, 5, "Phytoplankton functional types")
        plot_pft_maps(
            plotter=plotter,
            ptrc_ds=ptrc_ds,
            pft_list=PHYTOS,
            pft_type='phyto',
            output_path=output_dir / f"{args.run_name}_{args.year}_phytos.png",
            cmap='turbo'
        )

        # 2.3 Zooplankton maps
        print_step(3, 5, "Zooplankton functional types")
        plot_pft_maps(
            plotter=plotter,
            ptrc_ds=ptrc_ds,
            pft_list=ZOOS,
            pft_type='zoo',
            output_path=output_dir / f"{args.run_name}_{args.year}_zoos.png",
            cmap='turbo'
        )

        # 2.4 Nutrient maps
        print_step(4, 5, "Nutrient distributions")
        nutrients = ['_NO3', '_PO4', '_Si', '_Fer']

        if not args.skip_observations:
            # Load observations for comparison
            obs_datasets = load_observations(obs_dir, nutrients=nutrients)

            plot_nutrient_comparison(
                plotter=plotter,
                ptrc_ds=ptrc_ds,
                obs_datasets=obs_datasets,
                output_path=output_dir / f"{args.run_name}_{args.year}_nutrients.png",
                nutrients=nutrients
            )

            # 2.5 Detailed difference maps (optional)
            if args.with_difference_maps:
                print("  [Bonus] Generating detailed model-observation difference maps...")
                for nutrient in nutrients:
                    if nutrient in ptrc_ds and nutrient in obs_datasets and obs_datasets[nutrient] is not None:
                        # Extract surface data
                        model_data = ptrc_ds[nutrient]
                        if 'deptht' in model_data.dims:
                            model_data = model_data.isel(deptht=0)
                        model_data = model_data.squeeze()

                        obs_data = obs_datasets[nutrient]
                        if 'depth' in obs_data.dims:
                            obs_data = obs_data.isel(depth=0)
                        elif 'deptht' in obs_data.dims:
                            obs_data = obs_data.isel(deptht=0)
                        obs_data = obs_data.squeeze()

                        # Create comparison panel
                        plot_comparison_panel(
                            plotter=plotter,
                            data1=model_data,
                            data2=obs_data,
                            variable=nutrient,
                            label1=args.run_name,
                            label2="Observations",
                            output_path=output_dir / f"{args.run_name}_{args.year}_diff{nutrient}.png",
                            show_difference=True,
                            show_stats=True
                        )
        else:
            # Model-only nutrient plots (simplified version)
            print("        (Skipping observations - model only)")
            # Could call a simplified nutrient plotting function here if needed

        # 2.5 Carbon chemistry maps
        print_step(5, 5, "Carbon chemistry (ALK, DIC)")
        carbon_vars = ['_ALK', '_DIC']

        if not args.skip_observations:
            # Load carbon chemistry observations from GLODAP
            carbon_obs = load_observations(obs_dir, nutrients=[], carbon_chemistry=carbon_vars)

            plot_carbon_chemistry(
                plotter=plotter,
                ptrc_ds=ptrc_ds,
                obs_datasets=carbon_obs,
                output_path=output_dir / f"{args.run_name}_{args.year}_carbon_chemistry.png",
                variables=carbon_vars
            )
        else:
            # Model-only carbon chemistry plots
            print("        (Skipping observations - model only)")
            plot_carbon_chemistry(
                plotter=plotter,
                ptrc_ds=ptrc_ds,
                obs_datasets=None,
                output_path=output_dir / f"{args.run_name}_{args.year}_carbon_chemistry.png",
                variables=carbon_vars
            )

        print_success("Spatial maps complete\n")

    # ========================================================================
    # STEP 3: Generate Transects
    # ========================================================================
    if not args.skip_transects:
        print_header("Step 3: Generating Vertical Transects")

        # Get navigation coordinates
        try:
            nav_lon, nav_lat = get_nav_coordinates(ptrc_file)

            # 3.1 Nutrient transects
            if not args.skip_observations:
                print_step(1, 2, "Nutrient transects (Atlantic 35°W, Pacific 170°W)")
                nutrients = ['_NO3', '_PO4', '_Si', '_Fer']
                obs_datasets = load_observations(obs_dir, nutrients=nutrients)

                plot_basin_transects(
                    plotter=plotter,
                    ptrc_ds=ptrc_ds,
                    obs_datasets=obs_datasets,
                    nav_lon=nav_lon,
                    nav_lat=nav_lat,
                    output_dir=output_dir,
                    run_name=args.run_name,
                    year=args.year,
                    nutrients=nutrients
                )
            else:
                print("  [1/2] Nutrient transects (skipped - no observations)")

            # 3.2 PFT transects
            print_step(2, 2, "PFT transects (Atlantic 35°W, Pacific 170°W)")
            plot_pft_transects(
                plotter=plotter,
                ptrc_ds=ptrc_ds,
                nav_lon=nav_lon,
                nav_lat=nav_lat,
                output_dir=output_dir,
                run_name=args.run_name,
                year=args.year,
                pfts=PHYTOS + ZOOS,
                max_depth=args.max_depth
            )

            print_success("Transects complete\n")

        except ValueError as e:
            print_warning(f"{e}")
            print(f"  Skipping transects\n")

    # ========================================================================
    # Summary
    # ========================================================================
    print_header("Visualization Complete!")
    print(f"  Output directory: {output_dir}")
    print()
    print("Generated files:")

    # List generated files
    generated_files = sorted(output_dir.glob(f"{args.run_name}_{args.year}_*.png"))
    for i, f in enumerate(generated_files, 1):
        print(f"  [{i}] {f.name}")

    if not generated_files:
        print("  (No files generated - check options)")

    print()


if __name__ == '__main__':
    main()
