#!/usr/bin/env python3
"""
Complete visualization workflow for multi-model NEMO/PlankTom comparisons.
Generates comparison plots, transects, and HTML reports for multiple model runs.

This is the main entry point for multi-model visualizations.
It orchestrates preprocessing and calls individual comparison modules.

Usage:
    python visualise_multimodel.py <models_csv> [OPTIONS]
"""

import argparse
import sys
import csv
from pathlib import Path

# Import configuration
try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

# Import from parent visualise directory
sys.path.insert(0, str(Path(__file__).parent.parent))
from map_utils import OceanMapPlotter
from logging_utils import print_header, print_step, print_success, print_warning

# Import multimodel-specific modules
from preprocess_multimodel import get_nav_coordinates


def load_config():
    """Load visualise_config.toml"""
    script_dir = Path(__file__).parent
    config_path = script_dir.parent / "visualise_config.toml"

    if config_path.exists():
        with open(config_path, "rb") as f:
            return tomllib.load(f)
    return None


def parse_models_csv(csv_path):
    """
    Parse modelsToPlot.csv to get model configurations.

    Args:
        csv_path: Path to modelsToPlot.csv

    Returns:
        List of dicts with model information
    """
    models = []

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Get model directory (use location if specified, otherwise default)
            model_dir = row.get('location', '').strip()
            if not model_dir:
                model_dir = str(Path.home() / 'scratch' / 'ModelRuns')

            models.append({
                'name': row['model_id'].strip(),
                'description': row['description'].strip(),
                'start_year': int(row['start_year']),
                'to_year': int(row['to_year']),
                'year': int(row['to_year']),  # Use final year for visualization
                'model_dir': model_dir
            })

    return models


def create_output_dir(models, base_dir=None):
    """
    Create output directory based on model names.

    Args:
        models: List of model dicts
        base_dir: Base directory (default: current directory)

    Returns:
        Path to output directory
    """
    # Create folder name from model runs (e.g., JRA3-JRA1)
    # Strip TOM12_RY_ prefix from each run name
    clean_names = [m['name'].replace('TOM12_RY_', '') for m in models]
    folder_name = '-'.join(clean_names)

    if base_dir is None:
        base_dir = Path.cwd()
    else:
        base_dir = Path(base_dir)

    output_dir = base_dir / folder_name
    output_dir.mkdir(parents=True, exist_ok=True)

    return output_dir


def main():
    """Main entry point for multi-model visualization."""
    parser = argparse.ArgumentParser(
        description='Generate complete multi-model comparison visualization suite'
    )
    parser.add_argument('models_csv', help='Path to modelsToPlot.csv')
    parser.add_argument('--output-dir', default=None,
                       help='Output directory (default: ./<model1>-<model2>)')
    parser.add_argument('--mask-path',
                       default='/gpfs/data/greenocean/software/resources/breakdown/basin_mask.nc',
                       help='Path to basin mask file')
    parser.add_argument('--skip-timeseries', action='store_true',
                       help='Skip generating timeseries plots')
    parser.add_argument('--skip-maps', action='store_true',
                       help='Skip generating spatial comparison maps')
    parser.add_argument('--skip-transects', action='store_true',
                       help='Skip generating transects')
    parser.add_argument('--skip-config-comparison', action='store_true',
                       help='Skip configuration comparison (for 2-model comparisons)')
    parser.add_argument('--skip-html', action='store_true',
                       help='Skip HTML report generation')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug output')

    args = parser.parse_args()

    # Check if models CSV exists
    csv_path = Path(args.models_csv)
    if not csv_path.exists():
        print(f"Error: {csv_path} not found")
        sys.exit(1)

    # Parse models
    print_header("Multi-Model Comparison Visualization")

    models = parse_models_csv(csv_path)
    n_models = len(models)

    if n_models < 2:
        print("Error: Need at least 2 models for comparison")
        sys.exit(1)

    if n_models > 8:
        print("Error: Maximum 8 models supported")
        sys.exit(1)

    print(f"  Comparing {n_models} models:")
    for i, model in enumerate(models, 1):
        print(f"    [{i}] {model['name']} - {model['description']}")
        print(f"        Year: {model['year']}, Dir: {model['model_dir']}")
    print()

    # Create output directory
    output_dir = create_output_dir(models, args.output_dir)
    print(f"  Output directory: {output_dir}")

    # Copy models CSV to output directory
    import shutil
    shutil.copy(csv_path, output_dir / 'modelsToPlot.csv')
    print()

    # Load configuration
    config = load_config()

    # ========================================================================
    # STEP 1: Timeseries Comparison
    # ========================================================================
    if not args.skip_timeseries:
        print_header("Step 1: Timeseries Comparison")

        script_dir = Path(__file__).parent
        timeseries_script = script_dir / 'make_multimodel_timeseries.py'

        if timeseries_script.exists():
            import subprocess
            cmd = [sys.executable, str(timeseries_script), str(output_dir)]
            if args.debug:
                cmd.append('--debug')

            result = subprocess.run(cmd, cwd=output_dir)
            if result.returncode != 0:
                print_warning("Timeseries generation failed")
            else:
                print_success("Timeseries complete\n")
        else:
            print_warning(f"{timeseries_script} not found, skipping\n")

    # ========================================================================
    # STEP 2: Spatial Comparison Maps
    # ========================================================================
    if not args.skip_maps:
        print_header("Step 2: Spatial Comparison Maps")

        script_dir = Path(__file__).parent
        maps_script = script_dir / 'make_multimodel_maps.py'

        if maps_script.exists():
            import subprocess
            cmd = [sys.executable, str(maps_script),
                   str(output_dir / 'modelsToPlot.csv'),
                   str(output_dir)]

            result = subprocess.run(cmd)
            if result.returncode != 0:
                print_warning("Maps generation failed")
            else:
                print_success("Spatial maps complete\n")
        else:
            print_warning(f"{maps_script} not found, skipping\n")

    # ========================================================================
    # STEP 3: Transect Comparisons
    # ========================================================================
    if not args.skip_transects:
        print_header("Step 3: Vertical Transect Comparisons")

        script_dir = Path(__file__).parent
        transects_script = script_dir / 'make_multimodel_transects.py'

        if transects_script.exists():
            import subprocess
            cmd = [sys.executable, str(transects_script),
                   str(output_dir / 'modelsToPlot.csv'),
                   str(output_dir)]

            result = subprocess.run(cmd)
            if result.returncode != 0:
                print_warning("Transects generation failed")
            else:
                print_success("Transects complete\n")
        else:
            print_warning(f"{transects_script} not found, skipping\n")

    # ========================================================================
    # STEP 4: Configuration Comparison (2-model only)
    # ========================================================================
    if not args.skip_config_comparison and n_models == 2:
        print_header("Step 4: Configuration Comparison")

        script_dir = Path(__file__).parent

        # Compare setUpData files
        print_step(1, 2, "Comparing setUpData configuration")
        setupdata_script = script_dir / 'compare_setupdata.py'
        if setupdata_script.exists():
            import subprocess
            # Construct full model run directory paths
            model1_run_dir = Path(models[0]['model_dir']) / models[0]['name']
            model2_run_dir = Path(models[1]['model_dir']) / models[1]['name']

            cmd = [sys.executable, str(setupdata_script),
                   str(model1_run_dir),
                   str(model2_run_dir),
                   models[0]['name'],
                   models[1]['name'],
                   '--output', str(output_dir / 'setupdata_comparison.md')]

            # Run in output_dir for consistency
            subprocess.run(cmd, cwd=output_dir)

        # Compare namelists
        print_step(2, 2, "Comparing namelist.trc.sms")
        namelist_script = script_dir / 'compare_namelists.py'
        if namelist_script.exists():
            import subprocess
            # Construct full model run directory paths
            model1_run_dir = Path(models[0]['model_dir']) / models[0]['name']
            model2_run_dir = Path(models[1]['model_dir']) / models[1]['name']

            cmd = [sys.executable, str(namelist_script),
                   str(model1_run_dir),
                   str(model2_run_dir),
                   models[0]['name'],
                   models[1]['name'],
                   '--output', str(output_dir / 'namelist_comparison.md'),
                   '--namelist-name', 'namelist.trc.sms']

            # Run in output_dir so heatmap PNG files are saved there
            subprocess.run(cmd, cwd=output_dir)

        print_success("Configuration comparison complete\n")

    # ========================================================================
    # STEP 5: HTML Report Generation
    # ========================================================================
    if not args.skip_html:
        print_header("Step 5: HTML Report Generation")

        script_dir = Path(__file__).parent
        html_script = script_dir / 'make_multimodel_html.sh'

        if html_script.exists():
            import subprocess
            folder_name = output_dir.name
            cmd = [str(html_script), folder_name, '0']

            result = subprocess.run(cmd, cwd=output_dir)
            if result.returncode != 0:
                print_warning("HTML generation failed")
            else:
                print_success("HTML report complete\n")
        else:
            print_warning(f"{html_script} not found, skipping\n")

    # ========================================================================
    # Summary
    # ========================================================================
    print_header("Multi-Model Visualization Complete!")
    print(f"  Output directory: {output_dir}")
    print()

    # List key generated files
    print("Generated files:")
    key_files = [
        'timeseries_*.png',
        'map_*.png',
        'transect_*.png',
        '*.html',
        '*_comparison.md'
    ]

    file_count = 0
    for pattern in key_files:
        files = sorted(output_dir.glob(pattern))
        file_count += len(files)

    print(f"  Total: {file_count} files")

    # Look for generated HTML file
    html_files = list(output_dir.glob('*.html'))
    if html_files:
        print(f"  View report: {html_files[0]}")
    print()


if __name__ == '__main__':
    main()
