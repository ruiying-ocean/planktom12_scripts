#!/usr/bin/env python3
"""
Extract essential parameters from setUpData configuration file and generate HTML table.

This script reads a setUpData_*.dat file and creates a summary table of key parameters
for display in single-model reports.
"""

import argparse
import sys
from pathlib import Path
from typing import Dict


# Define essential parameters to display
ESSENTIAL_PARAMS = [
    'model',
    'yearStart',
    'yearEnd',
    'CO2',
    'type',
    'forcing',
    'dust.orca.nc',
    'restart_trc.nc',
    'namelist_top_cfg',
]

# Parameter descriptions
PARAM_DESCRIPTIONS = {
    'model': 'Model Version',
    'yearStart': 'Start Year',
    'yearEnd': 'End Year',
    'CO2': 'CO₂ Mode',
    'type': 'Run Type',
    'forcing': 'Forcing Data',
    'dust.orca.nc': 'Dust Deposition File',
    'restart_trc.nc': 'Restart File (Tracers)',
    'namelist_top_cfg': 'Namelist Configuration',
}


def load_setupdata(filepath: Path) -> Dict[str, str]:
    """
    Load a setUpData configuration file.

    Args:
        filepath: Path to the setUpData file

    Returns:
        Dictionary containing configuration data
    """
    if not filepath.exists():
        return {}

    config = {}
    try:
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                # Skip empty lines and comments
                if not line or line.startswith('#'):
                    continue
                # Parse name:value format
                if ':' in line:
                    parts = line.split(':', 1)
                    if len(parts) == 2:
                        name = parts[0].strip()
                        value = parts[1].strip()
                        config[name] = value
        return config
    except Exception as e:
        print(f"Warning: Failed to parse {filepath}: {e}", file=sys.stderr)
        return {}


def format_file_path(path: str) -> str:
    """
    Format file paths for better display - show filename and relevant parent dirs.

    Args:
        path: File path string

    Returns:
        Formatted path string
    """
    if not path or len(path) < 50:
        return path

    # For long paths, show .../<last_two_dirs>/filename
    parts = path.split('/')
    if len(parts) > 3:
        return '.../' + '/'.join(parts[-3:])
    return path


def generate_summary_table(config: Dict[str, str], essential_params: list = None) -> str:
    """
    Generate Quarto-native markdown table showing essential parameters.

    Args:
        config: Configuration dictionary
        essential_params: List of parameter names to include (None = use ESSENTIAL_PARAMS)

    Returns:
        Markdown string with parameter table
    """
    if essential_params is None:
        essential_params = ESSENTIAL_PARAMS

    # Build markdown table
    lines = []

    # Table header
    lines.append('| Parameter | Value |')
    lines.append('|:----------|:------|')

    # Table rows
    for param in essential_params:
        param_label = PARAM_DESCRIPTIONS.get(param, param)
        value = config.get(param)

        if value is not None:
            # Format file paths for better display
            if param.endswith('.nc') or 'namelist' in param.lower():
                display_value = format_file_path(value)
            else:
                display_value = value
            # Escape pipe characters in values to avoid breaking table
            display_value = display_value.replace('|', '\\|')
            # Use inline code formatting for values
            lines.append(f'| **{param_label}** | `{display_value}` |')
        else:
            lines.append(f'| **{param_label}** | *not set* |')

    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(
        description='Extract essential parameters from setUpData configuration file'
    )
    parser.add_argument('setupdata_path', type=str, help='Path to setUpData file or directory containing it')
    parser.add_argument('--output', '-o', type=str, default='setupdata_summary.md',
                       help='Output markdown file (default: setupdata_summary.md)')
    parser.add_argument('--params', type=str, nargs='+',
                       help='List of parameters to include (default: predefined essential params)')

    args = parser.parse_args()

    # Check if path is a directory or file
    input_path = Path(args.setupdata_path)

    if input_path.is_dir():
        # Find setUpData_*.dat file in directory
        setupdata_files = list(input_path.glob('setUpData_*.dat'))
        if not setupdata_files:
            print(f"Error: No setUpData_*.dat file found in {input_path}", file=sys.stderr)
            sys.exit(1)
        setupdata_path = setupdata_files[0]
    else:
        setupdata_path = input_path

    if not setupdata_path.exists():
        print(f"Error: File not found: {setupdata_path}", file=sys.stderr)
        sys.exit(1)

    # Load configuration
    config = load_setupdata(setupdata_path)

    if not config:
        print(f"Warning: No configuration data loaded from {setupdata_path}", file=sys.stderr)

    # Generate summary table
    params_to_show = args.params if args.params else None
    markdown_output = generate_summary_table(config, params_to_show)

    # Write output
    with open(args.output, 'w') as f:
        f.write(markdown_output)

    print(f"Summary table written to {args.output}")

    # Print summary to stdout
    essential_count = len([p for p in (args.params or ESSENTIAL_PARAMS) if config.get(p)])
    print(f"Extracted {essential_count} parameters from {setupdata_path.name}")


if __name__ == '__main__':
    main()
