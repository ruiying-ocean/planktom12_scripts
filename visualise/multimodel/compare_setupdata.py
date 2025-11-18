#!/usr/bin/env python3
"""
Compare setUpData configuration files between two models and generate markdown diff.

This script parses setUpData_*.dat files and creates a markdown table comparison.
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Tuple, List


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


def compare_configs(config1: Dict[str, str], config2: Dict[str, str]) -> Tuple[List[Tuple[str, str, str]], List[str], List[str]]:
    """
    Compare two configuration dictionaries and find differences.

    Args:
        config1: First configuration dictionary
        config2: Second configuration dictionary

    Returns:
        Tuple of (differences, only_in_1, only_in_2)
        - differences: List of (key, value1, value2) tuples
        - only_in_1: Keys only in first config
        - only_in_2: Keys only in second config
    """
    all_keys = set(config1.keys()) | set(config2.keys())

    differences = []
    only_in_1 = []
    only_in_2 = []

    for key in sorted(all_keys):
        val1 = config1.get(key)
        val2 = config2.get(key)

        if key in config1 and key in config2:
            # Compare values
            if val1 != val2:
                differences.append((key, val1, val2))
        elif key in config1:
            only_in_1.append(key)
        else:
            only_in_2.append(key)

    return differences, only_in_1, only_in_2


def generate_markdown_diff(model1_name: str, model2_name: str,
                           differences: List[Tuple[str, str, str]],
                           only_in_1: List[str], only_in_2: List[str],
                           config1: Dict[str, str], config2: Dict[str, str]) -> str:
    """
    Generate markdown table showing configuration differences.

    Args:
        model1_name: Name of first model
        model2_name: Name of second model
        differences: List of differing parameters
        only_in_1: Parameters only in model 1
        only_in_2: Parameters only in model 2
        config1: First config (for looking up values)
        config2: Second config (for looking up values)

    Returns:
        Markdown string with comparison table
    """
    total_diffs = len(differences) + len(only_in_1) + len(only_in_2)

    lines = []

    # Show message if no differences
    if total_diffs == 0:
        lines.append('*No differences found in setUpData configuration.*')
        lines.append('')
        return '\n'.join(lines)

    # Changed parameters
    if differences:
        lines.append('### Changed Parameters')
        lines.append('')
        lines.append(f'| Parameter | {model1_name} | {model2_name} |')
        lines.append('|:----------|:--------------|:--------------|')

        for param, val1, val2 in differences:
            # Escape pipe characters
            val1_esc = val1.replace('|', '\\|')
            val2_esc = val2.replace('|', '\\|')
            lines.append(f'| **{param}** | `{val1_esc}` | `{val2_esc}` |')

        lines.append('')

    # Parameters only in model 1
    if only_in_1:
        lines.append(f'### Only in {model1_name}')
        lines.append('')
        lines.append('| Parameter | Value |')
        lines.append('|:----------|:------|')

        for param in only_in_1:
            val = config1[param].replace('|', '\\|')
            lines.append(f'| **{param}** | `{val}` |')

        lines.append('')

    # Parameters only in model 2
    if only_in_2:
        lines.append(f'### Only in {model2_name}')
        lines.append('')
        lines.append('| Parameter | Value |')
        lines.append('|:----------|:------|')

        for param in only_in_2:
            val = config2[param].replace('|', '\\|')
            lines.append(f'| **{param}** | `{val}` |')

        lines.append('')

    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(
        description='Compare setUpData configuration files between two models'
    )
    parser.add_argument('model1_path', type=str, help='Path to first model directory')
    parser.add_argument('model2_path', type=str, help='Path to second model directory')
    parser.add_argument('model1_name', type=str, help='Name of first model')
    parser.add_argument('model2_name', type=str, help='Name of second model')
    parser.add_argument('--output', '-o', type=str, default='setupdata_comparison.md',
                       help='Output markdown file (default: setupdata_comparison.md)')

    args = parser.parse_args()

    # Find setUpData files in model directories
    path1 = None
    path2 = None

    model_dir1 = Path(args.model1_path)
    model_dir2 = Path(args.model2_path)

    # Look for setUpData_*.dat files
    setupdata_files1 = list(model_dir1.glob('setUpData_*.dat'))
    setupdata_files2 = list(model_dir2.glob('setUpData_*.dat'))

    if setupdata_files1:
        path1 = setupdata_files1[0]
    if setupdata_files2:
        path2 = setupdata_files2[0]

    # Check if files exist
    missing = []
    if not path1:
        missing.append(f"{args.model1_name}: No setUpData_*.dat found in {model_dir1}")
    if not path2:
        missing.append(f"{args.model2_name}: No setUpData_*.dat found in {model_dir2}")

    if missing:
        markdown_output = '**⚠ Warning:** Could not find setUpData configuration files:\n\n'
        for item in missing:
            markdown_output += f'- `{item}`\n'
        markdown_output += '\n'

        with open(args.output, 'w') as f:
            f.write(markdown_output)

        print(f"Warning: Missing setUpData files. Wrote warning to {args.output}", file=sys.stderr)
        sys.exit(0)

    # Load configs
    config1 = load_setupdata(path1)
    config2 = load_setupdata(path2)

    # Compare
    differences, only_in_1, only_in_2 = compare_configs(config1, config2)

    # Generate markdown diff
    markdown_output = generate_markdown_diff(args.model1_name, args.model2_name,
                                            differences, only_in_1, only_in_2,
                                            config1, config2)

    # Write output
    with open(args.output, 'w') as f:
        f.write(markdown_output)

    print(f"Comparison written to {args.output}")

    # Print summary to stdout
    total_diffs = len(differences) + len(only_in_1) + len(only_in_2)
    if total_diffs == 0:
        print(f"✓ No differences found between {args.model1_name} and {args.model2_name}")
    else:
        print(f"Found {total_diffs} difference(s):")
        if differences:
            print(f"  • {len(differences)} parameter(s) with different values")
        if only_in_1:
            print(f"  • {len(only_in_1)} parameter(s) only in {args.model1_name}")
        if only_in_2:
            print(f"  • {len(only_in_2)} parameter(s) only in {args.model2_name}")


if __name__ == '__main__':
    main()
