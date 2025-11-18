#!/usr/bin/env python3
"""
Compare namelist.trc.sms files between two models and generate markdown diff with HTML heatmaps.

This script parses Fortran namelist files and creates a markdown table comparison,
with special HTML heatmap visualization for matrix parameters like rn_prfzoo.
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Any, Tuple, List
import warnings

try:
    import f90nml
except ImportError:
    print("Error: f90nml library not found. Install with: pip install f90nml", file=sys.stderr)
    sys.exit(1)

try:
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib/numpy not available. Heatmaps will be disabled.", file=sys.stderr)


def load_namelist(filepath: Path) -> Dict[str, Any]:
    """
    Load a Fortran namelist file.

    Args:
        filepath: Path to the namelist file

    Returns:
        Dictionary containing namelist data
    """
    if not filepath.exists():
        return {}

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            nml = f90nml.read(filepath)
        return nml
    except Exception as e:
        print(f"Warning: Failed to parse {filepath}: {e}", file=sys.stderr)
        return {}


def flatten_namelist(nml: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    """
    Flatten nested namelist structure into dot-notation keys.

    Args:
        nml: Namelist dictionary (potentially nested)
        prefix: Prefix for keys (used in recursion)

    Returns:
        Flattened dictionary with dot-notation keys
    """
    flattened = {}

    for key, value in nml.items():
        full_key = f"{prefix}.{key}" if prefix else key

        # If value is a namelist group (dict-like), recurse
        if isinstance(value, (dict, f90nml.Namelist)):
            flattened.update(flatten_namelist(value, full_key))
        else:
            flattened[full_key] = value

    return flattened


def format_value(value: Any) -> str:
    """
    Format a namelist value for display.

    Args:
        value: Value from namelist

    Returns:
        Formatted string representation
    """
    if isinstance(value, bool):
        return ".true." if value else ".false."
    elif isinstance(value, str):
        return f"'{value}'"
    elif isinstance(value, (list, tuple)):
        # Format arrays
        if len(value) == 0:
            return "[]"
        # For large arrays, show count instead
        if len(value) > 10:
            return f"[array of {len(value)} elements]"
        formatted_items = [format_value(v) for v in value]
        return "[" + ", ".join(formatted_items) + "]"
    elif value is None:
        return "null"
    else:
        return str(value)


def compare_namelists(nml1: Dict[str, Any], nml2: Dict[str, Any]) -> Tuple[List[Tuple[str, Any, Any]], List[str], List[str]]:
    """
    Compare two namelists and find differences.

    Args:
        nml1: First namelist dictionary
        nml2: Second namelist dictionary

    Returns:
        Tuple of (differences, only_in_1, only_in_2)
        - differences: List of (key, value1, value2) tuples
        - only_in_1: Keys only in first namelist
        - only_in_2: Keys only in second namelist
    """
    flat1 = flatten_namelist(nml1)
    flat2 = flatten_namelist(nml2)

    all_keys = set(flat1.keys()) | set(flat2.keys())

    differences = []
    only_in_1 = []
    only_in_2 = []

    for key in sorted(all_keys):
        val1 = flat1.get(key)
        val2 = flat2.get(key)

        if key in flat1 and key in flat2:
            # Compare values
            if val1 != val2:
                differences.append((key, val1, val2))
        elif key in flat1:
            only_in_1.append(key)
        else:
            only_in_2.append(key)

    return differences, only_in_1, only_in_2


def is_matrix_param(param: str, value: Any) -> bool:
    """
    Check if a parameter is a matrix/array that should get special visualization.

    Args:
        param: Parameter name
        value: Parameter value

    Returns:
        True if this is a matrix parameter
    """
    # Known matrix parameters
    matrix_params = ['rn_prfzoo', 'rn_prfphy', 'rn_prfbac']
    param_base = param.split('.')[-1] if '.' in param else param

    # Check if it's a known matrix
    if param_base.lower() in matrix_params:
        return True

    return False


def calculate_normalized_prfzoo(rn_prfzoo: np.ndarray, rn_bmspft: np.ndarray) -> np.ndarray:
    """
    Calculate normalized/effective grazing preferences using biomass weighting.

    This implements the PlankTOM formula:
    prfzoo(jm,jl) = (totbio / biowprf) * rn_prfzoo(jm,jl)

    where:
    totbio  = Σ rn_bmspft(jl)  where rn_prfzoo(jm,jl) > 0
    biowprf = Σ (rn_bmspft(jl) * rn_prfzoo(jm,jl))

    Args:
        rn_prfzoo: Raw preference matrix (15 food sources × 5 zooplankton)
        rn_bmspft: Biomass array (15 food sources)

    Returns:
        Normalized preference matrix
    """
    if not MATPLOTLIB_AVAILABLE:
        return rn_prfzoo

    prfzoo = np.zeros_like(rn_prfzoo)

    # Loop over each zooplankton type (columns)
    for jm in range(rn_prfzoo.shape[1]):
        # Calculate totbio and biowprf for this zooplankton
        totbio = 0.0
        biowprf = 0.0

        for jl in range(rn_prfzoo.shape[0]):
            if rn_prfzoo[jl, jm] > 1e-10:  # rtrn threshold
                totbio += rn_bmspft[jl]
                biowprf += rn_bmspft[jl] * rn_prfzoo[jl, jm]

        # Calculate normalized preferences
        if biowprf > 1e-10:
            for jl in range(rn_prfzoo.shape[0]):
                prfzoo[jl, jm] = (totbio / biowprf) * rn_prfzoo[jl, jm]

    return prfzoo


def parse_array_to_matrix(value: Any, param: str) -> np.ndarray:
    """
    Parse array value into numpy matrix.

    Args:
        value: Array value from namelist
        param: Parameter name (to determine shape)

    Returns:
        Numpy array
    """
    if not isinstance(value, (list, tuple)):
        value = [value]

    arr = np.array(value, dtype=float)

    # Known matrix shapes
    param_base = param.split('.')[-1] if '.' in param else param
    if param_base.lower() == 'rn_prfzoo':
        # 15 food sources x 5 zooplankton
        if len(arr) == 75:
            arr = arr.reshape(15, 5)

    return arr


def create_matrix_heatmap(matrix1: np.ndarray, matrix2: np.ndarray,
                          model1_name: str, model2_name: str,
                          param: str, output_path: str) -> bool:
    """
    Create side-by-side heatmap comparison of two matrices.

    Colors are normalized per column (each zooplankton type separately)
    to better show relative differences in preferences.

    Args:
        matrix1: First matrix
        matrix2: Second matrix
        model1_name: Name of first model
        model2_name: Name of second model
        param: Parameter name
        output_path: Path to save figure

    Returns:
        True if successful
    """
    if not MATPLOTLIB_AVAILABLE:
        return False

    try:
        fig, axes = plt.subplots(1, 3, figsize=(16, 7))

        # Define axis labels for rn_prfzoo
        param_base = param.split('.')[-1] if '.' in param else param
        if 'rn_prfzoo' in param_base.lower():
            food_sources = ['POC', 'GOC', 'HOC', 'BAC', 'PRO', 'PTE', 'MES', 'GEL',
                           'MAC', 'DIA', 'MIX', 'COC', 'PIC', 'PHA', 'FIX']
            zooplankton = ['PRO', 'PTE', 'MES', 'GEL', 'CRU']
        else:
            food_sources = [str(i) for i in range(matrix1.shape[0])]
            zooplankton = [str(i) for i in range(matrix1.shape[1])]

        # Create column-normalized versions for visualization
        # Normalize each column (zooplankton type) separately to [0, 1]
        matrix1_norm = np.zeros_like(matrix1)
        matrix2_norm = np.zeros_like(matrix2)

        for col in range(matrix1.shape[1]):
            col_min = min(matrix1[:, col].min(), matrix2[:, col].min())
            col_max = max(matrix1[:, col].max(), matrix2[:, col].max())

            if col_max > col_min:
                matrix1_norm[:, col] = (matrix1[:, col] - col_min) / (col_max - col_min)
                matrix2_norm[:, col] = (matrix2[:, col] - col_min) / (col_max - col_min)
            else:
                # If all values in column are the same, set to 0.5 (middle of colormap)
                matrix1_norm[:, col] = 0.5
                matrix2_norm[:, col] = 0.5

        # Model 1 - using column-normalized data
        im1 = axes[0].imshow(matrix1_norm, cmap='Blues', vmin=0, vmax=1, aspect='auto')
        axes[0].set_title(f'{model1_name}', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Food Source', fontsize=10)
        axes[0].set_xlabel('Zooplankton Type', fontsize=10)
        axes[0].set_yticks(range(len(food_sources)))
        axes[0].set_yticklabels(food_sources, fontsize=8)
        axes[0].set_xticks(range(len(zooplankton)))
        axes[0].set_xticklabels(zooplankton, fontsize=8)
        cbar1 = plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
        cbar1.set_label('Relative preference\n(per zooplankton)', fontsize=8)

        # Model 2 - using column-normalized data
        im2 = axes[1].imshow(matrix2_norm, cmap='Blues', vmin=0, vmax=1, aspect='auto')
        axes[1].set_title(f'{model2_name}', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Zooplankton Type', fontsize=10)
        axes[1].set_yticks(range(len(food_sources)))
        axes[1].set_yticklabels(food_sources, fontsize=8)
        axes[1].set_xticks(range(len(zooplankton)))
        axes[1].set_xticklabels(zooplankton, fontsize=8)
        cbar2 = plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
        cbar2.set_label('Relative preference\n(per zooplankton)', fontsize=8)

        # Difference - still using absolute values (not normalized)
        diff = matrix2 - matrix1
        max_abs_diff = np.abs(diff).max()
        im3 = axes[2].imshow(diff, cmap='RdBu_r', vmin=-max_abs_diff, vmax=max_abs_diff, aspect='auto')
        axes[2].set_title('Difference (Model2 - Model1)', fontsize=12, fontweight='bold')
        axes[2].set_xlabel('Zooplankton Type', fontsize=10)
        axes[2].set_yticks(range(len(food_sources)))
        axes[2].set_yticklabels(food_sources, fontsize=8)
        axes[2].set_xticks(range(len(zooplankton)))
        axes[2].set_xticklabels(zooplankton, fontsize=8)
        cbar3 = plt.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)
        cbar3.set_label('Absolute difference', fontsize=8)

        plt.tight_layout()
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.close()

        return True
    except Exception as e:
        print(f"Warning: Failed to create heatmap: {e}", file=sys.stderr)
        return False


def generate_sparse_diff(matrix1: np.ndarray, matrix2: np.ndarray,
                        param: str, model1_name: str, model2_name: str) -> str:
    """
    Generate sparse diff markdown table showing only changed elements.

    Shows actual effective/normalized values (not the column-normalized visualization).

    Args:
        matrix1: First matrix (actual values)
        matrix2: Second matrix (actual values)
        param: Parameter name
        model1_name: Name of first model
        model2_name: Name of second model

    Returns:
        Markdown string with sparse diff table
    """
    # Find differences
    diff_mask = matrix1 != matrix2
    num_changed = np.sum(diff_mask)
    total_elements = matrix1.size

    if num_changed == 0:
        return '*All values identical*\n\n'

    # Get row/col labels if known
    param_base = param.split('.')[-1] if '.' in param else param
    row_labels = None
    col_labels = None

    if param_base.lower() == 'rn_prfzoo':
        row_labels = ['POC', 'GOC', 'HOC', 'BAC', 'PRO', 'PTE', 'MES', 'GEL',
                     'MAC', 'DIA', 'MIX', 'COC', 'PIC', 'PHA', 'FIX']
        col_labels = ['PRO(zoo)', 'PTE', 'MES', 'GEL', 'CRU']

    lines = []
    lines.append(f'**Changed elements: {num_changed} of {total_elements}**')
    lines.append('')
    lines.append('*Showing raw namelist values (heatmap above shows normalized/effective values with relative color scale)*')
    lines.append('')

    # List changed elements in a table
    lines.append(f'| Parameter | {model1_name} | {model2_name} |')
    lines.append('|:----------|:--------------|:--------------|')

    # List changed elements
    rows, cols = np.where(diff_mask)
    for i, (row, col) in enumerate(zip(rows, cols)):
        if i >= 20:  # Limit to first 20 changes
            lines.append(f'| *...and {num_changed - 20} more* | | |')
            break

        val1 = matrix1[row, col]
        val2 = matrix2[row, col]

        # Format with labels if available
        if row_labels and col_labels:
            row_label = row_labels[row] if row < len(row_labels) else f'Row {row}'
            col_label = col_labels[col] if col < len(col_labels) else f'Col {col}'
            location = f'{row_label} → {col_label}'
        else:
            location = f'[{row}, {col}]'

        lines.append(f'| **{location}** | `{val1:.3f}` | `{val2:.3f}` |')

    lines.append('')
    lines.append(f'*({total_elements - num_changed} values unchanged)*')
    lines.append('')

    return '\n'.join(lines)


def format_param_name(param: str) -> str:
    """
    Format parameter name for better display.
    Converts 'natpre.rn_prfzoo' to '&natpre / rn_prfzoo'

    Args:
        param: Dot-notation parameter name

    Returns:
        Formatted parameter name
    """
    if '.' in param:
        parts = param.split('.')
        # First part is the namelist group, rest is the parameter path
        group = parts[0]
        param_path = '.'.join(parts[1:])
        return f'&{group} / {param_path}'
    return param


def generate_markdown_diff(model1_name: str, model2_name: str,
                           differences: List[Tuple[str, Any, Any]],
                           only_in_1: List[str], only_in_2: List[str],
                           nml1: Dict[str, Any], nml2: Dict[str, Any]) -> str:
    """
    Generate markdown diff with embedded HTML for matrix visualizations.

    Args:
        model1_name: Name of first model
        model2_name: Name of second model
        differences: List of differing parameters
        only_in_1: Parameters only in model 1
        only_in_2: Parameters only in model 2
        nml1: First namelist (for looking up values)
        nml2: Second namelist (for looking up values)

    Returns:
        Markdown string with diff and embedded HTML for matrices
    """
    flat1 = flatten_namelist(nml1)
    flat2 = flatten_namelist(nml2)

    total_diffs = len(differences) + len(only_in_1) + len(only_in_2)

    lines = []

    # Show message if no differences
    if total_diffs == 0:
        lines.append('*No differences found in namelist.trc.sms configuration.*')
        lines.append('')
        return '\n'.join(lines)

    # Changed parameters
    if differences:
        lines.append('### Changed Parameters')
        lines.append('')

        # Separate matrix and non-matrix parameters
        matrix_params = []
        regular_params = []

        for param, val1, val2 in differences:
            if MATPLOTLIB_AVAILABLE and is_matrix_param(param, val1):
                matrix_params.append((param, val1, val2))
            else:
                regular_params.append((param, val1, val2))

        # Show regular parameters in table
        if regular_params:
            lines.append(f'| Parameter | {model1_name} | {model2_name} |')
            lines.append('|:----------|:--------------|:--------------|')

            for param, val1, val2 in regular_params:
                formatted_param = format_param_name(param)
                val1_str = format_value(val1).replace('|', '\\|')
                val2_str = format_value(val2).replace('|', '\\|')
                lines.append(f'| **{formatted_param}** | `{val1_str}` | `{val2_str}` |')

            lines.append('')

        # Show matrix parameters with heatmaps
        if matrix_params:
            for param, val1, val2 in matrix_params:
                try:
                    # Parse matrices
                    matrix1_raw = parse_array_to_matrix(val1, param)
                    matrix2_raw = parse_array_to_matrix(val2, param)

                    # For rn_prfzoo, calculate normalized values
                    param_base = param.split('.')[-1] if '.' in param else param
                    if param_base.lower() == 'rn_prfzoo':
                        # Get rn_bmspft from namelists
                        bmspft1 = flat1.get('natpre.rn_bmspft')
                        bmspft2 = flat2.get('natpre.rn_bmspft')

                        if bmspft1 is not None and bmspft2 is not None:
                            bmspft1_arr = np.array(bmspft1, dtype=float) if isinstance(bmspft1, (list, tuple)) else np.array([bmspft1], dtype=float)
                            bmspft2_arr = np.array(bmspft2, dtype=float) if isinstance(bmspft2, (list, tuple)) else np.array([bmspft2], dtype=float)

                            # Calculate normalized (effective) preferences
                            matrix1 = calculate_normalized_prfzoo(matrix1_raw, bmspft1_arr)
                            matrix2 = calculate_normalized_prfzoo(matrix2_raw, bmspft2_arr)
                            display_param = format_param_name(param) + " (normalized/effective)"
                        else:
                            # Fallback to raw values if rn_bmspft not found
                            matrix1 = matrix1_raw
                            matrix2 = matrix2_raw
                            display_param = format_param_name(param)
                    else:
                        matrix1 = matrix1_raw
                        matrix2 = matrix2_raw
                        display_param = format_param_name(param)

                    # Generate heatmap
                    heatmap_filename = f'heatmap_{param.replace(".", "_")}.png'
                    if create_matrix_heatmap(matrix1, matrix2, model1_name, model2_name,
                                            param, heatmap_filename):
                        # Add subheading for this matrix parameter
                        lines.append(f'#### {display_param}')
                        lines.append('')

                        # Embed image in markdown
                        lines.append(f'![{display_param}]({heatmap_filename})')
                        lines.append('')

                        # Show sparse diff using RAW values (not normalized)
                        lines.append(generate_sparse_diff(matrix1_raw, matrix2_raw, param, model1_name, model2_name))
                    else:
                        # Fallback to table format
                        lines.append(f'| **{format_param_name(param)}** | `{format_value(val1)}` | `{format_value(val2)}` |')

                except Exception as e:
                    print(f"Warning: Matrix visualization failed for {param}: {e}", file=sys.stderr)
                    # Fallback to table format
                    formatted_param = format_param_name(param)
                    val1_str = format_value(val1).replace('|', '\\|')
                    val2_str = format_value(val2).replace('|', '\\|')
                    lines.append(f'| **{formatted_param}** | `{val1_str}` | `{val2_str}` |')

    # Parameters only in model 1
    if only_in_1:
        lines.append(f'### Only in {model1_name}')
        lines.append('')
        lines.append('| Parameter | Value |')
        lines.append('|:----------|:------|')

        for param in only_in_1:
            formatted_param = format_param_name(param)
            param_only = param.split('.')[-1] if '.' in param else param
            val_str = format_value(flat1[param]).replace('|', '\\|')
            lines.append(f'| **{formatted_param}** | `{param_only} = {val_str}` |')

        lines.append('')

    # Parameters only in model 2
    if only_in_2:
        lines.append(f'### Only in {model2_name}')
        lines.append('')
        lines.append('| Parameter | Value |')
        lines.append('|:----------|:------|')

        for param in only_in_2:
            formatted_param = format_param_name(param)
            param_only = param.split('.')[-1] if '.' in param else param
            val_str = format_value(flat2[param]).replace('|', '\\|')
            lines.append(f'| **{formatted_param}** | `{param_only} = {val_str}` |')

        lines.append('')

    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(
        description='Compare namelist.trc.sms files between two models'
    )
    parser.add_argument('model1_path', type=str, help='Path to first model directory')
    parser.add_argument('model2_path', type=str, help='Path to second model directory')
    parser.add_argument('model1_name', type=str, help='Name of first model')
    parser.add_argument('model2_name', type=str, help='Name of second model')
    parser.add_argument('--output', '-o', type=str, default='namelist_comparison.md',
                       help='Output markdown file (default: namelist_comparison.md)')
    parser.add_argument('--namelist-name', type=str, default='namelist.trc.sms',
                       help='Name of namelist file to compare (default: namelist.trc.sms)')

    args = parser.parse_args()

    # Construct paths to namelist files
    path1 = Path(args.model1_path) / args.namelist_name
    path2 = Path(args.model2_path) / args.namelist_name

    # Check if files exist
    missing = []
    if not path1.exists():
        missing.append(f"{args.model1_name}: {path1}")
    if not path2.exists():
        missing.append(f"{args.model2_name}: {path2}")

    if missing:
        markdown_output = f'**⚠ Warning:** Could not find `{args.namelist_name}` for:\n\n'
        for item in missing:
            markdown_output += f'- `{item}`\n'
        markdown_output += '\n'

        with open(args.output, 'w') as f:
            f.write(markdown_output)

        print(f"Warning: Missing namelist files. Wrote warning to {args.output}", file=sys.stderr)
        sys.exit(0)

    # Load namelists
    nml1 = load_namelist(path1)
    nml2 = load_namelist(path2)

    # Compare
    differences, only_in_1, only_in_2 = compare_namelists(nml1, nml2)

    # Generate markdown diff
    markdown_output = generate_markdown_diff(args.model1_name, args.model2_name,
                                            differences, only_in_1, only_in_2,
                                            nml1, nml2)

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
