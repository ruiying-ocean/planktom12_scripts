#!/usr/bin/env python3
"""
Compare namelist.trc.sms files between two models and generate modern diff-style HTML report.

This script parses Fortran namelist files and creates a fancy side-by-side comparison
with syntax highlighting and modern UI similar to GitHub diffs.
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Any, Tuple, List
import warnings
import html

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

    # Check if it's a known matrix or a large array
    if param_base.lower() in matrix_params:
        return True

    # Check if it's a list/array with many elements
    if isinstance(value, (list, tuple)) and len(value) > 10:
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

        # Determine color scale
        vmin = min(matrix1.min(), matrix2.min())
        vmax = max(matrix1.max(), matrix2.max())

        # Define axis labels for rn_prfzoo
        param_base = param.split('/')[-1].strip() if '/' in param else param
        if 'rn_prfzoo' in param_base.lower():
            food_sources = ['POC', 'GOC', 'HOC', 'BAC', 'PRO', 'PTE', 'MES', 'GEL',
                           'MAC', 'DIA', 'MIX', 'COC', 'PIC', 'PHA', 'FIX']
            zooplankton = ['PRO', 'PTE', 'MES', 'GEL', 'CRU']
        else:
            food_sources = [str(i) for i in range(matrix1.shape[0])]
            zooplankton = [str(i) for i in range(matrix1.shape[1])]

        # Model 1
        im1 = axes[0].imshow(matrix1, cmap='Blues', vmin=vmin, vmax=vmax, aspect='auto')
        axes[0].set_title(f'{model1_name}', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Food Source', fontsize=10)
        axes[0].set_xlabel('Zooplankton Type', fontsize=10)
        axes[0].set_yticks(range(len(food_sources)))
        axes[0].set_yticklabels(food_sources, fontsize=8)
        axes[0].set_xticks(range(len(zooplankton)))
        axes[0].set_xticklabels(zooplankton, fontsize=8)
        plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

        # Model 2
        im2 = axes[1].imshow(matrix2, cmap='Blues', vmin=vmin, vmax=vmax, aspect='auto')
        axes[1].set_title(f'{model2_name}', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Zooplankton Type', fontsize=10)
        axes[1].set_yticks(range(len(food_sources)))
        axes[1].set_yticklabels(food_sources, fontsize=8)
        axes[1].set_xticks(range(len(zooplankton)))
        axes[1].set_xticklabels(zooplankton, fontsize=8)
        plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

        # Difference
        diff = matrix2 - matrix1
        max_abs_diff = np.abs(diff).max()
        im3 = axes[2].imshow(diff, cmap='RdBu_r', vmin=-max_abs_diff, vmax=max_abs_diff, aspect='auto')
        axes[2].set_title('Difference (Model2 - Model1)', fontsize=12, fontweight='bold')
        axes[2].set_xlabel('Zooplankton Type', fontsize=10)
        axes[2].set_yticks(range(len(food_sources)))
        axes[2].set_yticklabels(food_sources, fontsize=8)
        axes[2].set_xticks(range(len(zooplankton)))
        axes[2].set_xticklabels(zooplankton, fontsize=8)
        plt.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)

        plt.suptitle(f'{param}', fontsize=14, fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.close()

        return True
    except Exception as e:
        print(f"Warning: Failed to create heatmap: {e}", file=sys.stderr)
        return False


def generate_sparse_diff(matrix1: np.ndarray, matrix2: np.ndarray,
                        param: str) -> str:
    """
    Generate sparse diff listing showing only changed elements.

    Args:
        matrix1: First matrix
        matrix2: Second matrix
        param: Parameter name

    Returns:
        HTML string with sparse diff
    """
    # Find differences
    diff_mask = matrix1 != matrix2
    num_changed = np.sum(diff_mask)
    total_elements = matrix1.size

    if num_changed == 0:
        return '<p style="color: #2da44e; font-style: italic;">All values identical</p>'

    # Get row/col labels if known
    param_base = param.split('.')[-1] if '.' in param else param
    row_labels = None
    col_labels = None

    if param_base.lower() == 'rn_prfzoo':
        row_labels = ['POC', 'GOC', 'HOC', 'BAC', 'PRO', 'PTE', 'MES', 'GEL',
                     'MAC', 'DIA', 'MIX', 'COC', 'PIC', 'PHA', 'FIX']
        col_labels = ['PRO(zoo)', 'PTE', 'MES', 'GEL', 'CRU']

    html_parts = []
    html_parts.append(f'<p style="margin: 8px 0;"><strong>Changed elements: {num_changed} of {total_elements}</strong></p>')
    html_parts.append('<ul style="margin: 8px 0; padding-left: 20px; font-family: monospace; font-size: 12px;">')

    # List changed elements
    rows, cols = np.where(diff_mask)
    for i, (row, col) in enumerate(zip(rows, cols)):
        if i >= 20:  # Limit to first 20 changes
            html_parts.append(f'<li style="margin: 4px 0;">... and {num_changed - 20} more</li>')
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

        html_parts.append(f'<li style="margin: 4px 0;">{location}: <span style="color: #cf222e;">{val1:.3f}</span> → <span style="color: #2da44e;">{val2:.3f}</span></li>')

    html_parts.append('</ul>')
    html_parts.append(f'<p style="margin: 8px 0; color: #57606a; font-size: 12px; font-style: italic;">({total_elements - num_changed} values unchanged)</p>')

    return '\n'.join(html_parts)


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


def generate_modern_diff_html(model1_name: str, model2_name: str,
                               differences: List[Tuple[str, Any, Any]],
                               only_in_1: List[str], only_in_2: List[str],
                               nml1: Dict[str, Any], nml2: Dict[str, Any]) -> str:
    """
    Generate modern GitHub-style diff HTML report.

    Args:
        model1_name: Name of first model
        model2_name: Name of second model
        differences: List of differing parameters
        only_in_1: Parameters only in model 1
        only_in_2: Parameters only in model 2
        nml1: First namelist (for looking up values)
        nml2: Second namelist (for looking up values)

    Returns:
        HTML string with modern diff UI
    """
    flat1 = flatten_namelist(nml1)
    flat2 = flatten_namelist(nml2)

    total_diffs = len(differences) + len(only_in_1) + len(only_in_2)

    html_parts = []

    # Modern CSS inspired by GitHub/GitLab diffs
    html_parts.append('''
<style>
.diff-container {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    font-size: 14px;
    line-height: 1.5;
    color: #24292f;
}

.diff-header {
    background: #f6f8fa;
    border: 1px solid #d0d7de;
    border-bottom: 2px solid #0969da;
    padding: 16px 20px;
    border-radius: 6px 6px 0 0;
    margin-bottom: 0;
}

.diff-header h3 {
    margin: 0 0 4px 0;
    font-size: 16px;
    font-weight: 600;
    color: #24292f;
}

.diff-header .subtitle {
    font-size: 13px;
    color: #57606a;
}

.diff-stats {
    background: #f6f8fa;
    border: 1px solid #d0d7de;
    border-top: none;
    padding: 12px 24px;
    font-size: 13px;
    border-radius: 0 0 8px 8px;
    margin-bottom: 16px;
}

.diff-stat {
    display: inline-block;
    margin-right: 24px;
    line-height: 24px;
}

.diff-stat-icon {
    width: 12px;
    height: 12px;
    border-radius: 2px;
    display: inline-block;
    margin-right: 6px;
    vertical-align: middle;
}

.stat-changed { background-color: #fb8500; }
.stat-added { background-color: #2da44e; }
.stat-removed { background-color: #cf222e; }

.diff-file {
    border: 1px solid #d0d7de;
    border-radius: 6px;
    margin-bottom: 16px;
    overflow: hidden;
    box-shadow: 0 1px 3px rgba(0,0,0,0.12);
}

.diff-file-header {
    background: #f6f8fa;
    padding: 10px 16px;
    font-family: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace;
    font-size: 13px;
    font-weight: 600;
    border-bottom: 1px solid #d0d7de;
    color: #0969da;
    letter-spacing: 0.3px;
}

.diff-table {
    width: 100%;
    border-collapse: collapse;
    font-family: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace;
    font-size: 12px;
}

.diff-table td {
    padding: 0;
    vertical-align: top;
    border: none;
}

.diff-line {
    display: flex;
    width: 100%;
}

.diff-line-num {
    min-width: 50px;
    padding: 0 10px;
    text-align: right;
    color: #57606a;
    background: #f6f8fa;
    user-select: none;
    border-right: 1px solid #d0d7de;
}

.diff-line-content {
    padding: 0 10px;
    flex: 1;
    white-space: pre-wrap;
    word-break: break-all;
}

/* Changed line styling */
.diff-line-changed {
    background: #ffffff;
    border-left: 2px solid #0969da;
}

.diff-line-changed .diff-line-num {
    background: #f6f8fa;
    color: #57606a;
}

/* Removed line styling */
.diff-line-removed {
    background: #ffebe9;
}

.diff-line-removed .diff-line-num {
    background: #ffd7d5;
    color: #82071e;
}

.diff-line-removed .diff-line-content::before {
    content: '− ';
    color: #cf222e;
    font-weight: bold;
}

/* Added line styling */
.diff-line-added {
    background: #dafbe1;
}

.diff-line-added .diff-line-num {
    background: #b4f1c2;
    color: #055d20;
}

.diff-line-added .diff-line-content::before {
    content: '+ ';
    color: #2da44e;
    font-weight: bold;
}

/* Side-by-side diff */
.diff-split {
    display: grid;
    grid-template-columns: 1fr 1fr;
    border-bottom: 1px solid #d0d7de;
}

.diff-split:last-child {
    border-bottom: none;
}

.diff-split-side {
    display: flex;
    border-right: 1px solid #d0d7de;
}

.diff-split-side:last-child {
    border-right: none;
}

.diff-side-label {
    font-weight: 600;
    color: #0969da;
    background: #ddf4ff;
    padding: 6px 12px;
    border-bottom: 2px solid #0969da;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.param-name {
    font-weight: 600;
    color: #0969da;
    padding: 8px 12px;
    background: #f6f8fa;
    border-bottom: 1px solid #d0d7de;
}

.no-changes {
    background: #dafbe1;
    border: 1px solid #2da44e;
    border-radius: 6px;
    padding: 16px 20px;
    margin: 16px 0;
    display: flex;
    align-items: center;
    gap: 12px;
}

.no-changes-icon {
    font-size: 24px;
}

.section-divider {
    border: none;
    height: 1px;
    background: linear-gradient(to right, transparent, #d0d7de, transparent);
    margin: 24px 0;
}
</style>
''')

    html_parts.append('<div class="diff-container">')

    # Show message if no differences
    if total_diffs == 0:
        html_parts.append('<div class="no-changes">')
        html_parts.append('<span class="no-changes-icon">✓</span>')
        html_parts.append('<div>')
        html_parts.append(f'<strong>No differences found</strong><br>')
        html_parts.append('The <code>namelist.trc.sms</code> configuration is identical between both models.')
        html_parts.append('</div>')
        html_parts.append('</div>')
    else:
        # Add spacing before diff content
        html_parts.append('<div style="margin-top: 16px;"></div>')

        # Diff content
        diff_num = 0

        # Changed parameters - side-by-side view
        if differences:
            for param, val1, val2 in differences:
                diff_num += 1
                html_parts.append('<div class="diff-file">')
                html_parts.append(f'<div class="diff-file-header">{html.escape(format_param_name(param))}</div>')

                # Check if this is a matrix parameter
                if MATPLOTLIB_AVAILABLE and is_matrix_param(param, val1):
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
                                                display_param, heatmap_filename):
                            # Show heatmap
                            html_parts.append('<div style="padding: 16px;">')
                            html_parts.append(f'<img src="{heatmap_filename}" style="max-width: 100%; height: auto;" />')
                            html_parts.append('</div>')

                            # Show sparse diff
                            html_parts.append('<div style="padding: 0 16px 16px 16px;">')
                            html_parts.append(generate_sparse_diff(matrix1, matrix2, param))
                            html_parts.append('</div>')
                        else:
                            # Fallback to regular display
                            html_parts.append('<div class="diff-split">')
                            html_parts.append('<div class="diff-split-side">')
                            html_parts.append(f'<div class="diff-line diff-line-changed" style="width: 100%;">')
                            html_parts.append(f'<div class="diff-line-num">{html.escape(model1_name)}</div>')
                            html_parts.append(f'<div class="diff-line-content">{html.escape(format_value(val1))}</div>')
                            html_parts.append('</div>')
                            html_parts.append('</div>')
                            html_parts.append('<div class="diff-split-side">')
                            html_parts.append(f'<div class="diff-line diff-line-changed" style="width: 100%;">')
                            html_parts.append(f'<div class="diff-line-num">{html.escape(model2_name)}</div>')
                            html_parts.append(f'<div class="diff-line-content">{html.escape(format_value(val2))}</div>')
                            html_parts.append('</div>')
                            html_parts.append('</div>')
                            html_parts.append('</div>')
                    except Exception as e:
                        print(f"Warning: Matrix visualization failed for {param}: {e}", file=sys.stderr)
                        # Fallback to regular display
                        html_parts.append('<div class="diff-split">')
                        html_parts.append('<div class="diff-split-side">')
                        html_parts.append(f'<div class="diff-line diff-line-changed" style="width: 100%;">')
                        html_parts.append(f'<div class="diff-line-num">{html.escape(model1_name)}</div>')
                        html_parts.append(f'<div class="diff-line-content">{html.escape(format_value(val1))}</div>')
                        html_parts.append('</div>')
                        html_parts.append('</div>')
                        html_parts.append('<div class="diff-split-side">')
                        html_parts.append(f'<div class="diff-line diff-line-changed" style="width: 100%;">')
                        html_parts.append(f'<div class="diff-line-num">{html.escape(model2_name)}</div>')
                        html_parts.append(f'<div class="diff-line-content">{html.escape(format_value(val2))}</div>')
                        html_parts.append('</div>')
                        html_parts.append('</div>')
                        html_parts.append('</div>')
                else:
                    # Regular parameter - side-by-side view
                    html_parts.append('<div class="diff-split">')

                    # Left side (model 1)
                    html_parts.append('<div class="diff-split-side">')
                    html_parts.append(f'<div class="diff-line diff-line-changed" style="width: 100%;">')
                    html_parts.append(f'<div class="diff-line-num">{html.escape(model1_name)}</div>')
                    html_parts.append(f'<div class="diff-line-content">{html.escape(format_value(val1))}</div>')
                    html_parts.append('</div>')
                    html_parts.append('</div>')

                    # Right side (model 2)
                    html_parts.append('<div class="diff-split-side">')
                    html_parts.append(f'<div class="diff-line diff-line-changed" style="width: 100%;">')
                    html_parts.append(f'<div class="diff-line-num">{html.escape(model2_name)}</div>')
                    html_parts.append(f'<div class="diff-line-content">{html.escape(format_value(val2))}</div>')
                    html_parts.append('</div>')
                    html_parts.append('</div>')

                    html_parts.append('</div>')

                html_parts.append('</div>')

        # Removed parameters (only in model 1)
        if only_in_1:
            if differences:
                html_parts.append('<hr class="section-divider">')

            for param in only_in_1:
                diff_num += 1
                formatted_param = format_param_name(param)
                # Extract just the parameter name (after the /)
                param_only = param.split('.')[-1] if '.' in param else param
                html_parts.append('<div class="diff-file">')
                html_parts.append(f'<div class="diff-file-header">{html.escape(formatted_param)}</div>')

                html_parts.append('<div class="diff-line diff-line-removed">')
                html_parts.append(f'<div class="diff-line-num">{html.escape(model1_name)}</div>')
                html_parts.append(f'<div class="diff-line-content">{html.escape(param_only)} = {html.escape(format_value(flat1[param]))}</div>')
                html_parts.append('</div>')

                html_parts.append('</div>')

        # Added parameters (only in model 2)
        if only_in_2:
            if differences or only_in_1:
                html_parts.append('<hr class="section-divider">')

            for param in only_in_2:
                diff_num += 1
                formatted_param = format_param_name(param)
                # Extract just the parameter name (after the /)
                param_only = param.split('.')[-1] if '.' in param else param
                html_parts.append('<div class="diff-file">')
                html_parts.append(f'<div class="diff-file-header">{html.escape(formatted_param)}</div>')

                html_parts.append('<div class="diff-line diff-line-added">')
                html_parts.append(f'<div class="diff-line-num">{html.escape(model2_name)}</div>')
                html_parts.append(f'<div class="diff-line-content">{html.escape(param_only)} = {html.escape(format_value(flat2[param]))}</div>')
                html_parts.append('</div>')

                html_parts.append('</div>')

    html_parts.append('</div>')  # Close diff-container

    return '\n'.join(html_parts)


def main():
    parser = argparse.ArgumentParser(
        description='Compare namelist.trc.sms files between two models with modern diff UI'
    )
    parser.add_argument('model1_path', type=str, help='Path to first model directory')
    parser.add_argument('model2_path', type=str, help='Path to second model directory')
    parser.add_argument('model1_name', type=str, help='Name of first model')
    parser.add_argument('model2_name', type=str, help='Name of second model')
    parser.add_argument('--output', '-o', type=str, default='namelist_comparison.html',
                       help='Output HTML file (default: namelist_comparison.html)')
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
        html_output = '<div style="font-family: sans-serif; padding: 20px;">'
        html_output += '<div style="background: #fff3cd; border-left: 4px solid #ffc107; padding: 16px; border-radius: 4px;">'
        html_output += '<strong>⚠ Warning:</strong> Could not find <code>' + html.escape(args.namelist_name) + '</code> for:'
        html_output += '<ul style="margin: 8px 0;">'
        for item in missing:
            html_output += f'<li><code>{html.escape(item)}</code></li>'
        html_output += '</ul>'
        html_output += '</div></div>'

        with open(args.output, 'w') as f:
            f.write(html_output)

        print(f"Warning: Missing namelist files. Wrote warning to {args.output}", file=sys.stderr)
        sys.exit(0)

    # Load namelists
    nml1 = load_namelist(path1)
    nml2 = load_namelist(path2)

    # Compare
    differences, only_in_1, only_in_2 = compare_namelists(nml1, nml2)

    # Generate modern diff HTML
    html_output = generate_modern_diff_html(args.model1_name, args.model2_name,
                                            differences, only_in_1, only_in_2,
                                            nml1, nml2)

    # Write output
    with open(args.output, 'w') as f:
        f.write(html_output)

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
