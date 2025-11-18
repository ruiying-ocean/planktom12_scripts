#!/usr/bin/env python3
"""
Compare setUpData configuration files between two models and generate modern diff-style HTML report.

This script parses setUpData_*.dat files and creates a fancy side-by-side comparison
with syntax highlighting and modern UI similar to GitHub diffs.
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Tuple, List
import html
import re


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


def generate_modern_diff_html(model1_name: str, model2_name: str,
                               differences: List[Tuple[str, str, str]],
                               only_in_1: List[str], only_in_2: List[str],
                               config1: Dict[str, str], config2: Dict[str, str]) -> str:
    """
    Generate modern GitHub-style diff HTML report.

    Args:
        model1_name: Name of first model
        model2_name: Name of second model
        differences: List of differing parameters
        only_in_1: Parameters only in model 1
        only_in_2: Parameters only in model 2
        config1: First config (for looking up values)
        config2: Second config (for looking up values)

    Returns:
        HTML string with modern diff UI
    """
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
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 20px 24px;
    border-radius: 8px 8px 0 0;
    margin-bottom: 0;
}

.diff-header h3 {
    margin: 0 0 8px 0;
    font-size: 18px;
    font-weight: 600;
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
        html_parts.append('The setUpData configuration is identical between both models.')
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
                html_parts.append(f'<div class="diff-file-header">{html.escape(param)}</div>')

                html_parts.append('<div class="diff-split">')

                # Left side (model 1)
                html_parts.append('<div class="diff-split-side">')
                html_parts.append(f'<div class="diff-line diff-line-changed" style="width: 100%;">')
                html_parts.append(f'<div class="diff-line-num">{html.escape(model1_name)}</div>')
                html_parts.append(f'<div class="diff-line-content">{html.escape(val1)}</div>')
                html_parts.append('</div>')
                html_parts.append('</div>')

                # Right side (model 2)
                html_parts.append('<div class="diff-split-side">')
                html_parts.append(f'<div class="diff-line diff-line-changed" style="width: 100%;">')
                html_parts.append(f'<div class="diff-line-num">{html.escape(model2_name)}</div>')
                html_parts.append(f'<div class="diff-line-content">{html.escape(val2)}</div>')
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
                html_parts.append('<div class="diff-file">')
                html_parts.append(f'<div class="diff-file-header">{html.escape(param)}</div>')

                html_parts.append('<div class="diff-line diff-line-removed">')
                html_parts.append(f'<div class="diff-line-num">{html.escape(model1_name)}</div>')
                html_parts.append(f'<div class="diff-line-content">{html.escape(config1[param])}</div>')
                html_parts.append('</div>')

                html_parts.append('</div>')

        # Added parameters (only in model 2)
        if only_in_2:
            if differences or only_in_1:
                html_parts.append('<hr class="section-divider">')

            for param in only_in_2:
                diff_num += 1
                html_parts.append('<div class="diff-file">')
                html_parts.append(f'<div class="diff-file-header">{html.escape(param)}</div>')

                html_parts.append('<div class="diff-line diff-line-added">')
                html_parts.append(f'<div class="diff-line-num">{html.escape(model2_name)}</div>')
                html_parts.append(f'<div class="diff-line-content">{html.escape(config2[param])}</div>')
                html_parts.append('</div>')

                html_parts.append('</div>')

    html_parts.append('</div>')  # Close diff-container

    return '\n'.join(html_parts)


def main():
    parser = argparse.ArgumentParser(
        description='Compare setUpData configuration files between two models with modern diff UI'
    )
    parser.add_argument('model1_path', type=str, help='Path to first model directory')
    parser.add_argument('model2_path', type=str, help='Path to second model directory')
    parser.add_argument('model1_name', type=str, help='Name of first model')
    parser.add_argument('model2_name', type=str, help='Name of second model')
    parser.add_argument('--output', '-o', type=str, default='setupdata_comparison.html',
                       help='Output HTML file (default: setupdata_comparison.html)')

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
        html_output = '<div style="font-family: sans-serif; padding: 20px;">'
        html_output += '<div style="background: #fff3cd; border-left: 4px solid #ffc107; padding: 16px; border-radius: 4px;">'
        html_output += '<strong>⚠ Warning:</strong> Could not find setUpData configuration files:'
        html_output += '<ul style="margin: 8px 0;">'
        for item in missing:
            html_output += f'<li><code>{html.escape(item)}</code></li>'
        html_output += '</ul>'
        html_output += '</div></div>'

        with open(args.output, 'w') as f:
            f.write(html_output)

        print(f"Warning: Missing setUpData files. Wrote warning to {args.output}", file=sys.stderr)
        sys.exit(0)

    # Load configs
    config1 = load_setupdata(path1)
    config2 = load_setupdata(path2)

    # Compare
    differences, only_in_1, only_in_2 = compare_configs(config1, config2)

    # Generate modern diff HTML
    html_output = generate_modern_diff_html(args.model1_name, args.model2_name,
                                            differences, only_in_1, only_in_2,
                                            config1, config2)

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
