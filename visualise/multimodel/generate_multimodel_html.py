#!/usr/bin/env python3
"""
Generate multimodel comparison HTML using Jinja2 templates.

This script replaces the shell-based template_N.html and benchmark_N.html
system with a modern Python approach using a single consolidated template.
"""

import sys
import os
import csv
from pathlib import Path
from jinja2 import Environment, FileSystemLoader, TemplateNotFound


def read_models_csv(csv_file='modelsToPlot.csv'):
    """
    Read model information from CSV file.

    Expected CSV format: model_id,description,start_year,to_year,location

    Args:
        csv_file: Path to the CSV file containing model information

    Returns:
        list: List of dictionaries containing model info

    Raises:
        FileNotFoundError: If CSV file doesn't exist
        ValueError: If CSV format is invalid
    """
    csv_path = Path(csv_file)

    if not csv_path.exists():
        raise FileNotFoundError(f"Models CSV file '{csv_file}' not found")

    models = []

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Replace underscores with spaces in description
            description = row.get('description', '').replace('_', ' ')

            models.append({
                'id': row.get('model_id', ''),
                'description': description,
                'year': row.get('to_year', '')  # Using to_year as the display year
            })

    if len(models) < 2:
        raise ValueError("At least 2 models required for comparison")

    if len(models) > 8:
        raise ValueError("Maximum of 8 models allowed for comparison")

    return models


def generate_multimodel_html(output_name, include_benchmarking=False, template_name='multimodel.html'):
    """
    Generate HTML file from template and model data.

    Args:
        output_name: Base name for the output HTML file
        include_benchmarking: Whether to include the benchmarking tab
        template_name: Name of the Jinja2 template to use

    Returns:
        Path: Path to the generated HTML file
    """
    # Read model information from CSV
    try:
        models = read_models_csv()
    except Exception as e:
        print(f"Error reading models CSV: {e}", file=sys.stderr)
        sys.exit(1)

    num_models = len(models)
    print(f"Generating HTML for {num_models} models")

    # Set up template environment
    # First try: templates/ in current directory
    template_dir = Path('templates')

    # Second try: templates/ relative to this script's location
    if not template_dir.exists():
        template_dir = Path(__file__).parent / 'templates'

    # Third try: absolute path to multimodel location
    if not template_dir.exists():
        template_dir = Path('/gpfs/data/greenocean/software/source/multimodel/templates')

    if not template_dir.exists():
        print(f"Error: Templates directory not found. Searched:", file=sys.stderr)
        print(f"  - ./templates/", file=sys.stderr)
        print(f"  - {Path(__file__).parent / 'templates'}", file=sys.stderr)
        print(f"  - /gpfs/data/greenocean/software/source/multimodel/templates", file=sys.stderr)
        sys.exit(1)

    try:
        env = Environment(
            loader=FileSystemLoader(str(template_dir)),
            autoescape=True,
            trim_blocks=True,
            lstrip_blocks=True
        )
        template = env.get_template(template_name)
    except TemplateNotFound:
        print(f"Error: Template '{template_name}' not found in {template_dir}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error loading template: {e}", file=sys.stderr)
        sys.exit(1)

    # Render the template
    try:
        html_content = template.render(
            models=models,
            include_benchmarking=include_benchmarking,
            num_models=num_models
        )
    except Exception as e:
        print(f"Error rendering template: {e}", file=sys.stderr)
        sys.exit(1)

    # Write HTML file
    output_file = Path(f'{output_name}.html')

    try:
        with open(output_file, 'w') as f:
            f.write(html_content)
    except Exception as e:
        print(f"Error writing HTML file: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Successfully generated: {output_file}")
    return output_file


def main():
    """Main entry point for the script."""
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("Usage: generate_multimodel_html.py <output_name> [benchmarking_flag]", file=sys.stderr)
        print("  output_name: Name for the output HTML file (without .html extension)", file=sys.stderr)
        print("  benchmarking_flag: 0 or 1 to include benchmarking tab (default: 0)", file=sys.stderr)
        sys.exit(1)

    output_name = sys.argv[1]
    include_benchmarking = False

    if len(sys.argv) == 3:
        try:
            include_benchmarking = int(sys.argv[2]) != 0
        except ValueError:
            print(f"Error: benchmarking_flag must be 0 or 1, got '{sys.argv[2]}'", file=sys.stderr)
            sys.exit(1)

    generate_multimodel_html(output_name, include_benchmarking)


if __name__ == '__main__':
    main()
