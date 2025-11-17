#!/bin/sh

# Script to generate Quarto-based HTML for multimodel comparison
# Usage: ./make_multimodel_html.sh <timestamp> [benchmark_flag]
#
# Arguments:
#   timestamp: Directory timestamp (e.g., 151124-143022)
#   benchmark_flag: Optional, 0 or 1 (default 0)

if [ $# -lt 1 ]; then
    echo "Usage: $0 <timestamp> [benchmark_flag]"
    exit 1
fi

timestamp=$1
benchmark=${2:-0}

# Get the directory where this script is located
script_dir="$(cd "$(dirname "$0")" && pwd)"

# Check if modelsToPlot.csv exists
if [ ! -f "modelsToPlot.csv" ]; then
    echo "Error: modelsToPlot.csv not found in current directory"
    exit 1
fi

# Read model information from CSV (model_id, description, start_year, to_year, location)
runs=( $( cut -f 1 -d , modelsToPlot.csv | tail -n +2 ) )
desc=( $( cut -f 2 -d , modelsToPlot.csv | tail -n +2 ) )
to=( $( cut -f 4 -d , modelsToPlot.csv | tail -n +2 ) )

length=${#runs[@]}

echo "Creating Quarto HTML for $length models..."

# Copy template and stylesheet to current directory
cp "${script_dir}/template_multimodel.qmd" ./temp_template.qmd
cp "${script_dir}/custom.scss" ./

# Build the spatial maps section
# The multimodel_maps.py script generates grid-format PNG files
model_maps_section=""

# Get image format from first model's config or default to png
img_format="png"
if [ -f "../visualise_config.toml" ]; then
    img_format=$(python3 -c "
try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib
try:
    with open('../visualise_config.toml', 'rb') as f:
        config = tomllib.load(f)
        print(config.get('figure', {}).get('format', 'png'))
except:
    print('png')
" 2>/dev/null || echo "png")
fi

# Add ecosystem maps
if [ -f "multimodel_spatial_ecosystem.${img_format}" ]; then
    model_maps_section="${model_maps_section}## Ecosystem Variables\n\n"
    model_maps_section="${model_maps_section}![](multimodel_spatial_ecosystem.${img_format})\n\n"
    model_maps_section="${model_maps_section}---\n\n"
fi

# Add phytoplankton maps
if [ -f "multimodel_spatial_phytoplankton.${img_format}" ]; then
    model_maps_section="${model_maps_section}## Phytoplankton\n\n"
    model_maps_section="${model_maps_section}![](multimodel_spatial_phytoplankton.${img_format})\n\n"
    model_maps_section="${model_maps_section}---\n\n"
fi

# Add zooplankton maps
if [ -f "multimodel_spatial_zooplankton.${img_format}" ]; then
    model_maps_section="${model_maps_section}## Zooplankton\n\n"
    model_maps_section="${model_maps_section}![](multimodel_spatial_zooplankton.${img_format})\n\n"
    model_maps_section="${model_maps_section}---\n\n"
fi

# Add nutrient maps
if [ -f "multimodel_spatial_nutrients.${img_format}" ]; then
    model_maps_section="${model_maps_section}## Nutrients\n\n"
    model_maps_section="${model_maps_section}![](multimodel_spatial_nutrients.${img_format})\n\n"
    model_maps_section="${model_maps_section}---\n\n"
fi

# Add derived variable maps
if [ -f "multimodel_spatial_derived.${img_format}" ]; then
    model_maps_section="${model_maps_section}## Derived Ecosystem Variables\n\n"
    model_maps_section="${model_maps_section}![](multimodel_spatial_derived.${img_format})\n\n"
    model_maps_section="${model_maps_section}---\n\n"
fi

# Build the derived variables section
derived_section=""

# Check for difference maps (only for 2-model comparisons)
if [ $length -eq 2 ]; then
    # Get year from one of the models
    year="${to[0]}"

    # Check for derived variable difference map
    if [ -f "difference_${year}_derived.png" ]; then
        derived_section="${derived_section}## Model Differences (${runs[0]} - ${runs[1]})\n\n"
        derived_section="${derived_section}![](difference_${year}_derived.png)\n\n"
    fi
fi

# If no difference maps, show a placeholder or note
if [ -z "$derived_section" ]; then
    derived_section="Time series plots available above showing SP, recycle, e-ratio, and Teff.\n\n"
fi

# Substitute variables in the template
# Use printf with -e to interpret escape sequences properly
printf "%b" "$model_maps_section" > temp_model_maps.txt
printf "%b" "$derived_section" > temp_derived_section.txt

# Use sed to substitute the timestamp and model maps section
sed -e "s/\${timestamp}/${timestamp}/g" \
    temp_template.qmd > temp_with_timestamp.qmd

# Replace the MODEL_MAPS and DERIVED_SECTION placeholders with the actual content
# Using a multi-line approach with awk for better handling
awk '
/\$\{MODEL_MAPS\}/ {
    while ((getline line < "temp_model_maps.txt") > 0) {
        print line
    }
    next
}
/\$\{DERIVED_SECTION\}/ {
    while ((getline line < "temp_derived_section.txt") > 0) {
        print line
    }
    next
}
{print}
' temp_with_timestamp.qmd > multimodel.qmd

# Render Quarto document
echo "Rendering Quarto document..."
quarto render multimodel.qmd --output multimodel.html

# Clean up temporary files
rm temp_template.qmd temp_with_timestamp.qmd temp_model_maps.txt temp_derived_section.txt multimodel.qmd custom.scss

if [ -f "multimodel.html" ]; then
    echo "✓ Multi-model HTML report generated: $(pwd)/multimodel.html"
else
    echo "✗ Error: Failed to generate multimodel.html"
    exit 1
fi
