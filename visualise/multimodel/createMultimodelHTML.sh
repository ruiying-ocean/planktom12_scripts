#!/bin/sh

# Script to generate Quarto-based HTML for multimodel comparison
# Usage: ./createMultimodelHTML.sh <timestamp> [benchmark_flag]
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
scriptDir="$(cd "$(dirname "$0")" && pwd)"

# Check if modelsToPlot.csv exists
if [ ! -f "modelsToPlot.csv" ]; then
    echo "Error: modelsToPlot.csv not found in current directory"
    exit 1
fi

# Read model information from CSV
runs=( $( cut -f 1 -d , modelsToPlot.csv | tail -n +2 ) )
desc=( $( cut -f 2 -d , modelsToPlot.csv | tail -n +2 ) )
to=( $( cut -f 5 -d , modelsToPlot.csv | tail -n +2 ) )

length=${#runs[@]}

echo "Creating Quarto HTML for $length models..."

# Copy template and stylesheet to current directory
cp "${scriptDir}/template_multimodel.qmd" ./temp_template.qmd
cp "${scriptDir}/custom.scss" ./

# Build the model maps section as a grid/table
# Format: Rows = variables, Columns = models + anomaly (if 2 models)
model_maps_section=""

# Check if we have difference maps (only for 2 models)
has_diff_maps=0
if [ $length -eq 2 ] && [ -f "difference_${to[0]}_diagnostics.png" ]; then
    has_diff_maps=1
fi

# Define the map types (rows)
map_types=("diagnostics" "phytos" "zoos" "nutrients")
map_labels=("Ecosystem Diagnostics" "Phytoplankton" "Zooplankton" "Nutrients")

# Build markdown table for each map type
for idx in ${!map_types[@]}; do
    map_type=${map_types[$idx]}
    map_label=${map_labels[$idx]}

    model_maps_section="${model_maps_section}## ${map_label}\n\n"
    model_maps_section="${model_maps_section}::: {.grid}\n\n"

    # Add each model's map
    for i in ${!runs[@]}; do
        run=${runs[$i]}
        year=${to[$i]}
        display_name=${desc[$i]//_/ }

        map_file="${run}_${year}_${map_type}.png"

        if [ -f "$map_file" ]; then
            model_maps_section="${model_maps_section}::: {.g-col-6}\n"
            model_maps_section="${model_maps_section}### ${display_name}\n\n"
            model_maps_section="${model_maps_section}![](${map_file})\n"
            model_maps_section="${model_maps_section}:::\n\n"
        fi
    done

    # Add difference map if it exists (for 2-model comparison)
    if [ $has_diff_maps -eq 1 ]; then
        diff_file="difference_${to[0]}_${map_type}.png"
        if [ -f "$diff_file" ]; then
            model_maps_section="${model_maps_section}::: {.g-col-6}\n"
            model_maps_section="${model_maps_section}### Anomaly (${desc[0]} - ${desc[1]})\n\n"
            model_maps_section="${model_maps_section}![](${diff_file})\n"
            model_maps_section="${model_maps_section}:::\n\n"
        fi
    fi

    model_maps_section="${model_maps_section}:::\n\n"
    model_maps_section="${model_maps_section}---\n\n"
done

# Substitute variables in the template
# Use printf to handle newlines properly
printf "%s" "$model_maps_section" > temp_model_maps.txt

# Use sed to substitute the timestamp and model maps section
sed -e "s/\${timestamp}/${timestamp}/g" \
    temp_template.qmd > temp_with_timestamp.qmd

# Replace the MODEL_MAPS placeholder with the actual content
# Using a multi-line approach with awk for better handling
awk '
/\$\{MODEL_MAPS\}/ {
    while ((getline line < "temp_model_maps.txt") > 0) {
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
rm temp_template.qmd temp_with_timestamp.qmd temp_model_maps.txt multimodel.qmd custom.scss

if [ -f "multimodel.html" ]; then
    echo "✓ Multi-model HTML report generated: $(pwd)/multimodel.html"
else
    echo "✗ Error: Failed to generate multimodel.html"
    exit 1
fi
