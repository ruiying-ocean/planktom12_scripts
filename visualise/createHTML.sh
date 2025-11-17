#!/bin/sh

# Usage: ./createHTML.sh <model_id> [base_dir]
# Example: ./createHTML.sh TOM12_RY_SPE2
#          ./createHTML.sh TOM12_RY_SPE2 ~/scratch/ModelRuns

# Check for required argument
if [ $# -lt 1 ]; then
    echo "Usage: $0 <model_id> [base_dir]"
    echo "  model_id: Model run identifier"
    echo "  base_dir: Base directory for model output (default: ~/scratch/ModelRuns)"
    echo ""
    echo "Example: $0 TOM12_RY_SPE2"
    exit 1
fi

model_id=$1
modelOutputDir=${2:-~/scratch/ModelRuns}

# Expand tilde in modelOutputDir
modelOutputDir="${modelOutputDir/#\~/$HOME}"

# Set up save directory
# Note: visualise.py saves to monitor/ not visualise/
saveDir="${modelOutputDir}/monitor/${model_id}/"
mkdir -p "${saveDir}"

# Get the directory where the script is located (visualise directory)
scriptDir="$(cd "$(dirname "$0")" && pwd)"

# Read image format from config file
img_format=$(python3 -c "
try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib
with open('${scriptDir}/visualise_config.toml', 'rb') as f:
    config = tomllib.load(f)
    print(config['figure']['format'])
" 2>/dev/null || echo "png")

echo "Using image format: $img_format"

# Find the latest year from existing map files
latest_year=$(find "${saveDir}" -name "${model_id}_[0-9]*_diagnostics.${img_format}" 2>/dev/null | \
              sed -n "s/.*${model_id}_\([0-9]\+\)_diagnostics\.${img_format}/\1/p" | \
              sort -n | tail -1)

# If no year found, try to find from any breakdown files
if [ -z "$latest_year" ]; then
    if [ -f "${modelOutputDir}/${model_id}/breakdown.sur.annual.csv" ]; then
        latest_year=$(tail -1 "${modelOutputDir}/${model_id}/breakdown.sur.annual.csv" | cut -d',' -f1)
    fi
fi

# If still no year found, use current year as fallback
if [ -z "$latest_year" ]; then
    latest_year=$(date +%Y)
    echo "Warning: Could not determine model year, using current year: $latest_year"
else
    echo "Using time slice year: $latest_year"
fi

# Copy template and custom.scss to save directory
cp "${scriptDir}/template.qmd" "${saveDir}/temp_template.qmd"
cp "${scriptDir}/custom.scss" "${saveDir}/"

# Substitute variables in the template
sed -e "s/IDENTIFIER_PLACEHOLDER/${model_id}/g" \
    -e "s/\${identifier}/${model_id}/g" \
    -e "s/\${start_year}/${start}/g" \
    -e "s/\${end_year}/${latest_year}/g" \
    -e "s/\${img_format}/${img_format}/g" \
    "${saveDir}/temp_template.qmd" > "${saveDir}/${model_id}.qmd"

# Change to save directory and render there
cd "${saveDir}"

# Render Quarto document
quarto render "${model_id}.qmd" --output "${model_id}.html"

# Clean up template files
rm temp_template.qmd "${model_id}.qmd" custom.scss

echo "✓ HTML report generated: ${saveDir}${model_id}.html"
