#!/bin/sh

# Read in parameters for creating html file
read -r -a parms < html_parms

run=${parms[0]}
version=${parms[1]}
date=${parms[2]}
start=${parms[3]}
end=${parms[4]}
co2=${parms[5]^}
forcing=${parms[6]}
type=${parms[7]^}

if [[ ${parms[8]} -eq 0 ]]; then
	temperature="No"
else
	temperature="Yes"
fi

if [[ ${parms[9]} -eq 0 ]]; then
	salinity="No"
else
	salinity="Yes"
fi

# Set up save directory
# Note: visualise.py saves to monitor/ not visualise/
modelOutputDir=$1
saveDir="${modelOutputDir}/monitor/${run}/"
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
latest_year=$(find "${saveDir}" -name "${run}_[0-9]*_diagnostics.png" 2>/dev/null | \
              sed -n "s/.*${run}_\([0-9]\+\)_diagnostics.png/\1/p" | \
              sort -n | tail -1)

# If no year found, use end year from html_parms
if [ -z "$latest_year" ]; then
    latest_year=$end
fi

echo "Using time slice year: $latest_year"

# Copy template and custom.scss to save directory
cp "${scriptDir}/template.qmd" "${saveDir}/temp_template.qmd"
cp "${scriptDir}/custom.scss" "${saveDir}/"

# Substitute variables in the template
sed -e "s/IDENTIFIER_PLACEHOLDER/${run}/g" \
    -e "s/\${identifier}/${run}/g" \
    -e "s/\${start_year}/${start}/g" \
    -e "s/\${end_year}/${latest_year}/g" \
    -e "s/\${img_format}/${img_format}/g" \
    "${saveDir}/temp_template.qmd" > "${saveDir}/${run}.qmd"

# Change to save directory and render there
cd "${saveDir}"

# Render Quarto document
quarto render "${run}.qmd" --output "${run}.html"

# Clean up template files
rm temp_template.qmd "${run}.qmd" custom.scss

echo "✓ HTML report generated: ${saveDir}${run}.html"
