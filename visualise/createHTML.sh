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

# Copy template and custom.css to save directory
cp "${scriptDir}/template.qmd" "${saveDir}/temp_template.qmd"
cp "${scriptDir}/custom.css" "${saveDir}/"

# Substitute variables in the template
sed -e "s/\${identifier}/${run}/g" \
    -e "s/\${version}/${version}/g" \
    -e "s/\${date}/${date}/g" \
    -e "s/\${start_year}/${start}/g" \
    -e "s/\${end_year}/${end}/g" \
    -e "s/\${co2}/${co2}/g" \
    -e "s/\${forcing}/${forcing}/g" \
    -e "s/\${type}/${type}/g" \
    -e "s/\${temperature_restoring}/${temperature}/g" \
    -e "s/\${salinity_restoring}/${salinity}/g" \
    "${saveDir}/temp_template.qmd" > "${saveDir}/${run}.qmd"

# Change to save directory and render there
cd "${saveDir}"

# Render Quarto document
quarto render "${run}.qmd" --output "${run}.html"

# Clean up template files
rm temp_template.qmd "${run}.qmd" custom.css

echo "✓ HTML report generated: ${saveDir}${run}.html"
