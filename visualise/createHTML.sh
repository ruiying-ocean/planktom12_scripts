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
baseDir=$1
saveDir="${baseDir}/visualise/${run}/"
mkdir -p "${saveDir}"

# Render Quarto document with parameters
quarto render template.qmd \
  -P identifier:"${run}" \
  -P version:"${version}" \
  -P date:"${date}" \
  -P start_year:"${start}" \
  -P end_year:"${end}" \
  -P co2:"${co2}" \
  -P forcing:"${forcing}" \
  -P type:"${type}" \
  -P temperature_restoring:"${temperature}" \
  -P salinity_restoring:"${salinity}" \
  --output "${saveDir}${run}.html" \
  --execute-dir "${saveDir}"

echo "✓ HTML report generated: ${saveDir}${run}.html"
