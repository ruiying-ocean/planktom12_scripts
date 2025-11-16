#!/bin/sh

# Load Ferret for map generation
module add ferret/7.6.0
if [ -n "$FER_PATH" ]; then
    source $FER_PATH
fi

# Use miniforge Python instead of module python/3.8
export PATH=~/miniforge3/bin:$PATH


# Source central directory where the files and scripts are located
# Resolve symlinks to find the actual script location
scriptPath="$0"
if [ -L "$scriptPath" ]; then
    scriptPath="$(readlink -f "$scriptPath" 2>/dev/null || readlink "$scriptPath")"
fi
scriptDir="$(cd "$(dirname "$scriptPath")" && pwd)"
srcDir="${scriptDir}/"

# Verify we found multimodel.py
if [ ! -f "${srcDir}/multimodel.py" ]; then
    echo "Error: Cannot find multimodel.py at ${srcDir}"
    exit 1
fi

# Check if modelsToPlot.csv exists
if [ ! -f "modelsToPlot.csv" ]; then
    echo "Error: modelsToPlot.csv not found in current directory"
    exit 1
fi

# Read in runs to compare from csv files
runs=( $( cut -f 1 -d , modelsToPlot.csv | tail -n +2 ) )
desc=( $( cut -f 2 -d , modelsToPlot.csv | tail -n +2 ) )
strt=( $( cut -f 3 -d , modelsToPlot.csv | tail -n +2 ) )
from=( $( cut -f 4 -d , modelsToPlot.csv | tail -n +2 ) )
  to=( $( cut -f 5 -d , modelsToPlot.csv | tail -n +2 ) )
 end=( $( cut -f 6 -d , modelsToPlot.csv | tail -n +2 ) )
locs=( $( cut -f 7 -d , modelsToPlot.csv | tail -n +2 ) )

# Create folder name from model runs (e.g., JRA3-JRA1)
# Strip TOM12_RY_ prefix from each run name
cleanRuns=()
for run in "${runs[@]}"; do
    cleanRuns+=("${run#TOM12_RY_}")
done
folderName=$(IFS=- ; echo "${cleanRuns[*]}")

# Create directory to save files to
curDir=${PWD}
saveDir=${curDir}/${folderName}/

mkdir -p ${saveDir}
cp modelsToPlot.csv ${saveDir}
cd ${saveDir}

length=${#runs[@]}
if [ $length -gt 8 ]; then
	echo "Trying to plot too many runs, maximum is 8"
	exit 2
fi

# Copy in the required files
cp ${srcDir}/multimodel.py .

# Copy Python map generation scripts from visualise directory
# scriptDir is visualise/multimodel, so parent is visualise
visualiseDir="$(cd "$(dirname "${scriptDir}")" && pwd)/"

if [ ! -f "${visualiseDir}/make_maps.py" ]; then
    echo "Error: Cannot find make_maps.py at ${visualiseDir}"
    exit 1
fi
cp ${visualiseDir}/make_maps.py .
cp ${visualiseDir}/map_utils.py .

# Run the python script to generate the time series plots (with debug to see warnings)
./multimodel.py ${saveDir} --debug

# Generate spatial comparison maps (grid format with all models)
echo "Generating spatial comparison maps..."
python3 ${srcDir}/multimodel_maps.py modelsToPlot.csv ${saveDir}

# Copy observation images to save directory
cp ${srcDir}/*.png .

# Generate HTML using Quarto (replaces Python + Jinja2 system)
${srcDir}/createMultimodelHTML.sh "${folderName}" 0

echo "Finished: saved to" ${saveDir}
