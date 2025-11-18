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
script_path="$0"
if [ -L "$script_path" ]; then
    script_path="$(readlink -f "$script_path" 2>/dev/null || readlink "$script_path")"
fi
script_dir="$(cd "$(dirname "$script_path")" && pwd)"
src_dir="${script_dir}/"

# Verify we found make_multimodel_timeseries.py
if [ ! -f "${src_dir}/make_multimodel_timeseries.py" ]; then
    echo "Error: Cannot find make_multimodel_timeseries.py at ${src_dir}"
    exit 1
fi

# Check if modelsToPlot.csv exists
if [ ! -f "modelsToPlot.csv" ]; then
    echo "Error: modelsToPlot.csv not found in current directory"
    exit 1
fi

# Run the centralized multi-model visualization orchestrator
python3 ${src_dir}/visualise_multimodel.py modelsToPlot.csv --debug
