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

# Read in runs to compare from csv files (model_id, description, start_year, to_year, location)
runs=( $( cut -f 1 -d , modelsToPlot.csv | tail -n +2 ) )
desc=( $( cut -f 2 -d , modelsToPlot.csv | tail -n +2 ) )
  to=( $( cut -f 4 -d , modelsToPlot.csv | tail -n +2 ) )
locs=( $( cut -f 5 -d , modelsToPlot.csv | tail -n +2 ) )

# Create folder name from model runs (e.g., JRA3-JRA1)
# Strip TOM12_RY_ prefix from each run name
clean_runs=()
for run in "${runs[@]}"; do
    clean_runs+=("${run#TOM12_RY_}")
done
folder_name=$(IFS=- ; echo "${clean_runs[*]}")

# Create directory to save files to
cur_dir=${PWD}
save_dir=${cur_dir}/${folder_name}/

mkdir -p ${save_dir}
cp modelsToPlot.csv ${save_dir}
cd ${save_dir}

length=${#runs[@]}
if [ $length -gt 8 ]; then
	echo "Trying to plot too many runs, maximum is 8"
	exit 2
fi

# Copy in the required files
cp ${src_dir}/make_multimodel_timeseries.py .

# Copy Python map generation scripts from visualise directory
# script_dir is visualise/multimodel, so parent is visualise
visualise_dir="$(cd "$(dirname "${script_dir}")" && pwd)/"

if [ ! -f "${visualise_dir}/make_maps.py" ]; then
    echo "Error: Cannot find make_maps.py at ${visualise_dir}"
    exit 1
fi
cp ${visualise_dir}/make_maps.py .
cp ${visualise_dir}/map_utils.py .

# Export config path for Python scripts to find
if [ -f "${visualise_dir}/visualise_config.toml" ]; then
    export VISUALISE_CONFIG="${visualise_dir}/visualise_config.toml"
fi

# Run the python script to generate the time series plots (with debug to see warnings)
./make_multimodel_timeseries.py ${save_dir} --debug

# Generate spatial comparison maps (grid format with all models)
echo "Generating spatial comparison maps..."
python3 ${src_dir}/make_multimodel_maps.py modelsToPlot.csv ${save_dir}

# Copy observation images to save directory
cp ${src_dir}/*.png .

# Generate HTML using Quarto (replaces Python + Jinja2 system)
${src_dir}/make_multimodel_html.sh "${folder_name}" 0

echo "Finished: saved to" ${save_dir}
