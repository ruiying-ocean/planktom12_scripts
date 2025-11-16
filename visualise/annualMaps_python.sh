#!/bin/bash
# Python-based replacement for annualMaps.sh
# Generates oceanographic maps without requiring Ferret

run=$1
year=$2
modelOutputDir=$3

# Save to monitor/ directory to match visualise.py output
monitor="monitor/"
saveDir="${modelOutputDir}${monitor}${run}/"

# Create save directory if it doesn't exist
mkdir -p "${saveDir}"

echo "=== Generating maps with Python ==="
echo "Run: $run"
echo "Year: $year"
echo "Model output directory: $modelOutputDir"
echo "Output directory: $saveDir"

# Run Python map generation with observations
# This replaces all the Ferret scripts (maps.jnl, mapsPFT.jnl, mapsDiff.jnl)
python3 make_maps.py "$run" "$year" "$year" \
    --basedir "$modelOutputDir" \
    --output-dir "$saveDir" \
    --obs-dir "/gpfs/home/vhf24tbu/Observations"

echo "Output saved to: ${saveDir}"
