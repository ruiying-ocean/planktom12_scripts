#!/bin/bash

# Usage: ./setup_spin.sh <MODEL_ID> <SPINUP_MODEL_ID>
# Example: ./setup_spin.sh TOM12_RY_ABCD TOM12_RY_SPJ2

# Check if required arguments were provided
if [ -z "$1" ]; then
    echo "Error: No model ID provided."
    echo "Usage: $0 <MODEL_ID> <SPINUP_MODEL_ID>"
    exit 1
fi

if [ -z "$2" ]; then
    echo "Error: No spinup model ID provided."
    echo "Usage: $0 <MODEL_ID> <SPINUP_MODEL_ID>"
    exit 1
fi

# Assign input parameters
MODEL_ID=$1
SPINUP_MODEL_ID=$2
MODEL_RUN_DIR=$HOME/scratch/ModelRuns
SPIN_DIR=${MODEL_RUN_DIR}/${SPINUP_MODEL_ID}

# Validate paths exist
if [ ! -d "$MODEL_RUN_DIR" ]; then
    echo "Error: Model run directory does not exist: $MODEL_RUN_DIR"
    exit 1
fi

if [ ! -d "$SPIN_DIR" ]; then
    echo "Error: Spinup model directory does not exist: $SPIN_DIR"
    echo "Check that the spinup model ID '$SPINUP_MODEL_ID' is correct"
    exit 1
fi

if [ ! -d "${MODEL_RUN_DIR}/${MODEL_ID}" ]; then
    echo "Error: Model directory does not exist: ${MODEL_RUN_DIR}/${MODEL_ID}"
    exit 1
fi

STEPS_PER_YEAR=5475
FIRST_YEAR_SPINUP=1750

# Parse FIRST_YEAR_TRANSIENT from the model run directory's setUpData file
TRANSIENT_SETUP_DATA=$(find "${MODEL_RUN_DIR}/${MODEL_ID}" -maxdepth 1 -name "setUpData*.dat" | head -1)
if [ -z "$TRANSIENT_SETUP_DATA" ] || [ ! -f "$TRANSIENT_SETUP_DATA" ]; then
    echo "Error: Could not find setUpData*.dat file in ${MODEL_RUN_DIR}/${MODEL_ID}"
    exit 1
fi

FIRST_YEAR_TRANSIENT=$(grep "^yearStart:" "$TRANSIENT_SETUP_DATA" | cut -d':' -f2)
if [ -z "$FIRST_YEAR_TRANSIENT" ]; then
    echo "Error: Could not parse yearStart from $TRANSIENT_SETUP_DATA"
    exit 1
fi

# Parse forcing type from setup data file (for logging only)
FORCING=$(grep "^forcing:" "$TRANSIENT_SETUP_DATA" | cut -d':' -f2)
if [ -z "$FORCING" ]; then
    FORCING="JRA"
fi

# Parse forcing mode from setup data file
FORCING_MODE=$(grep "^forcing_mode:" "$TRANSIENT_SETUP_DATA" | cut -d':' -f2)
if [ -z "$FORCING_MODE" ]; then
    echo "Warning: Could not parse forcing_mode from $TRANSIENT_SETUP_DATA, defaulting to spinup"
    FORCING_MODE="spinup"
fi

TIMESTEP=$(printf "%08d" $((($FIRST_YEAR_TRANSIENT - $FIRST_YEAR_SPINUP) * $STEPS_PER_YEAR)))

echo "============================================"
echo "Using setUpData file: $TRANSIENT_SETUP_DATA"
echo "First year (transient): $FIRST_YEAR_TRANSIENT"
echo "Forcing: $FORCING"
echo "Forcing mode: $FORCING_MODE"
echo "Calculated TIMESTEP: $TIMESTEP"
echo "============================================"

# Command
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
bash ${SCRIPT_DIR}/setup_restarts_RY.sh $SPIN_DIR $TIMESTEP ${MODEL_RUN_DIR}/${MODEL_ID}
if [ $? -ne 0 ]; then
    echo "Error: Setup restarts script failed"
    exit 1
fi

echo "============================================"
echo "Setup complete for model ID: $MODEL_ID"
echo "============================================"

rm -f ${MODEL_RUN_DIR}/${MODEL_ID}/restart_ice_in.nc
rm -f ${MODEL_RUN_DIR}/${MODEL_ID}/restart_trc.nc
rm -f ${MODEL_RUN_DIR}/${MODEL_ID}/restart.nc

echo "============================================"
echo "Old restart cleaned for model ID: $MODEL_ID"
echo "============================================"

# Update namelist_ref to use other_years (which already points to cycling for BIAS, restart for DYNAMIC)
rm -f ${MODEL_RUN_DIR}/${MODEL_ID}/namelist_ref
ln -s namelist_ref_other_years ${MODEL_RUN_DIR}/${MODEL_ID}/namelist_ref
echo "============================================"
echo "Updated namelist_ref -> namelist_ref_other_years"
echo "============================================"

## copy EMP to new run directory
EMP_SOURCE="${SPIN_DIR}/EMPave_$((FIRST_YEAR_TRANSIENT - 1)).dat"
EMP_TARGET="${MODEL_RUN_DIR}/${MODEL_ID}/EMPave_$((FIRST_YEAR_TRANSIENT - 1)).dat"
EMP_OLD="${MODEL_RUN_DIR}/${MODEL_ID}/EMPave.old"

if [ ! -f "$EMP_SOURCE" ]; then
    echo "Error: Source EMP file does not exist: $EMP_SOURCE"
    exit 1
fi

if [ -f "$EMP_TARGET" ]; then
    mv "$EMP_TARGET" "$EMP_OLD"
fi

cp "$EMP_SOURCE" "$EMP_TARGET"
if [ $? -ne 0 ]; then
    echo "Error: Failed to copy EMP file"
    exit 1
fi

echo "============================================"
echo "EMP files in ${MODEL} copied for model ID: $MODEL_ID"
echo "============================================"

## copy namelist.trc.sms to make sure both are consistent - skip for spinup runs
if [ "$FORCING_MODE" == "spinup" ]; then
    echo "============================================"
    echo "Spinup run: keeping existing namelist.trc.sms"
    echo "============================================"
else
    NAMELIST_SOURCE="${SPIN_DIR}/namelist.trc.sms"
    NAMELIST_TARGET="${MODEL_RUN_DIR}/${MODEL_ID}/namelist.trc.sms"
    NAMELIST_OLD="${MODEL_RUN_DIR}/${MODEL_ID}/namelist.trc.sms.old"

    if [ ! -f "$NAMELIST_SOURCE" ]; then
        echo "Error: Source namelist file does not exist: $NAMELIST_SOURCE"
        exit 1
    fi

    if [ -f "$NAMELIST_TARGET" ]; then
        mv "$NAMELIST_TARGET" "$NAMELIST_OLD"
    fi

    cp "$NAMELIST_SOURCE" "$NAMELIST_TARGET"
    if [ $? -ne 0 ]; then
        echo "Error: Failed to copy namelist file"
        exit 1
    fi

    echo "============================================"
    echo "namelist.trc.sms copied for model ID: $MODEL_ID"
    echo "============================================"
fi

## Record spinup info
SPINUP_RECORD="${MODEL_RUN_DIR}/${MODEL_ID}/spinup_info.txt"
echo "Spinup Model ID: $SPINUP_MODEL_ID" > "$SPINUP_RECORD"
echo "Spinup Year: $((FIRST_YEAR_TRANSIENT - 1))" >> "$SPINUP_RECORD"
echo "Setup Date: $(date '+%Y-%m-%d %H:%M:%S')" >> "$SPINUP_RECORD"

echo "============================================"
echo "Spinup record saved to: $SPINUP_RECORD"
echo "============================================"
