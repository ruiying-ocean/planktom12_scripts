#!/bin/bash

# Usage: ./setUpRestarts.sh <MODEL_ID>
# Example: ./setUpRestarts.sh TOM12_RY_ABCD

# Check if a model ID was provided
if [ -z "$1" ]; then
    echo "Error: No model ID provided."
    echo "Usage: $0 <MODEL_ID>"
    exit 1
fi

# Assign input parameters
MODEL_ID=$1
MODEL_RUN_DIR=$HOME/scratch/ModelRuns
SPIN_DIR=${MODEL_RUN_DIR}/TOM12_RY_SPJ2

STEPS_PER_YEAR=5475
FIRST_YEAR_SPINUP=1750
FIRST_YEAR_TRANSIENT=1940 # should be identical to the new start year

TIMESTEP=$(printf "%08d" $((($FIRST_YEAR_TRANSIENT - $FIRST_YEAR_SPINUP) * $STEPS_PER_YEAR)))

echo "============================================"
echo "Calculated TIMESTEP: $TIMESTEP"
echo "============================================"

# Command
bash /gpfs/home/vhf24tbu/setUpRuns/HALI-DEV/setup_restarts_RY.sh $SPIN_DIR $TIMESTEP ${MODEL_RUN_DIR}/${MODEL_ID}

echo "============================================"
echo "Setup complete for model ID: $MODEL_ID"
echo "============================================"

# bash /gpfs/home/vhf24tbu/scratch/setUpRuns/rm_old_restart.sh ${MODEL_ID}

rm -f /gpfs/home/vhf24tbu/scratch/ModelRuns/${MODEL_ID}/restart_ice_in.nc
rm -f /gpfs/home/vhf24tbu/scratch/ModelRuns/${MODEL_ID}/restart_trc.nc
rm -f /gpfs/home/vhf24tbu/scratch/ModelRuns/${MODEL_ID}/restart.nc

echo "============================================"
echo "Old restart cleaned for model ID: $MODEL_ID"
echo "============================================"

## copy EMP to new run directory
mv ${MODEL_RUN_DIR}/${MODEL_ID}/EMPave_$((FIRST_YEAR_TRANSIENT - 1)).dat ${MODEL_RUN_DIR}/${MODEL_ID}/EMPave.old
cp ${SPIN_DIR}/EMPave_$((FIRST_YEAR_TRANSIENT - 1)).dat ${MODEL_RUN_DIR}/${MODEL_ID}/EMPave_$((FIRST_YEAR_TRANSIENT - 1)).dat

echo "============================================"
echo "EMP files in ${MODEL} copied for model ID: $MODEL_ID"
echo "============================================"

## copy namelist.trc.sms to make sure both are consistent
mv ${MODEL_RUN_DIR}/${MODEL_ID}/namelist.trc.sms ${MODEL_RUN_DIR}/${MODEL_ID}/namelist.trc.sms.old
cp ${SPIN_DIR}/namelist.trc.sms ${MODEL_RUN_DIR}/${MODEL_ID}/namelist.trc.sms

echo "============================================"
echo "Namelist copied for model ID: $MODEL_ID"
echo "============================================"
