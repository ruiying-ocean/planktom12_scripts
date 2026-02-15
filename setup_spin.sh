#!/bin/bash

# Usage: ./setup_spin.sh <MODEL_ID> <SPINUP_MODEL_ID>
# Example: ./setup_spin.sh TOM12_RY_ABCD TOM12_RY_SPJ2

# Colors
GREEN='\e[1;32m'
CYAN='\e[1;36m'
YELLOW='\e[1;33m'
RED='\e[1;31m'
RESET='\e[0m'

# Check if required arguments were provided
if [ -z "$1" ]; then
    echo -e "${RED}Error${RESET}: No model ID provided."
    echo "Usage: $0 <MODEL_ID> <SPINUP_MODEL_ID>"
    exit 1
fi

if [ -z "$2" ]; then
    echo -e "${RED}Error${RESET}: No spinup model ID provided."
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
    echo -e "${RED}Error${RESET}: Model run directory does not exist: $MODEL_RUN_DIR"
    exit 1
fi

if [ ! -d "$SPIN_DIR" ]; then
    echo -e "${RED}Error${RESET}: Spinup model directory does not exist: $SPIN_DIR"
    echo "Check that the spinup model ID '$SPINUP_MODEL_ID' is correct"
    exit 1
fi

if [ ! -d "${MODEL_RUN_DIR}/${MODEL_ID}" ]; then
    echo -e "${RED}Error${RESET}: Model directory does not exist: ${MODEL_RUN_DIR}/${MODEL_ID}"
    exit 1
fi

STEPS_PER_YEAR=5475
FIRST_YEAR_SPINUP=1750

# Parse FIRST_YEAR_TRANSIENT from the model run directory's setUpData file
TRANSIENT_SETUP_DATA=$(find "${MODEL_RUN_DIR}/${MODEL_ID}" -maxdepth 1 -name "setUpData*.dat" | head -1)
if [ -z "$TRANSIENT_SETUP_DATA" ] || [ ! -f "$TRANSIENT_SETUP_DATA" ]; then
    echo -e "${RED}Error${RESET}: Could not find setUpData*.dat file in ${MODEL_RUN_DIR}/${MODEL_ID}"
    exit 1
fi

FIRST_YEAR_TRANSIENT=$(grep "^yearStart:" "$TRANSIENT_SETUP_DATA" | cut -d':' -f2)
if [ -z "$FIRST_YEAR_TRANSIENT" ]; then
    echo -e "${RED}Error${RESET}: Could not parse yearStart from $TRANSIENT_SETUP_DATA"
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
    echo -e "${YELLOW}Warning${RESET}: Could not parse forcing_mode from $TRANSIENT_SETUP_DATA, defaulting to spinup"
    FORCING_MODE="spinup"
fi

# Parse redate_restart from setup data file
REDATE_RESTART=$(grep "^redate_restart:" "$TRANSIENT_SETUP_DATA" | cut -d':' -f2)

if [ "$REDATE_RESTART" == "true" ]; then
    # Find the latest restart timestep from the spinup directory
    LATEST_RESTART=$(ls "${SPIN_DIR}"/ORCA2_*_restart_0000.nc 2>/dev/null | sort | tail -1)
    if [ -z "$LATEST_RESTART" ]; then
        echo -e "${RED}Error${RESET}: No restart files found in ${SPIN_DIR}"
        exit 1
    fi
    TIMESTEP=$(basename "$LATEST_RESTART" | sed 's/ORCA2_\(.*\)_restart_0000\.nc/\1/')
else
    TIMESTEP=$(printf "%08d" $((($FIRST_YEAR_TRANSIENT - $FIRST_YEAR_SPINUP) * $STEPS_PER_YEAR)))
fi

echo -e "${CYAN}--- Configuration ---${RESET}"
echo "  setUpData file:   $TRANSIENT_SETUP_DATA"
echo "  First year:       $FIRST_YEAR_TRANSIENT"
echo "  Forcing:          $FORCING"
echo "  Forcing mode:     $FORCING_MODE"
echo "  Timestep:         $TIMESTEP"

# Command
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
bash ${SCRIPT_DIR}/setup_restarts_RY.sh $SPIN_DIR $TIMESTEP ${MODEL_RUN_DIR}/${MODEL_ID}
if [ $? -ne 0 ]; then
    echo -e "${RED}Error${RESET}: Setup restarts script failed"
    exit 1
fi

echo -e "${GREEN}[OK]${RESET} Restart links created for $MODEL_ID"

rm -f ${MODEL_RUN_DIR}/${MODEL_ID}/restart_ice_in.nc
rm -f ${MODEL_RUN_DIR}/${MODEL_ID}/restart_trc.nc
rm -f ${MODEL_RUN_DIR}/${MODEL_ID}/restart.nc

echo -e "${GREEN}[OK]${RESET} Old single-file restarts cleaned"

# Update namelist_ref to use other_years (which already points to cycling for BIAS, restart for DYNAMIC)
rm -f ${MODEL_RUN_DIR}/${MODEL_ID}/namelist_ref
ln -s namelist_ref_other_years ${MODEL_RUN_DIR}/${MODEL_ID}/namelist_ref
echo -e "${GREEN}[OK]${RESET} namelist_ref -> namelist_ref_other_years"

## copy EMP to new run directory
if [ "$REDATE_RESTART" == "true" ]; then
    # Find the latest EMP file from the spinup directory
    EMP_SOURCE=$(ls "${SPIN_DIR}"/EMPave_*.dat 2>/dev/null | sort -t'_' -k2 -n | tail -1)
    if [ -z "$EMP_SOURCE" ]; then
        echo -e "${RED}Error${RESET}: No EMPave files found in ${SPIN_DIR}"
        exit 1
    fi
    echo -e "${CYAN}Redate mode${RESET}: using $(basename "$EMP_SOURCE")"
else
    EMP_SOURCE="${SPIN_DIR}/EMPave_$((FIRST_YEAR_TRANSIENT - 1)).dat"
fi
EMP_TARGET="${MODEL_RUN_DIR}/${MODEL_ID}/EMPave_$((FIRST_YEAR_TRANSIENT - 1)).dat"
EMP_OLD="${MODEL_RUN_DIR}/${MODEL_ID}/EMPave.old"

if [ ! -f "$EMP_SOURCE" ]; then
    echo -e "${RED}Error${RESET}: Source EMP file does not exist: $EMP_SOURCE"
    exit 1
fi

if [ -f "$EMP_TARGET" ]; then
    mv "$EMP_TARGET" "$EMP_OLD"
fi

cp "$EMP_SOURCE" "$EMP_TARGET"
if [ $? -ne 0 ]; then
    echo -e "${RED}Error${RESET}: Failed to copy EMP file"
    exit 1
fi

echo -e "${GREEN}[OK]${RESET} EMP file copied"

## copy namelist.trc.sms to make sure both are consistent - skip for spinup runs
if [ "$FORCING_MODE" == "spinup" ]; then
    echo -e "${YELLOW}[SKIP]${RESET} Spinup run: keeping existing namelist.trc.sms"
else
    NAMELIST_SOURCE="${SPIN_DIR}/namelist.trc.sms"
    NAMELIST_TARGET="${MODEL_RUN_DIR}/${MODEL_ID}/namelist.trc.sms"
    NAMELIST_OLD="${MODEL_RUN_DIR}/${MODEL_ID}/namelist.trc.sms.old"

    if [ ! -f "$NAMELIST_SOURCE" ]; then
        echo -e "${RED}Error${RESET}: Source namelist file does not exist: $NAMELIST_SOURCE"
        exit 1
    fi

    if [ -f "$NAMELIST_TARGET" ]; then
        mv "$NAMELIST_TARGET" "$NAMELIST_OLD"
    fi

    cp "$NAMELIST_SOURCE" "$NAMELIST_TARGET"
    if [ $? -ne 0 ]; then
        echo -e "${RED}Error${RESET}: Failed to copy namelist file"
        exit 1
    fi

    echo -e "${GREEN}[OK]${RESET} namelist.trc.sms copied"
fi

## Record spinup info
SPINUP_RECORD="${MODEL_RUN_DIR}/${MODEL_ID}/spinup_info.txt"
echo "Spinup Model ID: $SPINUP_MODEL_ID" > "$SPINUP_RECORD"
echo "Spinup Year: $((FIRST_YEAR_TRANSIENT - 1))" >> "$SPINUP_RECORD"
echo "Setup Date: $(date '+%Y-%m-%d %H:%M:%S')" >> "$SPINUP_RECORD"

echo -e "${GREEN}[OK]${RESET} Spinup record saved to: $SPINUP_RECORD"
