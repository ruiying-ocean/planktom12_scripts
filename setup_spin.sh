#!/bin/bash

# Usage: ./setup_spin.sh <MODEL_ID> <SPINUP_MODEL_ID>
# Example: ./setup_spin.sh TOM12_RY_ABCD TOM12_RY_SPJ2

# Colors
BOLD='\e[1m'
GREEN='\e[32m'
CYAN='\e[36m'
YELLOW='\e[33m'
RED='\e[1;31m'
DIM='\e[2m'
RESET='\e[0m'

# Log helpers
ok()   { echo -e "  ${GREEN}✓${RESET} $1"; }
skip() { echo -e "  ${YELLOW}⊘${RESET} ${DIM}$1${RESET}"; }
warn() { echo -e "  ${RED}✗${RESET} $1"; }
info() { echo -e "  ${CYAN}→${RESET} $1"; }

# Check if required arguments were provided
if [ -z "$1" ]; then
    warn "No model ID provided."
    echo "Usage: $0 <MODEL_ID> <SPINUP_MODEL_ID>"
    exit 1
fi

if [ -z "$2" ]; then
    warn "No spinup model ID provided."
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
    warn "Model run directory does not exist: $MODEL_RUN_DIR"
    exit 1
fi

if [ ! -d "$SPIN_DIR" ]; then
    warn "Spinup directory does not exist: $SPIN_DIR"
    echo "Check that the spinup model ID '$SPINUP_MODEL_ID' is correct"
    exit 1
fi

if [ ! -d "${MODEL_RUN_DIR}/${MODEL_ID}" ]; then
    warn "Model directory does not exist: ${MODEL_RUN_DIR}/${MODEL_ID}"
    exit 1
fi

STEPS_PER_YEAR=5475
FIRST_YEAR_SPINUP=1750

# Parse FIRST_YEAR_TRANSIENT from the model run directory's setUpData file
TRANSIENT_SETUP_DATA=$(find "${MODEL_RUN_DIR}/${MODEL_ID}" -maxdepth 1 -name "setUpData*.dat" | head -1)
if [ -z "$TRANSIENT_SETUP_DATA" ] || [ ! -f "$TRANSIENT_SETUP_DATA" ]; then
    warn "Could not find setUpData*.dat in ${MODEL_RUN_DIR}/${MODEL_ID}"
    exit 1
fi

FIRST_YEAR_TRANSIENT=$(grep "^yearStart:" "$TRANSIENT_SETUP_DATA" | cut -d':' -f2)
if [ -z "$FIRST_YEAR_TRANSIENT" ]; then
    warn "Could not parse yearStart from $TRANSIENT_SETUP_DATA"
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
    info "Could not parse forcing_mode, defaulting to spinup"
    FORCING_MODE="spinup"
fi

TIMESTEP=$(printf "%08d" $((($FIRST_YEAR_TRANSIENT - $FIRST_YEAR_SPINUP) * $STEPS_PER_YEAR)))

echo -e "  ${DIM}First year:${RESET}    $FIRST_YEAR_TRANSIENT"
echo -e "  ${DIM}Forcing:${RESET}       $FORCING ($FORCING_MODE)"
echo -e "  ${DIM}Timestep:${RESET}      $TIMESTEP"

# Command
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
bash ${SCRIPT_DIR}/setup_restarts.sh $SPIN_DIR $TIMESTEP ${MODEL_RUN_DIR}/${MODEL_ID}
if [ $? -ne 0 ]; then
    warn "Setup restarts script failed"
    exit 1
fi

ok "Restart links created (timestep $TIMESTEP)"

rm -f ${MODEL_RUN_DIR}/${MODEL_ID}/restart_ice_in.nc
rm -f ${MODEL_RUN_DIR}/${MODEL_ID}/restart_trc.nc
rm -f ${MODEL_RUN_DIR}/${MODEL_ID}/restart.nc

ok "Old single-file restarts cleaned"

# Update namelist_ref to use other_years (which already points to cycling for BIAS, restart for DYNAMIC)
rm -f ${MODEL_RUN_DIR}/${MODEL_ID}/namelist_ref
ln -s namelist_ref_other_years ${MODEL_RUN_DIR}/${MODEL_ID}/namelist_ref
ok "namelist_ref → namelist_ref_other_years"

## copy EMP to new run directory
EMP_SOURCE="${SPIN_DIR}/EMPave_$((FIRST_YEAR_TRANSIENT - 1)).dat"
EMP_TARGET="${MODEL_RUN_DIR}/${MODEL_ID}/EMPave_$((FIRST_YEAR_TRANSIENT - 1)).dat"
EMP_OLD="${MODEL_RUN_DIR}/${MODEL_ID}/EMPave.old"

if [ ! -f "$EMP_SOURCE" ]; then
    warn "Source EMP file does not exist: $EMP_SOURCE"
    exit 1
fi

if [ -f "$EMP_TARGET" ]; then
    mv "$EMP_TARGET" "$EMP_OLD"
fi

cp "$EMP_SOURCE" "$EMP_TARGET"
if [ $? -ne 0 ]; then
    warn "Failed to copy EMP file"
    exit 1
fi

ok "EMP file copied"

## copy namelist.trc.sms to make sure both are consistent - skip for spinup runs
if [ "$FORCING_MODE" == "spinup" ]; then
    warn "Spinup run: keeping existing namelist.trc.sms"
else
    NAMELIST_SOURCE="${SPIN_DIR}/namelist.trc.sms"
    NAMELIST_TARGET="${MODEL_RUN_DIR}/${MODEL_ID}/namelist.trc.sms"
    NAMELIST_OLD="${MODEL_RUN_DIR}/${MODEL_ID}/namelist.trc.sms.old"

    if [ ! -f "$NAMELIST_SOURCE" ]; then
        warn "Source namelist file does not exist: $NAMELIST_SOURCE"
        exit 1
    fi

    if [ -f "$NAMELIST_TARGET" ]; then
        mv "$NAMELIST_TARGET" "$NAMELIST_OLD"
    fi

    cp "$NAMELIST_SOURCE" "$NAMELIST_TARGET"
    if [ $? -ne 0 ]; then
        warn "Failed to copy namelist file"
        exit 1
    fi

    ok "namelist.trc.sms copied"
fi

## Record spinup info
SPINUP_RECORD="${MODEL_RUN_DIR}/${MODEL_ID}/spinup_info.txt"
echo "Spinup Model ID: $SPINUP_MODEL_ID" > "$SPINUP_RECORD"
echo "Spinup Year: $((FIRST_YEAR_TRANSIENT - 1))" >> "$SPINUP_RECORD"
echo "Setup Date: $(date '+%Y-%m-%d %H:%M:%S')" >> "$SPINUP_RECORD"

ok "Spinup record saved"
