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

NEMO_VERSION=$(grep "^nemoVersion:" "$TRANSIENT_SETUP_DATA" | cut -d':' -f2)
NEMO_CPUS=$(grep "^nemoCpus:" "$TRANSIENT_SETUP_DATA" | cut -d':' -f2)
STEPS_PER_YEAR=$(grep "^stepsPerYear:" "$TRANSIENT_SETUP_DATA" | cut -d':' -f2)
SPINUP_SETUP_DATA=$(find "$SPIN_DIR" -maxdepth 1 -name "setUpData*.dat" | head -1)
SOURCE_YEAR_START=$(grep "^yearStart:" "$SPINUP_SETUP_DATA" 2>/dev/null | cut -d':' -f2)
ICE_RESTART_NAME=$(grep "^iceRestartName:" "$TRANSIENT_SETUP_DATA" | cut -d':' -f2)
NEMO_CPUS=${NEMO_CPUS:-48}
if [ -z "$STEPS_PER_YEAR" ]; then
    if [ "$NEMO_VERSION" = "NEMO5" ]; then
        STEPS_PER_YEAR=5840
    else
        STEPS_PER_YEAR=5475
    fi
fi
if [ -z "$ICE_RESTART_NAME" ]; then
    if [ "$NEMO_VERSION" = "NEMO5" ]; then
        ICE_RESTART_NAME="restart_ice"
    else
        ICE_RESTART_NAME="restart_ice_in"
    fi
fi
if [ "$NEMO_VERSION" = "NEMO5" ]; then
    CONTROL_NAMELIST="namelist_cfg"
else
    CONTROL_NAMELIST="namelist_ref"
fi

# Resolve the epoch used in ORCA2_<timestep> restart filenames. Some grafted
# transient runs keep the upstream restart counter rather than resetting at
# their own yearStart. Infer that counter from source output years when possible:
# restart step S belongs to the start of output_year+1.
EPOCHS=()
EPOCH_LABELS=()
add_epoch_candidate() {
    local epoch=$1
    local label=$2
    local existing

    [ -n "$epoch" ] || return
    case "$epoch" in
        *[!0-9]*) return ;;
    esac
    for existing in "${EPOCHS[@]}"; do
        [ "$existing" = "$epoch" ] && return
    done
    EPOCHS+=("$epoch")
    EPOCH_LABELS+=("$label")
}

record_epoch_score() {
    local epoch=$1
    local existing

    for existing in "${!INFERRED_EPOCHS[@]}"; do
        if [ "${INFERRED_EPOCHS[$existing]}" = "$epoch" ]; then
            INFERRED_COUNTS[$existing]=$((INFERRED_COUNTS[$existing] + 1))
            return
        fi
    done

    INFERRED_EPOCHS+=("$epoch")
    INFERRED_COUNTS+=(1)
}

format_timestep_for_epoch() {
    local epoch=$1
    local offset=$((FIRST_YEAR_TRANSIENT - epoch))

    [ "$offset" -ge 0 ] || return 1
    printf "%08d" $((offset * STEPS_PER_YEAR))
}

FOUND_STEPS=$(ls "${SPIN_DIR}"/ORCA2_*_restart_0000.nc 2>/dev/null | sed 's/.*ORCA2_\([0-9]*\)_restart.*/\1/' | sort -u)
SOURCE_OUTPUT_YEARS=$(find "$SPIN_DIR" -maxdepth 1 \( \
    -name "ORCA2_*_[0-9][0-9][0-9][0-9]0101_[0-9][0-9][0-9][0-9]1231_*.nc" -o \
    -name "ocean.output_[0-9][0-9][0-9][0-9]" -o \
    -name "EMPave_[0-9][0-9][0-9][0-9].dat" \
    \) 2>/dev/null | sed -n '
        s/.*_\([0-9]\{4\}\)0101_[0-9]\{4\}1231_.*/\1/p
        s/.*ocean\.output_\([0-9]\{4\}\)$/\1/p
        s/.*EMPave_\([0-9]\{4\}\)\.dat$/\1/p
    ' | sort -u)

INFERRED_EPOCHS=()
INFERRED_COUNTS=()
for step in $FOUND_STEPS; do
    STEP_NUM=$((10#$step))
    [ $((STEP_NUM % STEPS_PER_YEAR)) -eq 0 ] || continue
    STEP_YEAR_OFFSET=$((STEP_NUM / STEPS_PER_YEAR))
    for output_year in $SOURCE_OUTPUT_YEARS; do
        record_epoch_score $((output_year + 1 - STEP_YEAR_OFFSET))
    done
done

BEST_INFERRED_EPOCH=""
BEST_INFERRED_COUNT=0
for i in "${!INFERRED_EPOCHS[@]}"; do
    if [ "${INFERRED_COUNTS[$i]}" -gt "$BEST_INFERRED_COUNT" ]; then
        BEST_INFERRED_COUNT=${INFERRED_COUNTS[$i]}
        BEST_INFERRED_EPOCH=${INFERRED_EPOCHS[$i]}
    fi
done

add_epoch_candidate "$BEST_INFERRED_EPOCH" "inferred from source outputs"
add_epoch_candidate "$SOURCE_YEAR_START" "source yearStart"
add_epoch_candidate "1750" "default"

FIRST_YEAR_SPINUP=""
SPINUP_EPOCH_SOURCE=""
TIMESTEP=""
EXPECTED_RESTART=""
for i in "${!EPOCHS[@]}"; do
    candidate_epoch=${EPOCHS[$i]}
    candidate_timestep=$(format_timestep_for_epoch "$candidate_epoch") || continue
    candidate_restart="${SPIN_DIR}/ORCA2_${candidate_timestep}_restart_0000.nc"
    if [ -f "$candidate_restart" ]; then
        FIRST_YEAR_SPINUP=$candidate_epoch
        SPINUP_EPOCH_SOURCE=${EPOCH_LABELS[$i]}
        TIMESTEP=$candidate_timestep
        EXPECTED_RESTART=$candidate_restart
        break
    fi
done

if [ -z "$FIRST_YEAR_SPINUP" ]; then
    FIRST_YEAR_SPINUP=${EPOCHS[0]:-1750}
    SPINUP_EPOCH_SOURCE=${EPOCH_LABELS[0]:-default}
    TIMESTEP=$(format_timestep_for_epoch "$FIRST_YEAR_SPINUP")
    EXPECTED_RESTART="${SPIN_DIR}/ORCA2_${TIMESTEP}_restart_0000.nc"
fi

if [ ! -f "$EXPECTED_RESTART" ]; then
    # If none of the candidate epochs matched exactly, choose the epoch that
    # makes the available restart years closest to the requested first year.
    if [ -n "$FOUND_STEPS" ]; then
        BEST_EPOCH=""
        BEST_LABEL=""
        BEST_SCORE=""
        for i in "${!EPOCHS[@]}"; do
            candidate_epoch=${EPOCHS[$i]}
            for step in $FOUND_STEPS; do
                STEP_NUM=$((10#$step))
                YEAR=$(($STEP_NUM / $STEPS_PER_YEAR + candidate_epoch))
                SCORE=$((YEAR - FIRST_YEAR_TRANSIENT))
                [ "$SCORE" -lt 0 ] && SCORE=$((-SCORE))
                if [ -z "$BEST_SCORE" ] || [ "$SCORE" -lt "$BEST_SCORE" ]; then
                    BEST_SCORE=$SCORE
                    BEST_EPOCH=$candidate_epoch
                    BEST_LABEL=${EPOCH_LABELS[$i]}
                fi
            done
        done
        if [ -n "$BEST_EPOCH" ] && [ "$BEST_EPOCH" != "$FIRST_YEAR_SPINUP" ]; then
            FIRST_YEAR_SPINUP=$BEST_EPOCH
            SPINUP_EPOCH_SOURCE="$BEST_LABEL; nearest available restart"
            TIMESTEP=$(format_timestep_for_epoch "$FIRST_YEAR_SPINUP")
            EXPECTED_RESTART="${SPIN_DIR}/ORCA2_${TIMESTEP}_restart_0000.nc"
        fi
    fi
fi

echo -e "  ${DIM}First year:${RESET}    $FIRST_YEAR_TRANSIENT"
echo -e "  ${DIM}Spinup epoch:${RESET}  $FIRST_YEAR_SPINUP ($SPINUP_EPOCH_SOURCE)"
echo -e "  ${DIM}Forcing:${RESET}       $FORCING ($FORCING_MODE)"
echo -e "  ${DIM}Timestep:${RESET}      $TIMESTEP"

# Check if restart files exist for the expected timestep
if [ ! -f "$EXPECTED_RESTART" ]; then
    warn "Restart file not found for timestep $TIMESTEP (needed to start year $FIRST_YEAR_TRANSIENT)"
    info "Searching for nearby restart files in $SPIN_DIR ..."

    if [ -z "$FOUND_STEPS" ]; then
        warn "No restart files found in $SPIN_DIR"
        exit 1
    fi

    echo ""
    info "Tried restart-counter epochs:"
    for i in "${!EPOCHS[@]}"; do
        candidate_epoch=${EPOCHS[$i]}
        candidate_timestep=$(format_timestep_for_epoch "$candidate_epoch") || candidate_timestep="before-epoch"
        echo -e "    ${EPOCH_LABELS[$i]}: $candidate_epoch -> $candidate_timestep"
    done

    # Show nearby timesteps (within ±1 year of expected)
    STEP_MINUS1=$(printf "%08d" $((($FIRST_YEAR_TRANSIENT - 1 - $FIRST_YEAR_SPINUP) * $STEPS_PER_YEAR)))
    STEP_PLUS1=$(printf "%08d" $((($FIRST_YEAR_TRANSIENT + 1 - $FIRST_YEAR_SPINUP) * $STEPS_PER_YEAR)))

    echo ""
    info "Available restart timesteps:"
    for step in $FOUND_STEPS; do
        STEP_NUM=$((10#$step))
        YEAR=$(($STEP_NUM / $STEPS_PER_YEAR + $FIRST_YEAR_SPINUP))
        MARKER=""
        if [ "$step" = "$STEP_MINUS1" ] || [ "$step" = "$STEP_PLUS1" ]; then
            MARKER=" ${YELLOW}← close to expected${RESET}"
        fi
        echo -e "    $step  (starts year $YEAR)${MARKER}"
    done
    echo ""
    info "Expected timestep was $TIMESTEP (starts year $FIRST_YEAR_TRANSIENT)"
    exit 1
fi

# Command
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
bash ${SCRIPT_DIR}/setup_restarts.sh $SPIN_DIR $TIMESTEP ${MODEL_RUN_DIR}/${MODEL_ID} $ICE_RESTART_NAME $NEMO_CPUS
if [ $? -ne 0 ]; then
    warn "Setup restarts script failed"
    exit 1
fi

ok "Restart links created (timestep $TIMESTEP)"

rm -f ${MODEL_RUN_DIR}/${MODEL_ID}/${ICE_RESTART_NAME}.nc
rm -f ${MODEL_RUN_DIR}/${MODEL_ID}/restart_trc.nc
rm -f ${MODEL_RUN_DIR}/${MODEL_ID}/restart.nc

ok "Old single-file restarts cleaned"

# Active namelist for the spinup graft's FIRST year -- version-dependent because
# daymod.F90 (day_rst) handles nn_rstctl=2 differently between versions:
#   NEMO3.6: nn_rstctl=2 has NO nit000 check and OVERRIDES nit000 = restart_kt+1,
#            so other_years works directly for a graft (historical, intended path).
#   NEMO5:   nn_rstctl=2 ENFORCES nit000 == restart_kt+1 (check is `nrstdt /= 0`)
#            and no longer auto-resets nit000, so a graft on other_years aborts
#            ("problem with nit000 for the restart"). It must boot on first_year
#            (nn_rstctl=0, no check, fresh clock + namelist nn_date0); the resubmit
#            job auto-promotes to other_years after year 1 writes a local restart.
rm -f ${MODEL_RUN_DIR}/${MODEL_ID}/${CONTROL_NAMELIST}
if [ "$NEMO_VERSION" = "NEMO5" ]; then
    ln -s ${CONTROL_NAMELIST}_first_year ${MODEL_RUN_DIR}/${MODEL_ID}/${CONTROL_NAMELIST}
    ok "${CONTROL_NAMELIST} → ${CONTROL_NAMELIST}_first_year (NEMO5: nn_rstctl=0 for graft)"
else
    ln -s ${CONTROL_NAMELIST}_other_years ${MODEL_RUN_DIR}/${MODEL_ID}/${CONTROL_NAMELIST}
    ok "${CONTROL_NAMELIST} → ${CONTROL_NAMELIST}_other_years"
fi

## copy EMP to new run directory (only needed when nn_fwb=2)
## NEMO5 stores the freshwater-budget state (a_fwb / emp_corr) in the ocean
## restart, not in an EMPave*.dat text file (see sbcfwb.F90 iom_rstput/iom_get),
## so NEMO5 never writes one and none is needed here -- skip regardless of nn_fwb.
NN_FWB=$(grep -E "^\s*nn_fwb\s*=" ${MODEL_RUN_DIR}/${MODEL_ID}/${CONTROL_NAMELIST}_other_years 2>/dev/null | head -1 | awk -F'=' '{print $2}' | awk '{print $1}')
if [ -z "$NN_FWB" ]; then
    NN_FWB=2  # default in NEMO
fi

if [ "$NEMO_VERSION" = "NEMO5" ]; then
    skip "EMP file not needed (NEMO5 carries fwb in the ocean restart)"
elif [ "$NN_FWB" -eq 2 ]; then
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
else
    skip "EMP file not needed (nn_fwb=$NN_FWB)"
fi

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
