#!/usr/bin/env bash
set -euo pipefail

STEP_PER_YEAR=5475

if [ $# -lt 1 ] || [ $# -gt 2 ]; then
    echo "Usage: $0 RUN_NAME [MODE]"
    echo "  MODE: fixed    - keep latest, -20yr, -40yr (default)"
    echo "        adaptive - keep first, middle, latest"
    exit 1
fi

MODEL_RUN_DIR=$HOME/scratch/ModelRuns
RUN_NAME=$1
MODE=${2:-fixed}

if [[ "$MODE" != "fixed" && "$MODE" != "adaptive" ]]; then
    echo "Error: MODE must be 'fixed' or 'adaptive', got '$MODE'"
    exit 1
fi

DIR="${MODEL_RUN_DIR}/${RUN_NAME}"

shopt -s nullglob

for family in "restart_" "restart_ice_" "restart_trc_"; do
    files=( "$DIR"/ORCA2_*_${family}[0-9][0-9][0-9][0-9].nc )
    [[ -e "${files[0]}" ]] || continue

    # find first and latest steps
    sorted=($(printf '%s\n' "${files[@]}" | sort -V))
    first="${sorted[0]}"
    latest="${sorted[-1]}"

    first_step_str=$(echo "$first" | sed -n 's/.*_\([0-9]\{8\}\)_'${family}'.*/\1/p')
    first_step=$((10#$first_step_str))

    latest_step_str=$(echo "$latest" | sed -n 's/.*_\([0-9]\{8\}\)_'${family}'.*/\1/p')
    latest_step=$((10#$latest_step_str))

    if [[ "$MODE" == "adaptive" ]]; then
	middle_step=$(( (first_step + latest_step) / 2 ))
	# round middle to nearest year boundary
	middle_step=$(( (middle_step / STEP_PER_YEAR) * STEP_PER_YEAR ))
	keep_steps=(
	    $first_step
	    $middle_step
	    $latest_step
	)
    else
	# fixed mode: latest, -20yr, -40yr
	keep_steps=(
	    $latest_step
	    $(( latest_step - 20*STEP_PER_YEAR ))
	    $(( latest_step - 40*STEP_PER_YEAR ))
	)
    fi

    echo "Run ${RUN_NAME} | family=${family%_} | keep steps: ${keep_steps[*]}"

    declare -A keep_map=()
    for k in "${keep_steps[@]}"; do ((k>=0)) && keep_map["$k"]=1; done

    for f in "${files[@]}"; do
	step_str=$(echo "$f" | sed -n 's/.*_\([0-9]\{8\}\)_'${family}'.*/\1/p')
	step=$((10#$step_str))
	if [[ -z "${keep_map[$step]+x}" ]]; then
	    rm "$f"
	fi
    done
done
