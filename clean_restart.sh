#!/usr/bin/env bash
set -euo pipefail

STEP_PER_YEAR=5475

if [ $# -ne 1 ]; then
    echo "Usage: $0 RUN_NAME"
    exit 1
fi

MODEL_RUN_DIR=$HOME/scratch/ModelRuns
RUN_NAME=$1
DIR="${MODEL_RUN_DIR}/${RUN_NAME}"

shopt -s nullglob

for family in "restart_" "restart_ice_" "restart_trc_"; do
    files=( "$DIR"/ORCA2_*_${family}[0-9][0-9][0-9][0-9].nc )
    [[ -e "${files[0]}" ]] || continue

    # find latest step
    latest=$(printf '%s\n' "${files[@]}" | sort -V | tail -n 1)
    latest_step_str=$(echo "$latest" | sed -n 's/.*_\([0-9]\{8\}\)_'${family}'.*/\1/p')
    latest_step=$((10#$latest_step_str))

    keep_steps=(
	$latest_step
	$(( latest_step - 20*STEP_PER_YEAR ))
	$(( latest_step - 40*STEP_PER_YEAR ))
    )

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
