#!/bin/bash

# Apply final retention and render the completed-run report.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=workflow_common.sh
. "$SCRIPT_DIR/workflow_common.sh"

run_dir=${1:-}
[ -n "$run_dir" ] || workflow_die "usage: report_run.sh RUN_DIR"
load_run_env "$run_dir"
activate_analysis_environment

archive_run_dir="${archiveDir%/}/$runId"
mkdir -p "$archive_run_dir"
cd "$modelDir"
mkdir -p "state/$yearEnd"
rm -f "state/$yearEnd/report.ok"

keep_restart_year() {
	local year=$1
	local since frequency
	[ "$year" -eq "$yearEnd" ] && return 0
	if [ "$year" -lt "$spinupEnd" ]; then
		since=$((year - spinupStart))
		frequency=$spinupRestartKeepFrequency
	else
		since=$((year - spinupEnd))
		frequency=$runRestartKeepFrequency
	fi
	[ "$frequency" -gt 0 ] || return 0
	[ $((since % frequency)) -eq 0 ]
}

keep_output_year() {
	local year=$1
	local since frequency
	[ "$year" -eq "$yearEnd" ] && return 0
	if [ "$year" -lt "$spinupEnd" ]; then
		since=$((year - spinupStart))
		frequency=$spinupOutputKeepFrequency
	else
		since=$((year - spinupEnd))
		frequency=$runOutputKeepFrequency
	fi
	[ "$frequency" -gt 0 ] || return 0
	[ $((since % frequency)) -eq 0 ]
}

for ((year = yearStart; year <= yearEnd; year++)); do
	if ! keep_restart_year "$year"; then
		step_file="state/$year/restart_step"
		if [ -s "$step_file" ]; then
			timestep=$(<"$step_file")
		else
			timestep=$(printf '%08d' "$((stepsPerYear * (year - yearStart + 1)))")
		fi
		shopt -s nullglob
		restarts=("$archive_run_dir"/ORCA2_"${timestep}"_restart*.nc \
			ORCA2_"${timestep}"_restart*.nc)
		[ "${#restarts[@]}" -gt 0 ] && rm -f "${restarts[@]}"
		shopt -u nullglob
	fi

	if ! keep_output_year "$year"; then
		for file_type in grid_T diad_T ptrc_T icemod grid_U grid_V grid_W limphy gflux_T; do
			rm -f "$archive_run_dir/ORCA2_${outputFrequency}_${year}0101_${year}1231_${file_type}.nc"
			rm -f "ORCA2_${outputFrequency}_${year}0101_${year}1231_${file_type}.nc"
		done
	fi
done

shopt -s nullglob
analysis_files=(analyser.*)
for file in "${analysis_files[@]}"; do
	[ -f "$file" ] && cp -pL "$file" "$archive_run_dir/"
done
if [ -d MOC ]; then
	moc_files=(MOC/moc_*.nc)
	if [ "${#moc_files[@]}" -gt 0 ]; then
		mkdir -p "$archive_run_dir/MOC"
		cp -pL "${moc_files[@]}" "$archive_run_dir/MOC/"
	fi
fi
shopt -u nullglob

export PYTHONPATH="$toolkitDir:$toolkitDir/analyser:$toolkitDir/shared:$toolkitDir/visualise${PYTHONPATH:+:$PYTHONPATH}"
python3 "$toolkitDir/visualise/make_monthly_plots.py" --model-id "$runId" --model-dir "$basedir"
python3 "$toolkitDir/visualise/visualise_model.py" "$runId" "$yearEnd" \
	--model-run-dir "$basedir" --output-dir "$basedir/monitor/$runId/"
bash "$toolkitDir/visualise/make_html.sh" "$runId" "$basedir"

mark_task "$yearEnd" report
