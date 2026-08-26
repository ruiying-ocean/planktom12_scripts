#!/bin/bash

# Run the scientific diagnostics for one completed model year.
set -euo pipefail

run_dir=${1:-}
[ -n "$run_dir" ] || { echo "ERROR: usage: analyse_year.sh RUN_DIR YEAR" >&2; exit 1; }
[ -f "$run_dir/workflow_common.sh" ] || { echo "ERROR: workflow_common.sh not found in $run_dir" >&2; exit 1; }
# shellcheck source=workflow_common.sh
. "$run_dir/workflow_common.sh"

year=${2:-}
[ -n "$year" ] || workflow_die "usage: analyse_year.sh RUN_DIR YEAR"
load_run_env "$run_dir"
require_uint "$year" year
activate_analysis_environment

[ -f "$modelDir/analyser_config.toml" ] || workflow_die "analyser_config.toml is missing"
[ -f "$toolkitDir/analyser/analyser.py" ] || workflow_die "analyser snapshot is missing"

export PYTHONPATH="$toolkitDir:$toolkitDir/analyser:$toolkitDir/shared:$toolkitDir/visualise${PYTHONPATH:+:$PYTHONPATH}"
cd "$modelDir"
mkdir -p "state/$year"
rm -f "state/$year/analyse.ok"

# A retried analysis replaces its prior row instead of appending a duplicate.
shopt -s nullglob
for csv in analyser.*.annual.csv; do
	awk -F',' -v target_year="$year" 'NR == 1 || $1 != target_year' "$csv" > "${csv}.retry"
	mv -f "${csv}.retry" "$csv"
done
shopt -u nullglob

grid_v_file="ORCA2_1m_${year}0101_${year}1231_grid_V.nc"
grid_t_file="ORCA2_1m_${year}0101_${year}1231_grid_T.nc"
if [ -f "$grid_v_file" ]; then
	bash "$toolkitDir/compute_amoc.sh" "$grid_v_file" "$grid_t_file"
fi

python3 "$toolkitDir/analyser/analyser.py" "$modelDir/analyser_config.toml" "$year" "$year"
python3 "$toolkitDir/visualise/make_timeseries.py" "$runId" --model-run-dir "$basedir"

shopt -s nullglob
for csv in analyser.*.annual.csv; do
	(head -n 1 "$csv" && tail -n +2 "$csv" | sort -t',' -k1,1n) > "${csv}.sorted"
	mv -f "${csv}.sorted" "$csv"
done
shopt -u nullglob

mark_task "$year" analyse
