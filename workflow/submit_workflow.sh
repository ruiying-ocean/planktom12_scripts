#!/bin/bash

# Build the year-by-year workflow with native Slurm dependencies.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=workflow_common.sh
. "$SCRIPT_DIR/workflow_common.sh"

run_dir=${1:-}
start_year=${2:-}
end_year=${3:-}
partition=${4:-auto}
[ -n "$run_dir" ] || workflow_die \
	"usage: submit_workflow.sh RUN_DIR [START_YEAR [END_YEAR [auto|ib|compute]]]"

load_run_env "$run_dir"
start_year=${start_year:-$yearStart}
end_year=${end_year:-$yearEnd}
require_uint "$start_year" start_year
require_uint "$end_year" end_year
(( start_year >= yearStart && end_year <= yearEnd && start_year <= end_year )) || \
	workflow_die "submission range $start_year..$end_year is outside $yearStart..$yearEnd"
command -v sbatch >/dev/null 2>&1 || workflow_die "sbatch is not available"

if (( start_year > yearStart )); then
	previous_year=$((start_year - 1))
	[ -f "$modelDir/state/$previous_year/checkpoint.ok" ] || \
		workflow_die "cannot start at $start_year: checkpoint $previous_year is not complete"
	[ -f "$modelDir/state/$previous_year/archive.ok" ] || \
		workflow_die "cannot start at $start_year: archive $previous_year is not complete"
fi

if [ "$partition" = auto ]; then
	command -v squeue >/dev/null 2>&1 || workflow_die "squeue is required for partition=auto"
	ib_jobs=$(squeue -p ib -u "$USER" -h 2>/dev/null | wc -l | tr -d ' ')
	if [ "$ib_jobs" -ge 2 ]; then partition=compute; else partition=ib; fi
fi
case "$partition" in ib|compute) ;; *) workflow_die "unknown partition: $partition" ;; esac

for script in run_year.sh checkpoint_year.sh analyse_year.sh archive_year.sh report_run.sh; do
	[ -x "$modelDir/$script" ] || workflow_die "workflow task is missing or not executable: $modelDir/$script"
done

submit_job() {
	local response job_id
	response=$(sbatch --parsable "$@")
	job_id=${response%%;*}
	require_uint "$job_id" "Slurm job id"
	printf '%s\n' "$job_id"
}

record_job() {
	printf '%s\t%s\t%s\t%s\n' "$1" "$2" "$3" "${4:--}" >> "$jobs_file"
}

mkdir -p "$modelDir/logs" "$modelDir/state"
jobs_file="$modelDir/state/jobs.tsv"
if [ ! -f "$jobs_file" ]; then
	printf 'year\ttask\tjob_id\tdependency\n' > "$jobs_file"
fi

nemo_tasks=${nemoCpus:-48}
require_uint "$nemo_tasks" nemoCpus
total_tasks=$nemo_tasks
if [ "$nemoVersion" = NEMO5 ] && [ "${useXiosServer:-true}" = true ]; then
	xios_tasks=${xiosCpus:-0}
	require_uint "$xios_tasks" xiosCpus
	(( xios_tasks > 0 )) || workflow_die "xiosCpus must be greater than zero when XIOS is enabled"
	total_tasks=$((total_tasks + xios_tasks))
fi
require_uint "$total_tasks" total_tasks

run_resources=(--partition="$partition" --time=02:00:00 --mem=256G \
	--ntasks="$total_tasks" --mail-type=ALL)
if [ "$partition" = ib ]; then
	run_resources+=(--qos=ib --constraint=mlx5)
fi
if [ "$nemoVersion" != NEMO5 ]; then
	run_resources+=(--ntasks-per-node="$nemo_tasks")
fi

previous_checkpoint=
previous_archive=
for ((year = start_year; year <= end_year; year++)); do
	mkdir -p "$modelDir/logs/$year" "$modelDir/state/$year"

	run_args=("${run_resources[@]}" --job-name="${simulation}${year}" \
		--chdir="$modelDir" --output="$modelDir/planktom-GR.log" \
		--error="$modelDir/planktom-ER.log")
	if [ -n "$previous_checkpoint" ]; then
		run_args+=(--dependency="afterok:$previous_checkpoint")
	fi
	run_job=$(submit_job "${run_args[@]}" "$modelDir/run_year.sh" "$modelDir" "$year" "$partition")
	record_job "$year" run "$run_job" "$previous_checkpoint"

	checkpoint_job=$(submit_job --partition=compute --time=02:00:00 --mem=16G --ntasks=1 \
		--job-name="ckpt${simulation}${year}" --dependency="afterok:$run_job" \
		--chdir="$modelDir" --output="$modelDir/logs/$year/checkpoint-%j.log" \
		--error="$modelDir/logs/$year/checkpoint-%j.log" \
		"$modelDir/checkpoint_year.sh" "$modelDir" "$year")
	record_job "$year" checkpoint "$checkpoint_job" "$run_job"

	analysis_dependencies=$checkpoint_job
	[ -n "$previous_archive" ] && analysis_dependencies="${analysis_dependencies}:${previous_archive}"
	analyse_job=$(submit_job --partition=compute --time=06:00:00 --mem=64G --ntasks=1 \
		--exclude=compute086 \
		--job-name="ana${simulation}${year}" --dependency="afterok:$analysis_dependencies" \
		--chdir="$modelDir" --output="$modelDir/logs/$year/analyse-%j.log" \
		--error="$modelDir/logs/$year/analyse-%j.log" \
		"$modelDir/analyse_year.sh" "$modelDir" "$year")
	record_job "$year" analyse "$analyse_job" "$analysis_dependencies"

	archive_job=$(submit_job --partition=compute --time=06:00:00 --mem=64G --ntasks=1 \
		--exclude=compute086 \
		--job-name="arc${simulation}${year}" --dependency="afterok:$analyse_job" \
		--chdir="$modelDir" --output="$modelDir/logs/$year/archive-%j.log" \
		--error="$modelDir/logs/$year/archive-%j.log" \
		"$modelDir/archive_year.sh" "$modelDir" "$year")
	record_job "$year" archive "$archive_job" "$analyse_job"

	previous_checkpoint=$checkpoint_job
	previous_archive=$archive_job
done

if [ "$end_year" -eq "$yearEnd" ]; then
	report_job=$(submit_job --partition=compute --time=20:00:00 --mem=80G --ntasks=1 \
		--exclude=compute086 \
		--job-name="report${simulation}" --dependency="afterok:$previous_archive" \
		--chdir="$modelDir" --output="$modelDir/logs/report-%j.log" \
		--error="$modelDir/logs/report-%j.log" \
		"$modelDir/report_run.sh" "$modelDir")
	record_job "$end_year" report "$report_job" "$previous_archive"
fi

echo "Submitted $runId years $start_year..$end_year on $partition"
echo "Workflow jobs: $jobs_file"
echo "Status: $modelDir/status.sh $modelDir"
