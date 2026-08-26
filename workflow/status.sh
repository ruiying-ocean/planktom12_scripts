#!/bin/bash

# Summarize recorded jobs, scheduler state, and durable task markers.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=workflow_common.sh
. "$SCRIPT_DIR/workflow_common.sh"

run_dir=${1:-$SCRIPT_DIR}
load_run_env "$run_dir"
jobs_file="$modelDir/state/jobs.tsv"
[ -f "$jobs_file" ] || workflow_die "no submitted workflow found: $jobs_file"

job_ids=$(tail -n +2 "$jobs_file" | cut -f3 | sort -u | paste -sd, -)
echo "$runId ($yearStart..$yearEnd)"
echo
echo "Recorded jobs"
column -t -s $'\t' "$jobs_file" 2>/dev/null || cat "$jobs_file"

if [ -n "$job_ids" ] && command -v squeue >/dev/null 2>&1; then
	echo
	echo "Active Slurm jobs"
	squeue -j "$job_ids" -o '%.18i %.28j %.10T %.10M %R' || true
fi

if [ -n "$job_ids" ] && command -v sacct >/dev/null 2>&1; then
	echo
	echo "Slurm history"
	sacct -X -j "$job_ids" --format=JobIDRaw,JobName%28,State,Elapsed,ExitCode || true
fi

echo
echo "Completed task markers"
find "$modelDir/state" -mindepth 2 -maxdepth 2 -name '*.ok' -print | sort | \
	sed "s#^$modelDir/state/##" || true
