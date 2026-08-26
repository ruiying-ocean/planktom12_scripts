#!/bin/bash

# Execute one model year inside a Slurm allocation.
set -euo pipefail

run_dir=${1:-}
[ -n "$run_dir" ] || { echo "ERROR: usage: run_year.sh RUN_DIR YEAR ib|compute" >&2; exit 1; }
[ -f "$run_dir/workflow_common.sh" ] || { echo "ERROR: workflow_common.sh not found in $run_dir" >&2; exit 1; }
# shellcheck source=workflow_common.sh
. "$run_dir/workflow_common.sh"

year=${2:-}
partition=${3:-}
[ -n "$year" ] && [ -n "$partition" ] || \
	workflow_die "usage: run_year.sh RUN_DIR YEAR ib|compute"

load_run_env "$run_dir"
require_uint "$year" year
(( year >= yearStart && year <= yearEnd )) || \
	workflow_die "year $year is outside configured range $yearStart..$yearEnd"

restore_nounset=0
case $- in *u*) restore_nounset=1; set +u ;; esac
# shellcheck disable=SC1091
. /etc/profile
module purge
case "$partition" in
	ib)
		module add gcc/9.2.0 netcdf/4.7.4/parallel/gcc-openmpi \
			hdf5/1.10.6/gcc-openmpi mpi/openmpi/4.0.3/gcc/ib perl
		mpi_transport=(--mca btl_openib_if_include mlx5_0:1,mlx5_1:1 \
			--mca btl_vader_single_copy_mechanism none)
		;;
	compute)
		module add gcc/9.2.0 netcdf/4.7.4/parallel/gcc-openmpi \
			hdf5/1.10.6/gcc-openmpi mpi/openmpi/4.0.3/gcc perl
		mpi_transport=(--mca btl '^openib' --mca btl_vader_single_copy_mechanism none)
		;;
	*) workflow_die "unknown partition '$partition' (expected ib or compute)" ;;
esac
[ "$restore_nounset" -eq 1 ] && set -u
module list

cpus=${nemoCpus:-48}
xios=${xiosCpus:-0}
app="./${executable:-opa}"
use_xios=${useXiosServer:-false}
require_uint "$cpus" nemoCpus
require_uint "$xios" xiosCpus

cd "$modelDir"
[ -x "$app" ] || workflow_die "model executable is missing or not executable: $modelDir/$app"
rm -f "state/$year/run.ok"

echo "Running $runId year $year on $partition ($cpus NEMO ranks, $xios XIOS ranks)"
date
if [ "$nemoVersion" = "NEMO5" ] && [ "$use_xios" = "true" ]; then
	[ -x ./xios_server.exe ] || workflow_die "xios_server.exe is missing or not executable"
	mpirun "${mpi_transport[@]}" -np "$xios" ./xios_server.exe : -np "$cpus" "$app"
else
	mpirun "${mpi_transport[@]}" -np "$cpus" "$app"
fi
date

[ -s time.step ] || workflow_die "model completed without writing time.step"
mark_task "$year" run
