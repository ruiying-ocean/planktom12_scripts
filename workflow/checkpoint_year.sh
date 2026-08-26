#!/bin/bash

# Validate and publish one year's restart state.
set -euo pipefail

run_dir=${1:-}
[ -n "$run_dir" ] || { echo "ERROR: usage: checkpoint_year.sh RUN_DIR YEAR" >&2; exit 1; }
[ -f "$run_dir/workflow_common.sh" ] || { echo "ERROR: workflow_common.sh not found in $run_dir" >&2; exit 1; }
# shellcheck source=workflow_common.sh
. "$run_dir/workflow_common.sh"

year=${2:-}
[ -n "$year" ] || workflow_die "usage: checkpoint_year.sh RUN_DIR YEAR"
load_run_env "$run_dir"
require_uint "$year" year

namelist_value() {
	local key=$1
	local file value
	shift
	for file in "$@"; do
		[ -f "$file" ] || continue
		value=$(awk -v wanted="$key" '
			{
				line = $0
				sub(/!.*/, "", line)
				equals = index(line, "=")
				if (!equals) next
				name = substr(line, 1, equals - 1)
				gsub(/[[:space:]]/, "", name)
				if (name != wanted) next
				value = substr(line, equals + 1)
				sub(/^[[:space:]]*[\047\"]/, "", value)
				sub(/[\047\"].*$/, "", value)
				gsub(/^[[:space:]]+|[[:space:]]+$/, "", value)
				print value
				exit
			}' "$file")
		if [ -n "$value" ]; then
			printf '%s\n' "$value"
			return
		fi
	done
}

namelist_uint() {
	local key=$1
	local file=$2
	awk -v wanted="$key" '
		{
			line = $0
			sub(/!.*/, "", line)
			equals = index(line, "=")
			if (!equals) next
			name = substr(line, 1, equals - 1)
			gsub(/[[:space:]]/, "", name)
			if (name != wanted) next
			value = substr(line, equals + 1)
			gsub(/[[:space:]]/, "", value)
			if (match(value, /^[0-9]+/)) print substr(value, RSTART, RLENGTH)
			exit
		}' "$file"
}

archive_restarts() {
	local archive_run_dir="${archiveDir%/}/$runId"
	local file dest tmp
	local files=()

	mkdir -p "$archive_run_dir"
	shopt -s nullglob
	files=(ORCA2_"${timestep}"_*restart*.nc)
	shopt -u nullglob
	[ "${#files[@]}" -gt 0 ] || workflow_die "no restart files found for timestep $timestep"

	for file in "${files[@]}"; do
		dest="$archive_run_dir/$(basename "$file")"
		if [ -L "$file" ] && [ -e "$dest" ] && [ "$file" -ef "$dest" ]; then
			continue
		fi
		tmp="${dest}.part.${SLURM_JOB_ID:-$$}"
		cp -pL "$file" "$tmp"
		mv -f "$tmp" "$dest"
		rm -f "$file"
		ln -s "$dest" "$file"
	done
}

cd "$modelDir"
mkdir -p "logs/$year" "state/$year"
rm -f "state/$year/checkpoint.ok"
if [ -s time.step ]; then
	timestep=$(awk '{printf "%.8d\n", $0}' time.step)
elif [ -s "state/$year/restart_step" ]; then
	# A retry may occur after the completed timestep was moved to old.time.step.
	timestep=$(awk '{printf "%.8d\n", $0}' "state/$year/restart_step")
else
	workflow_die "time.step is missing after model year $year"
fi
require_uint "$timestep" timestep
printf '%s\n' "$timestep" > "state/$year/restart_step"

cpus=${nemoCpus:-48}
ice_restart_name=${iceRestartName:-restart_ice_in}
require_uint "$cpus" nemoCpus

if [ "$nemoVersion" = "NEMO5" ]; then
	ocerst=$(namelist_value cn_ocerst_out namelist_cfg namelist_ref); ocerst=${ocerst:-restart}
	trcrst=$(namelist_value cn_trcrst_out namelist_top_cfg namelist_top_ref); trcrst=${trcrst:-restart_trc}
	icerst=$(namelist_value cn_icerst_out namelist_ice_cfg namelist_ice_ref); icerst=${icerst:-restart_ice}

	[ -f "ORCA2_${timestep}_${ocerst}_0000.nc" ] || \
		workflow_die "ocean restart was not produced for year $year (timestep $timestep)"
	[ -e restart_trc.nc ] && mv -f restart_trc.nc restart_trc_first_year.nc
	[ -e restart.nc ] && mv -f restart.nc restart_first_year.nc
	[ -e "${ice_restart_name}.nc" ] && \
		mv -f "${ice_restart_name}.nc" "${ice_restart_name}_first_year.nc"

	ln -sfn namelist_cfg_other_years namelist_cfg
	cur_it000=$(namelist_uint nn_it000 namelist_cfg_other_years)
	cur_itend=$(namelist_uint nn_itend namelist_cfg_other_years)
	require_uint "$cur_it000" nn_it000
	require_uint "$cur_itend" nn_itend
	span=$((cur_itend - cur_it000 + 1))
	next_it000=$((10#$timestep + 1))
	next_itend=$((10#$timestep + span))
	sed -i.bak \
		-e "s/^\( *nn_it000 *=\).*/\1 $next_it000   !  first time step/" \
		-e "s/^\( *nn_itend *=\).*/\1 $next_itend   !  last  time step/" \
		namelist_cfg_other_years
	rm -f namelist_cfg_other_years.bak
else
	ocerst=restart
	icerst=restart_ice
	trcrst=restart_trc
	[ -f "ORCA2_${timestep}_${ocerst}_0000.nc" ] || \
		workflow_die "ocean restart was not produced for year $year (timestep $timestep)"
	[ -e restart_trc.nc ] && mv -f restart_trc.nc restart_trc_first_year.nc
	[ -e restart.nc ] && mv -f restart.nc restart_first_year.nc
	[ -e restart_ice_in.nc ] && mv -f restart_ice_in.nc restart_ice_in_first_year.nc
	if [ -f EMPave.dat ]; then
		mv -f EMPave.dat "EMPave_${year}.dat"
	elif [ ! -f "EMPave_${year}.dat" ]; then
		workflow_die "EMPave.dat was not produced for NEMO3.6 year $year"
	fi
	ln -sfn "EMPave_${year}.dat" EMPave_old.dat
	ln -sfn namelist_ref_other_years namelist_ref
fi

for ((rank = 0; rank < cpus; rank++)); do
	proc=$(printf '%04d' "$rank")
	for restart_base in "$ocerst" "$icerst" "$trcrst"; do
		[ -f "ORCA2_${timestep}_${restart_base}_${proc}.nc" ] || \
			workflow_die "missing ${restart_base} rank ${proc} for timestep $timestep"
	done
	ln -sfn "ORCA2_${timestep}_${ocerst}_${proc}.nc" "restart_${proc}.nc"
	ln -sfn "ORCA2_${timestep}_${icerst}_${proc}.nc" "${ice_restart_name}_${proc}.nc"
	ln -sfn "ORCA2_${timestep}_${trcrst}_${proc}.nc" "restart_trc_${proc}.nc"
done

for log_file in ocean.output planktom-GR.log planktom-ER.log; do
	if [ -f "$log_file" ]; then
		cp -f "$log_file" "logs/$year/$log_file"
		case "$log_file" in
			ocean.output) cp -f "$log_file" "ocean.output_${year}" ;;
			planktom-GR.log) cp -f "$log_file" "planktom-GR_${year}.log" ;;
			planktom-ER.log) cp -f "$log_file" "planktom-ER_${year}.log" ;;
		esac
	fi
done

archive_restarts
[ ! -f time.step ] || mv -f time.step old.time.step
mark_task "$year" checkpoint
echo "Checkpointed $runId year $year at timestep $timestep"
