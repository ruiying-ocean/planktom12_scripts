#!/bin/bash

# Compress and publish one year's configured model outputs.
set -euo pipefail

run_dir=${1:-}
[ -n "$run_dir" ] || { echo "ERROR: usage: archive_year.sh RUN_DIR YEAR" >&2; exit 1; }
[ -f "$run_dir/workflow_common.sh" ] || { echo "ERROR: workflow_common.sh not found in $run_dir" >&2; exit 1; }
# shellcheck source=workflow_common.sh
. "$run_dir/workflow_common.sh"

year=${2:-}
[ -n "$year" ] || workflow_die "usage: archive_year.sh RUN_DIR YEAR"
load_run_env "$run_dir"
require_uint "$year" year
activate_netcdf_environment

archive_run_dir="${archiveDir%/}/$runId"
mkdir -p "$archive_run_dir"
cd "$modelDir"
mkdir -p "state/$year"
rm -f "state/$year/archive.ok"

compress_nc() {
	local file=$1
	local tmp
	[ -f "$file" ] || return 0
	if ncdump -hs "$file" 2>/dev/null | grep -q '_DeflateLevel = [1-9]'; then
		return 0
	fi
	tmp="${file%.nc}.tmp.nc"
	if nccopy -d 4 -s "$file" "$tmp"; then
		mv -f "$tmp" "$file"
	else
		rm -f "$tmp"
		workflow_die "NetCDF compression failed: $file"
	fi
}

archive_file() {
	local file=$1
	local dest="$archive_run_dir/$(basename "$file")"
	local tmp="${dest}.part.${SLURM_JOB_ID:-$$}"

	if [ -L "$file" ] && [ -e "$dest" ] && [ "$file" -ef "$dest" ]; then
		return 0
	fi
	cp -pL "$file" "$tmp"
	mv -f "$tmp" "$dest"
	rm -f "$file"
	ln -s "$dest" "$file"
}

archive_output() {
	local keep=$1
	local file=$2
	if [ "$keep" -eq 1 ]; then
		if [ -f "$file" ]; then
			compress_nc "$file"
			archive_file "$file"
		else
			echo "WARNING: output missing, not archived: $file" >&2
		fi
	else
		rm -f "$file"
	fi
}

freq=${outputFrequency:-1m}
archive_output "$keepGrid_T" "ORCA2_${freq}_${year}0101_${year}1231_grid_T.nc"
archive_output "$keepDiad"   "ORCA2_${freq}_${year}0101_${year}1231_diad_T.nc"
archive_output "$keepPtrc"   "ORCA2_${freq}_${year}0101_${year}1231_ptrc_T.nc"
archive_output "$keepIce"    "ORCA2_${freq}_${year}0101_${year}1231_icemod.nc"
archive_output "$keepGrid_U" "ORCA2_${freq}_${year}0101_${year}1231_grid_U.nc"
archive_output "$keepGrid_V" "ORCA2_${freq}_${year}0101_${year}1231_grid_V.nc"
archive_output "$keepGrid_W" "ORCA2_${freq}_${year}0101_${year}1231_grid_W.nc"
archive_output "$keepLimPhy" "ORCA2_${freq}_${year}0101_${year}1231_limphy.nc"
archive_output "$keepGflux"  "ORCA2_${freq}_${year}0101_${year}1231_gflux_T.nc"

if [ -f "MOC/moc_${year}.nc" ]; then
	mkdir -p "$archive_run_dir/MOC"
	cp -pL "MOC/moc_${year}.nc" "$archive_run_dir/MOC/"
fi

shopt -s nullglob
metadata=("EMPave_${year}.dat" "ocean.output_${year}" "planktom-GR_${year}.log" \
	"planktom-ER_${year}.log" namelist* *.xml setUpData*dat)
for file in "${metadata[@]}"; do
	[ -f "$file" ] && cp -pL "$file" "$archive_run_dir/"
done
shopt -u nullglob
[ -f opa ] && cp -pL opa "$archive_run_dir/opa"

mark_task "$year" archive
