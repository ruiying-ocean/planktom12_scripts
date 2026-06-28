#!/bin/bash

# parameters for tidying up
yearFrom=$1
yearTo=$2
version=$3
modelOutputDir=$4

# Read parameters by name from the run-dir setUpData (single source of truth).
# Previously these were re-encoded positionally into a tidy_parms file by
# setUpRun; reading by name avoids order/format misalignment.
setupData=$(ls setUpData*dat 2>/dev/null | head -1)
getparm() { grep -h "^$1:" "$setupData" 2>/dev/null | head -1 | cut -d':' -f2; }

spinupStart=$(getparm spinupStart)
spinupEnd=$(getparm spinupEnd)
spinupRestartKeepFrequency=$(getparm spinupRestartKeepFrequency)
spinupOutputKeepFrequency=$(getparm spinupOutputKeepFrequency)
runRestartKeepFrequency=$(getparm runRestartKeepFrequency)
runOutputKeepFrequency=$(getparm runOutputKeepFrequency)
keepGrid_T=$(getparm keepGrid_T)
keepDiad=$(getparm keepDiad)
keepPtrc=$(getparm keepPtrc)
keepIce=$(getparm keepIce)
keepGrid_V=$(getparm keepGrid_V)
keepGrid_U=$(getparm keepGrid_U)
keepGrid_W=$(getparm keepGrid_W)
keepLimPhy=$(getparm keepLimPhy)
keepGflux=$(getparm keepGflux); keepGflux=${keepGflux:-0}
yearStart=${yearStart:-$(getparm yearStart)}
yearEnd=${yearEnd:-$(getparm yearEnd)}
spinupStart=${spinupStart:-$yearStart}
spinupEnd=${spinupEnd:-$yearStart}
spinupRestartKeepFrequency=${spinupRestartKeepFrequency:-1}
spinupOutputKeepFrequency=${spinupOutputKeepFrequency:-1}
runRestartKeepFrequency=${runRestartKeepFrequency:-1}
runOutputKeepFrequency=${runOutputKeepFrequency:-1}
keepGrid_T=${keepGrid_T:-0}
keepDiad=${keepDiad:-0}
keepPtrc=${keepPtrc:-0}
keepIce=${keepIce:-0}
keepGrid_V=${keepGrid_V:-0}
keepGrid_U=${keepGrid_U:-0}
keepGrid_W=${keepGrid_W:-0}
keepLimPhy=${keepLimPhy:-0}

# Echo parameters to tidy.log
echo $spinupStart
echo $spinupEnd
echo $spinupRestartKeepFrequency
echo $spinupOutputKeepFrequency
echo $runRestartKeepFrequency
echo $runOutputKeepFrequency
echo $keepGrid_T
echo $keepDiad
echo $keepPtrc
echo $keepIce
echo $keepGrid_V
echo $keepGrid_U
echo $keepGrid_W
echo $keepLimPhy
echo $keepGflux

# Model timesteps per year, used to build restart-file timestep numbers. This
# is per-run (depends on NEMO version / rn_rdt), so read it from setUpData like
# setup_spin.sh does (NEMO5 90-min step = 5840; NEMO3.6 96-min step = 5475).
nemoVersion=$(grep -h "^nemoVersion:" setUpData*dat 2>/dev/null | head -1 | cut -d':' -f2)
pointsPerYear=$(grep -h "^stepsPerYear:" setUpData*dat 2>/dev/null | head -1 | cut -d':' -f2)
if [ -z "$pointsPerYear" ]; then
	if [ "$nemoVersion" = "NEMO5" ]; then pointsPerYear=5840; else pointsPerYear=5475; fi
fi
freq=1m

# Compress a NetCDF file in place with nccopy -d 4 -s.
# No-op if file is missing or already has deflate>=1. Original is preserved on failure.
compress_nc() {
	local f=$1
	[ -f "$f" ] || return 0
	if ncdump -hs "$f" 2>/dev/null | grep -q '_DeflateLevel = [1-9]'; then
		return 0
	fi
	local tmp="${f%.nc}.tmp.nc"
	if nccopy -d 4 -s "$f" "$tmp"; then
		mv -f "$tmp" "$f"
	else
		echo "compress_nc FAILED on $f (original untouched)"
		rm -f "$tmp"
	fi
}

is_uint() {
	case "$1" in
		''|*[!0-9]*) return 1 ;;
		*) return 0 ;;
	esac
}

archive_output() {
	local keep=$1
	local file=$2

	[[ $keep -eq 1 ]] || return 0
	if [[ -f "$file" ]]; then
		cp "$file" "$afm_dir$model_id"
	else
		echo "WARNING: output missing, not archived: $file"
	fi
}

delete_local_output() {
	local keep=$1
	local file=$2
	local afm_file="${afm_dir}${model_id}/$file"

	if [[ -f "$afm_file" || $keep -eq 0 ]]; then
		rm -f "$file"
	fi
}

link_archived_output() {
	local keep=$1
	local file=$2
	local afm_file="${afm_dir}${model_id}/$file"

	[[ $keep -eq 1 ]] || return 0
	if [[ -f "$afm_file" ]]; then
		ln -sf "$afm_file" "$file"
	else
		echo "skip symlink, archive missing: $afm_file"
	fi
}

read_restart_scalar() {
	local var=$1
	local file=$2

	command -v ncdump >/dev/null 2>&1 || return
	ncdump -v "$var" "$file" 2>/dev/null |
		sed -n "s/^[[:space:]]*$var[[:space:]]*=[[:space:]]*\\([0-9][0-9]*\\).*/\\1/p" |
		head -1
}

restart_year_from_file() {
	local file=$1
	local ndastp

	ndastp=$(read_restart_scalar ndastp "$file")
	is_uint "$ndastp" || return 1
	echo "${ndastp:0:4}"
}

restart_step_from_file() {
	basename "$1" | sed -n 's/^ORCA2_\([0-9][0-9]*\)_restart.*/\1/p'
}

find_restart_file_for_step() {
	local step=$1
	local file

	for file in ORCA2_${step}_restart_0000.nc ORCA2_${step}_restart_out_0000.nc \
		${afm_dir}${model_id}/ORCA2_${step}_restart_0000.nc \
		${afm_dir}${model_id}/ORCA2_${step}_restart_out_0000.nc \
		ORCA2_${step}_restart*_0000.nc ${afm_dir}${model_id}/ORCA2_${step}_restart*_0000.nc; do
		[[ -f "$file" ]] && echo "$file" && return 0
	done

	return 1
}

restart_step_for_year() {
	local target_year=$1
	local file year step

	for file in ORCA2_*_restart*_0000.nc; do
		[[ -f "$file" ]] || continue
		year=$(restart_year_from_file "$file") || continue
		[[ "$year" = "$target_year" ]] || continue
		step=$(restart_step_from_file "$file")
		is_uint "$step" || continue
		echo "$step"
		return 0
	done

	return 1
}

restart_timestep_for_year() {
	local year=$1
	local step since points

	if [[ -n "${restartTimestep:-}" && "$yearFrom" -eq "$yearTo" ]]; then
		echo "$restartTimestep"
		return 0
	fi

	step=$(restart_step_for_year "$year")
	if is_uint "$step"; then
		echo "$step"
		return 0
	fi

	since=$(( year - yearStart + 1 ))
	[[ $since -ge 0 ]] || return 1
	points=$(( pointsPerYear * since ))
	printf "%08d" "$points"
}

keep_restart_year() {
	local year=$1
	local since frequency

	[[ "$year" -eq "$yearEnd" ]] && return 0
	if [[ "$year" -lt "$spinupEnd" ]]; then
		since=$(( year - spinupStart ))
		frequency=$spinupRestartKeepFrequency
	else
		since=$(( year - spinupEnd ))
		frequency=$runRestartKeepFrequency
	fi

	is_uint "$frequency" && [[ "$frequency" -gt 0 ]] || return 0
	[[ $(( since % frequency )) -eq 0 ]]
}

keep_output_year() {
	local year=$1
	local since frequency

	[[ "$year" -eq "$yearEnd" ]] && return 0
	if [[ "$year" -lt "$spinupEnd" ]]; then
		since=$(( year - spinupStart ))
		frequency=$spinupOutputKeepFrequency
	else
		since=$(( year - spinupEnd ))
		frequency=$runOutputKeepFrequency
	fi

	is_uint "$frequency" && [[ "$frequency" -gt 0 ]] || return 0
	[[ $(( since % frequency )) -eq 0 ]]
}

model_id=$(basename "$PWD")
id=${model_id: -4}
echo "Tidying up year: $1 to $2 for $model_id"

# Specify directory for copying files to central area
afm_dir="/gpfs/afm/greenocean/software/runs/"

# Make directory for run in central area
if [ ! -d $afm_dir$model_id ]; then
	echo "creating directory"
	mkdir $afm_dir$model_id
fi

# Per-run analyser config snapshot copied into the run dir by setUpRun
# (the grid-specific config the run was set up with).
analyserConfig=analyser_config.toml

# Per-year processing: copy every output and restart to AFM, leave a scratch symlink.
# Selective pruning by keep-frequency happens at end of simulation below.
echo "Copying data centrally"
for (( y=$yearFrom; y<=$yearTo; y++ )); do

	# Compute AMOC from grid_V using CDFtools (before analyser so moc file is ready)
	grid_v_file="ORCA2_${freq}_${y}0101_${y}1231_grid_V.nc"
	grid_t_file="ORCA2_${freq}_${y}0101_${y}1231_grid_T.nc"
	if [ -f "$grid_v_file" ]; then
		bash compute_amoc.sh "$grid_v_file" "$grid_t_file"
	fi

	# Python based totalling
	echo "Python version: $(which python)"
	python3 analyser.py "$analyserConfig" ${y} ${y}

	# Run timeseries visualization script
	python3 make_timeseries.py $model_id --model-run-dir $modelOutputDir

	echo "Compressing outputs $y"
	[[ $keepGrid_T -eq 1 ]] && compress_nc "ORCA2_${freq}_${y}0101_${y}1231_grid_T.nc"
	[[ $keepDiad -eq 1 ]]   && compress_nc "ORCA2_${freq}_${y}0101_${y}1231_diad_T.nc"
	[[ $keepPtrc -eq 1 ]]   && compress_nc "ORCA2_${freq}_${y}0101_${y}1231_ptrc_T.nc"
	[[ $keepIce -eq 1 ]]    && compress_nc "ORCA2_${freq}_${y}0101_${y}1231_icemod.nc"
	[[ $keepGrid_U -eq 1 ]] && compress_nc "ORCA2_${freq}_${y}0101_${y}1231_grid_U.nc"
	[[ $keepGrid_V -eq 1 ]] && compress_nc "ORCA2_${freq}_${y}0101_${y}1231_grid_V.nc"
	[[ $keepGrid_W -eq 1 ]] && compress_nc "ORCA2_${freq}_${y}0101_${y}1231_grid_W.nc"
	[[ $keepLimPhy -eq 1 ]] && compress_nc "ORCA2_${freq}_${y}0101_${y}1231_limphy.nc"
	[[ $keepGflux -eq 1 ]]  && compress_nc "ORCA2_${freq}_${y}0101_${y}1231_gflux_T.nc"

	echo "Copying output $y"
	archive_output "$keepGrid_T" "ORCA2_${freq}_${y}0101_${y}1231_grid_T.nc"
	archive_output "$keepDiad"   "ORCA2_${freq}_${y}0101_${y}1231_diad_T.nc"
	archive_output "$keepPtrc"   "ORCA2_${freq}_${y}0101_${y}1231_ptrc_T.nc"
	archive_output "$keepIce"    "ORCA2_${freq}_${y}0101_${y}1231_icemod.nc"
	archive_output "$keepGrid_U" "ORCA2_${freq}_${y}0101_${y}1231_grid_U.nc"
	archive_output "$keepGrid_V" "ORCA2_${freq}_${y}0101_${y}1231_grid_V.nc"
	archive_output "$keepGrid_W" "ORCA2_${freq}_${y}0101_${y}1231_grid_W.nc"
	archive_output "$keepLimPhy" "ORCA2_${freq}_${y}0101_${y}1231_limphy.nc"
	archive_output "$keepGflux"  "ORCA2_${freq}_${y}0101_${y}1231_gflux_T.nc"

	# Copy MOC file if it exists
	if [[ -f MOC/moc_${y}.nc ]]; then
		mkdir -p $afm_dir$model_id/MOC
		cp MOC/moc_${y}.nc $afm_dir$model_id/MOC/
	fi

	echo "Copying extra set up data and EMPave files"
	cp EMPave_${y}.dat $afm_dir$model_id
	cp namelist* $afm_dir$model_id
	cp *xml $afm_dir$model_id
	cp setUpData*dat $afm_dir$model_id
	cp ocean.output $afm_dir$model_id
	cp opa $afm_dir$model_id

	echo "Deleting local data if it exists centrally"
	delete_local_output "$keepGrid_T" "ORCA2_${freq}_${y}0101_${y}1231_grid_T.nc"
	delete_local_output "$keepDiad"   "ORCA2_${freq}_${y}0101_${y}1231_diad_T.nc"
	delete_local_output "$keepPtrc"   "ORCA2_${freq}_${y}0101_${y}1231_ptrc_T.nc"
	delete_local_output "$keepIce"    "ORCA2_${freq}_${y}0101_${y}1231_icemod.nc"
	delete_local_output "$keepGrid_U" "ORCA2_${freq}_${y}0101_${y}1231_grid_U.nc"
	delete_local_output "$keepGrid_V" "ORCA2_${freq}_${y}0101_${y}1231_grid_V.nc"
	delete_local_output "$keepGrid_W" "ORCA2_${freq}_${y}0101_${y}1231_grid_W.nc"
	delete_local_output "$keepLimPhy" "ORCA2_${freq}_${y}0101_${y}1231_limphy.nc"
	delete_local_output "$keepGflux"  "ORCA2_${freq}_${y}0101_${y}1231_gflux_T.nc"

	echo "Creating symlinks for copied data"
	link_archived_output "$keepGrid_T" "ORCA2_${freq}_${y}0101_${y}1231_grid_T.nc"
	link_archived_output "$keepDiad"   "ORCA2_${freq}_${y}0101_${y}1231_diad_T.nc"
	link_archived_output "$keepPtrc"   "ORCA2_${freq}_${y}0101_${y}1231_ptrc_T.nc"
	link_archived_output "$keepIce"    "ORCA2_${freq}_${y}0101_${y}1231_icemod.nc"
	link_archived_output "$keepGrid_U" "ORCA2_${freq}_${y}0101_${y}1231_grid_U.nc"
	link_archived_output "$keepGrid_V" "ORCA2_${freq}_${y}0101_${y}1231_grid_V.nc"
	link_archived_output "$keepGrid_W" "ORCA2_${freq}_${y}0101_${y}1231_grid_W.nc"
	link_archived_output "$keepLimPhy" "ORCA2_${freq}_${y}0101_${y}1231_limphy.nc"
	link_archived_output "$keepGflux"  "ORCA2_${freq}_${y}0101_${y}1231_gflux_T.nc"

	# Restart files are written at the end of output year y.
	if ! timestep=$(restart_timestep_for_year "$y"); then
		echo "WARNING: could not resolve restart timestep for year $y"
		continue
	fi

	echo "Copying restart $y (timestep $timestep)"
	if ls ORCA2_*${timestep}_restart_*.nc >/dev/null 2>&1; then
		cp ORCA2_*${timestep}_restart_*.nc $afm_dir$model_id
		rm -f ORCA2_*${timestep}_restart_*.nc
		ls -1 ${afm_dir}${model_id}/ORCA2_*${timestep}_restart_*.nc | awk '{print "ln -s "$1 }' | bash
	else
		echo "WARNING: no restart files found for timestep $timestep"
	fi
done

# Sort CSV files by year to ensure correct ordering
echo "Sorting analyser CSV files by year"
for csv in analyser/analyser.*.annual.csv; do
	if [[ -f "$csv" ]]; then
		(head -n 1 "$csv" && tail -n +2 "$csv" | sort -t',' -k1,1n) > "${csv}.sorted"
		mv "${csv}.sorted" "$csv"
	fi
done

# Commands to execute at the end of the simulation (final year only)
if [[ $yearTo -eq $yearEnd ]]; then

	# Prune non-keep restart and output files from scratch + AFM based on keep frequency.
	# Per-year tidyup copies every restart/output to AFM (for crash recovery) and
	# leaves scratch symlinks pointing at AFM; this is where the selective-saving
	# spec from setUpData is actually enforced.
	echo "Pruning non-keep restart files (scratch + AFM)"
	RESTART_STEPS=$(ls ORCA2_*_restart*_0000.nc ${afm_dir}${model_id}/ORCA2_*_restart*_0000.nc 2>/dev/null | sed -n 's/.*ORCA2_\([0-9][0-9]*\)_restart.*/\1/p' | sort -u)
	for yy_ts in $RESTART_STEPS; do
		restart_file=$(find_restart_file_for_step "$yy_ts")
		yy=$(restart_year_from_file "$restart_file")
		if ! is_uint "$yy"; then
			yy_num=$((10#$yy_ts))
			[[ $((yy_num % pointsPerYear)) -eq 0 ]] || continue
			yy=$(( yearStart + (yy_num / pointsPerYear) - 1 ))
		fi

		keep_restart_year "$yy" && continue

		echo "Removing restart year $yy (timestep $yy_ts)"
		rm -f ${afm_dir}${model_id}/ORCA2_${yy_ts}_restart*.nc
		rm -f ORCA2_${yy_ts}_restart*.nc
	done

	echo "Pruning non-keep output files (scratch + AFM)"
	for (( yy=$yearStart; yy<=$yearEnd; yy++ )); do
		keep_output_year "$yy" && continue

		echo "Removing output year $yy"
		for ftype in grid_T diad_T ptrc_T icemod grid_U grid_V grid_W limphy gflux_T; do
			rm -f ${afm_dir}${model_id}/ORCA2_${freq}_${yy}0101_${yy}1231_${ftype}.nc
			rm -f ORCA2_${freq}_${yy}0101_${yy}1231_${ftype}.nc
		done
	done

	# copy all analyser files to base model directory
	echo "copying analyser files"
	cp analyser* $afm_dir$model_id

	# copy all MOC files to base model directory
	if [ -d MOC ] && ls MOC/moc_*.nc 1>/dev/null 2>&1; then
		echo "copying MOC files"
		mkdir -p $afm_dir$model_id/MOC
		cp MOC/moc_*.nc $afm_dir$model_id/MOC/
	fi

	# run script to generate monthly regional plots
	python3 make_monthly_plots.py --model-id $model_id --model-dir $modelOutputDir

	# generate complete visualization suite (maps + transects)
	python3 visualise_model.py $model_id $yearTo --model-run-dir $modelOutputDir --output-dir $modelOutputDir/monitor/$model_id/

	# run script to create html file
	./make_html.sh $model_id $modelOutputDir
fi
