#!/bin/sh

# parameters for tidying up
yearFrom=$1
yearTo=$2
version=$3
modelOutputDir=$4

# Get parameters as specified in setUpData.dat file
read -r -a parms < tidy_parms

spinupStart=${parms[0]}
spinupEnd=${parms[1]}
spinupRestartKeepFrequency=${parms[2]}
spinupOutputKeepFrequency=${parms[3]}
runRestartKeepFrequency=${parms[4]}
runOutputKeepFrequency=${parms[5]}
keepGrid_T=${parms[6]}
keepDiad=${parms[7]}
keepPtrc=${parms[8]}
keepIce=${parms[9]}
keepGrid_V=${parms[10]}
keepGrid_U=${parms[11]}
keepGrid_W=${parms[12]}
keepLimPhy=${parms[13]}
keepGflux=${parms[14]:-0}

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

pointsPerYear=5475
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

# Per-run analyser config (e.g. NEMO5 vs NEMO3.6 grid). Use the named .toml
# only if it is present in the run dir; otherwise fall back to the canonical
# analyser_config.toml symlink set up by setUpRun (which points at the right
# variant already).
analyserConfig=$(grep -h "^analyser_config:" setUpData*dat 2>/dev/null | head -1 | awk -F':' '{print $2}')
{ [ -n "$analyserConfig" ] && [ -f "$analyserConfig" ]; } || analyserConfig=analyser_config.toml

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
	if [[ $keepGrid_T -eq 1 ]]; then cp ORCA2_${freq}_${y}0101_${y}1231_grid_T.nc $afm_dir$model_id; fi
	if [[ $keepDiad -eq 1 ]];   then cp ORCA2_${freq}_${y}0101_${y}1231_diad_T.nc $afm_dir$model_id; fi
	if [[ $keepPtrc -eq 1 ]];   then cp ORCA2_${freq}_${y}0101_${y}1231_ptrc_T.nc $afm_dir$model_id; fi
	if [[ $keepIce -eq 1 ]];    then cp ORCA2_${freq}_${y}0101_${y}1231_icemod.nc $afm_dir$model_id; fi
	if [[ $keepGrid_U -eq 1 ]]; then cp ORCA2_${freq}_${y}0101_${y}1231_grid_U.nc $afm_dir$model_id; fi
	if [[ $keepGrid_V -eq 1 ]]; then cp ORCA2_${freq}_${y}0101_${y}1231_grid_V.nc $afm_dir$model_id; fi
	if [[ $keepGrid_W -eq 1 ]]; then cp ORCA2_${freq}_${y}0101_${y}1231_grid_W.nc $afm_dir$model_id; fi
	if [[ $keepLimPhy -eq 1 ]]; then cp ORCA2_${freq}_${y}0101_${y}1231_limphy.nc $afm_dir$model_id; fi
	if [[ $keepGflux -eq 1 ]];  then cp ORCA2_${freq}_${y}0101_${y}1231_gflux_T.nc $afm_dir$model_id; fi

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
	if [[ -f $afm_dir$model_id/ORCA2_${freq}_${y}0101_${y}1231_grid_T.nc || $keepGrid_T -eq 0 ]]; then rm -f ORCA2_${freq}_${y}0101_${y}1231_grid_T.nc; fi
	if [[ -f $afm_dir$model_id/ORCA2_${freq}_${y}0101_${y}1231_diad_T.nc || $keepDiad -eq 0 ]];   then rm -f ORCA2_${freq}_${y}0101_${y}1231_diad_T.nc; fi
	if [[ -f $afm_dir$model_id/ORCA2_${freq}_${y}0101_${y}1231_ptrc_T.nc || $keepPtrc -eq 0 ]];   then rm -f ORCA2_${freq}_${y}0101_${y}1231_ptrc_T.nc; fi
	if [[ -f $afm_dir$model_id/ORCA2_${freq}_${y}0101_${y}1231_icemod.nc || $keepIce -eq 0 ]];    then rm -f ORCA2_${freq}_${y}0101_${y}1231_icemod.nc; fi
	if [[ -f $afm_dir$model_id/ORCA2_${freq}_${y}0101_${y}1231_grid_U.nc || $keepGrid_U -eq 0 ]]; then rm -f ORCA2_${freq}_${y}0101_${y}1231_grid_U.nc; fi
	if [[ -f $afm_dir$model_id/ORCA2_${freq}_${y}0101_${y}1231_grid_V.nc || $keepGrid_V -eq 0 ]]; then rm -f ORCA2_${freq}_${y}0101_${y}1231_grid_V.nc; fi
	if [[ -f $afm_dir$model_id/ORCA2_${freq}_${y}0101_${y}1231_grid_W.nc || $keepGrid_W -eq 0 ]]; then rm -f ORCA2_${freq}_${y}0101_${y}1231_grid_W.nc; fi
	if [[ -f $afm_dir$model_id/ORCA2_${freq}_${y}0101_${y}1231_limphy.nc || $keepLimPhy -eq 0 ]]; then rm -f ORCA2_${freq}_${y}0101_${y}1231_limphy.nc; fi
	if [[ -f $afm_dir$model_id/ORCA2_${freq}_${y}0101_${y}1231_gflux_T.nc || $keepGflux -eq 0 ]];  then rm -f ORCA2_${freq}_${y}0101_${y}1231_gflux_T.nc; fi

	echo "Creating symlinks for copied data"
	if [[ $keepGrid_T -eq 1 ]]; then ln -s ${afm_dir}${model_id}/ORCA2_${freq}_${y}0101_${y}1231_grid_T.nc; fi
	if [[ $keepDiad -eq 1 ]];   then ln -s ${afm_dir}${model_id}/ORCA2_${freq}_${y}0101_${y}1231_diad_T.nc; fi
	if [[ $keepPtrc -eq 1 ]];   then ln -s ${afm_dir}${model_id}/ORCA2_${freq}_${y}0101_${y}1231_ptrc_T.nc; fi
	if [[ $keepIce -eq 1 ]];    then ln -s ${afm_dir}${model_id}/ORCA2_${freq}_${y}0101_${y}1231_icemod.nc; fi
	if [[ $keepGrid_U -eq 1 ]]; then ln -s ${afm_dir}${model_id}/ORCA2_${freq}_${y}0101_${y}1231_grid_U.nc; fi
	if [[ $keepGrid_V -eq 1 ]]; then ln -s ${afm_dir}${model_id}/ORCA2_${freq}_${y}0101_${y}1231_grid_V.nc; fi
	if [[ $keepGrid_W -eq 1 ]]; then ln -s ${afm_dir}${model_id}/ORCA2_${freq}_${y}0101_${y}1231_grid_W.nc; fi
	if [[ $keepLimPhy -eq 1 ]]; then ln -s ${afm_dir}${model_id}/ORCA2_${freq}_${y}0101_${y}1231_limphy.nc; fi
	if [[ $keepGflux -eq 1 ]];  then ln -s ${afm_dir}${model_id}/ORCA2_${freq}_${y}0101_${y}1231_gflux_T.nc; fi

	# Restart files: timestep is end of year (y-1) = start of year y
	since=$(( y-yearStart ))
	points=$(( pointsPerYear * since ))
	printf -v timestep "%08d" $points

	echo "Copying restart $y (timestep $timestep)"
	cp ORCA2_*${timestep}_restart_*.nc $afm_dir$model_id
	rm -f ORCA2_*${timestep}_restart_*.nc
	ls -1 ${afm_dir}${model_id}/ORCA2_*${timestep}_restart_*.nc | awk '{print "ln -s "$1 }' | bash
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
	# spec from tidy_parms is actually enforced.
	echo "Pruning non-keep restart files (scratch + AFM)"
	for (( yy=$yearStart; yy<=$yearEnd; yy++ )); do
		if [[ $yy -lt $spinupEnd ]]; then
			yy_since=$(( yy - spinupStart ))
			yy_remainder=$(( yy_since % spinupRestartKeepFrequency ))
		else
			yy_since=$(( yy - spinupEnd ))
			yy_remainder=$(( yy_since % runRestartKeepFrequency ))
		fi

		# Keep frequency-matching years and always preserve the final year
		if [[ $yy_remainder -eq 0 || $yy -eq $yearEnd ]]; then
			continue
		fi

		yy_offset=$(( yy - yearStart ))
		yy_points=$(( pointsPerYear * yy_offset ))
		printf -v yy_ts "%08d" $yy_points

		echo "Removing restart year $yy (timestep $yy_ts)"
		rm -f ${afm_dir}${model_id}/ORCA2_*${yy_ts}_restart_*.nc
		rm -f ORCA2_*${yy_ts}_restart_*.nc
	done

	echo "Pruning non-keep output files (scratch + AFM)"
	for (( yy=$yearStart; yy<=$yearEnd; yy++ )); do
		if [[ $yy -lt $spinupEnd ]]; then
			yy_since=$(( yy - spinupStart ))
			yy_remainder=$(( yy_since % spinupOutputKeepFrequency ))
		else
			yy_since=$(( yy - spinupEnd ))
			yy_remainder=$(( yy_since % runOutputKeepFrequency ))
		fi

		if [[ $yy_remainder -eq 0 || $yy -eq $yearEnd ]]; then
			continue
		fi

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

