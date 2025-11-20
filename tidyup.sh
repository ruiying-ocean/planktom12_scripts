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

pointsPerYear=5475
freq=1m

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

echo "Copying data centrally"
for (( y=$yearFrom; y<=$yearTo; y++ )); do

         # Python based totalling
         echo "Python version: $(which python)"
	python3 breakdown.py breakdown_config.toml ${y} ${y}

	# Run timeseries visualization script
	python3 make_timeseries.py $model_id --model-run-dir $modelOutputDir

	# Process output files
	if [[ $y < $spinupEnd ]]; then
		# years from start
		since=$((y-spinupStart))
		remainder=$(( since % spinupOutputKeepFrequency ))
	else
        	since=$((y-spinupEnd))
		remainder=$(( since % runOutputKeepFrequency ))
	fi
    
	echo $remainder $since

	if [[ "$remainder" -eq 0 ]]; then

		echo "Copying output $y"
		if [[ $keepGrid_T -eq 1 ]]; then cp ORCA2_${freq}_${y}0101_${y}1231_grid_T.nc $afm_dir$model_id; fi
		if [[ $keepDiad -eq 1 ]];   then cp ORCA2_${freq}_${y}0101_${y}1231_diad_T.nc $afm_dir$model_id; fi
		if [[ $keepPtrc -eq 1 ]];   then cp ORCA2_${freq}_${y}0101_${y}1231_ptrc_T.nc $afm_dir$model_id; fi
		if [[ $keepIce -eq 1 ]];    then cp ORCA2_${freq}_${y}0101_${y}1231_icemod.nc $afm_dir$model_id; fi
		if [[ $keepGrid_U -eq 1 ]]; then cp ORCA2_${freq}_${y}0101_${y}1231_grid_U.nc $afm_dir$model_id; fi
		if [[ $keepGrid_V -eq 1 ]]; then cp ORCA2_${freq}_${y}0101_${y}1231_grid_V.nc $afm_dir$model_id; fi
		if [[ $keepGrid_W -eq 1 ]]; then cp ORCA2_${freq}_${y}0101_${y}1231_grid_W.nc $afm_dir$model_id; fi
		if [[ $keepLimPhy -eq 1 ]]; then cp ORCA2_${freq}_${y}0101_${y}1231_limphy.nc $afm_dir$model_id; fi

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

		echo "Creating symlinks for copied data"
		if [[ $keepGrid_T -eq 1 ]]; then ln -s ${afm_dir}${model_id}/ORCA2_${freq}_${y}0101_${y}1231_grid_T.nc; fi
		if [[ $keepDiad -eq 1 ]];   then ln -s ${afm_dir}${model_id}/ORCA2_${freq}_${y}0101_${y}1231_diad_T.nc; fi
		if [[ $keepPtrc -eq 1 ]];   then ln -s ${afm_dir}${model_id}/ORCA2_${freq}_${y}0101_${y}1231_ptrc_T.nc; fi
		if [[ $keepIce -eq 1 ]];    then ln -s ${afm_dir}${model_id}/ORCA2_${freq}_${y}0101_${y}1231_icemod.nc; fi
		if [[ $keepGrid_U -eq 1 ]]; then ln -s ${afm_dir}${model_id}/ORCA2_${freq}_${y}0101_${y}1231_grid_U.nc; fi
		if [[ $keepGrid_V -eq 1 ]]; then ln -s ${afm_dir}${model_id}/ORCA2_${freq}_${y}0101_${y}1231_grid_V.nc; fi
		if [[ $keepGrid_W -eq 1 ]]; then ln -s ${afm_dir}${model_id}/ORCA2_${freq}_${y}0101_${y}1231_grid_W.nc; fi
		if [[ $keepLimPhy -eq 1 ]]; then ln -s ${afm_dir}${model_id}/ORCA2_${freq}_${y}0101_${y}1231_limphy.nc; fi
	else
		rm -f ORCA2_${freq}_${y}0101_${y}1231_grid_*.nc
		rm -f ORCA2_${freq}_${y}0101_${y}1231_diad_T.nc
		rm -f ORCA2_${freq}_${y}0101_${y}1231_ptrc_T.nc
		rm -f ORCA2_${freq}_${y}0101_${y}1231_icemod.nc
		rm -f ORCA2_${freq}_${y}0101_${y}1231_limphy.nc
	fi

	# Process restart files
	if [[ $y < $spinupEnd ]]; then
		# Years from start
		since=$(( y-spinupStart ))
		remainder=$(( since % spinupRestartKeepFrequency ))
	else
		since=$(( y-spinupEnd ))
		remainder=$(( since % runRestartKeepFrequency ))
	fi

	# Reset since to correctly count points for linking and deleting purposes
	yr=$yearStart
	since=$(( y-yr ))

	# This is concerned with the previous year's restart files, no restarts are deleted until the next year is complete    
	points=$(( pointsPerYear * since ))
	printf -v timestep "%08d" $points
	
	echo $points $timestep $remainder $since

	if [[ "$remainder" -eq 0 ]]; then
		echo "Copying restart $y"
		cp ORCA2_*${timestep}_restart_*.nc $afm_dir$model_id
		rm -f ORCA2_*${timestep}_restart_*.nc
		ls -1 ${afm_dir}${model_id}/ORCA2_*${timestep}_restart_*.nc | awk '{print "ln -s "$1 }' | bash
	else
		echo "Deleting local data"
		rm -f ORCA2_*${timestep}_restart_*.nc
	fi
done

# Commands to execute at the end of the simulation (final year only)
if [[ $yearTo -eq $yearEnd ]]; then

	# copy all breakdown files to base model directory
	echo "copying breakdown files"
	cp breakdown* $afm_dir$model_id

	# run script to generate monthly regional plots
	python3 make_monthly_plots.py --model-id $model_id --model-dir $modelOutputDir

	# generate complete visualization suite (maps + transects)
	python3 visualise_model.py $model_id $yearTo --model-run-dir $modelOutputDir --output-dir $modelOutputDir/monitor/$model_id/

	# run script to create html file
	./make_html.sh $model_id $modelOutputDir
fi

