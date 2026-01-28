#!/bin/sh

date
echo "To use: setUpRun <setUpData.dat> <Full Run ID> [SPINUP_MODEL_ID]"

if [ "$#" -lt 2 ] || [ "$#" -gt 3 ]; then
	exit 1
fi

# Detect the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Input variables read from command line
setUpDatafile=$1
id=$2
spinupModelId=$3

# If setUpDatafile doesn't exist as-is, try looking in configs/ directory
if [ ! -f "$setUpDatafile" ]; then
	if [ -f "${SCRIPT_DIR}/configs/$(basename $setUpDatafile)" ]; then
		setUpDatafile="${SCRIPT_DIR}/configs/$(basename $setUpDatafile)"
		echo "Using config file: $setUpDatafile"
	fi
fi

# ----- Meta variables -----
version=$(echo $id | awk -F'_' '{print $1}')
initials=$(echo $id | awk -F'_' '{print $2}')
simulation=$(echo $id | awk -F'_' '{print $3}')

echo "Setting up:                  " $version $initials $simulation
echo "Reading run parameters from: " $setUpDatafile

# ----- Read setup data -----
# Check if setUpDatafile is already an absolute path
if [[ "$setUpDatafile" = /* ]]; then
	dataFileFullPath=$setUpDatafile
else
	dataFileFullPath=$(pwd)"/"$setUpDatafile
fi

while IFS= read -r line; do
	if [ ! ${line:0:1} == "#" ]; then

		# Pre-link processing
		name=$(echo $line | awk -F':' '{print $1}')
		val=$(echo $line | awk -F':' '{print $2}')
        
		if [[ $name != *"."* && $name != "namelist"* ]]; then
			if [[ $name == "yearStart" ]]; then yearStart=$val; fi
			if [[ $name == "yearEnd" ]]; then yearEnd=$val; fi
			if [[ $name == "CO2" ]]; then CO2=$val; fi
			if [[ $name == "forcing" ]]; then forcing=$val; fi
			if [[ $name == "basedir" ]]; then basedir=$val; fi
			if [[ $name == "EMPaveFile" ]]; then EMPaveFile=$val; fi
			if [[ $name == "model" ]]; then Model=$val; fi
			if [[ $name == "forcing_mode" ]]; then forcing_mode=$val; fi
			if [[ $name == "compilerKey" ]]; then compKey=$val; fi

			# Tidy up parameters
			if [[ $name == "spinupStart" ]]; then spinupStart=$val; fi
			if [[ $name == "spinupEnd" ]]; then spinupEnd=$val; fi
			if [[ $name == "spinupRestartKeepFrequency" ]]; then spinupRestartKeepFrequency=$val; fi
			if [[ $name == "spinupOutputKeepFrequency" ]]; then spinupOutputKeepFrequency=$val; fi
			if [[ $name == "runRestartKeepFrequency" ]]; then runRestartKeepFrequency=$val; fi
			if [[ $name == "runOutputKeepFrequency" ]]; then runOutputKeepFrequency=$val; fi
			if [[ $name == "keepGrid_T" ]]; then keepGrid_T=$val; fi
			if [[ $name == "keepDiad" ]]; then keepDiad=$val; fi
			if [[ $name == "keepPtrc" ]]; then keepPtrc=$val; fi
			if [[ $name == "keepIce" ]]; then keepIce=$val; fi
			if [[ $name == "keepGrid_U" ]]; then keepGrid_U=$val; fi
			if [[ $name == "keepGrid_V" ]]; then keepGrid_V=$val; fi
			if [[ $name == "keepGrid_W" ]]; then keepGrid_W=$val; fi
			if [[ $name == "keepLimPhy" ]]; then keepLimPhy=$val; fi
		fi
	fi
done < $dataFileFullPath

prevYear=$(($yearStart-1))

# ----- Move to or create model directory -----
# Adjust for a possible ~ expansion problem
if [ ${basedir:0:1} == "~" ]; then 
	homearea=$(readlink -f ~)
	basedir=$homearea${basedir:1:${#basedir}-1}
fi

modelDir=$basedir$id
if [ ! -d $modelDir ]; then
	mkdir $modelDir
fi

# Copy the setUpData file to the directory
cp $setUpDatafile $modelDir
cd $modelDir

# ----- Output tidy up parameters -----
echo $spinupStart $spinupEnd $spinupRestartKeepFrequency $spinupOutputKeepFrequency $runRestartKeepFrequency $runOutputKeepFrequency $keepGrid_T $keepDiad $keepPtrc $keepIce $keepGrid_V $keepGrid_U $keepGrid_W $keepLimPhy > tidy_parms

# ----- Create links -----
rm -f opa

while IFS= read -r line; do
	if [ ! ${line:0:1} == "#" ]; then
		
		# Pre-link processing
		name=$(echo $line | awk -F':' '{print $1}')
		val=$(echo $line | awk -F':' '{print $2}')

		# Make links for all .nc and xml files 
		if [[ $name == *"."* && $name != "namelist"* && $name != *".xml" ]]; then 

			if [[ $name == "restart"* ]]; then 
				# Only create link if restart file does not exist already (i.e. run already started in folder)
				if [ ! -f restart_0000.nc ]; then
					ln -fs $val $name
				fi
			else
				ln -fs $val $name
			fi
        	fi

		# Copy namelists in a way so changes can be made
		# Skip namelists that don't match the selected forcing
		if [[ $name == "namelist"* ]]; then
			skip=false
			if [[ $name == namelist_ref_era_* && $forcing != "ERA" ]]; then
				skip=true
			elif [[ $name == namelist_ref_jra_* && $forcing != "JRA" ]]; then
				skip=true
			elif [[ $name == namelist_ref_ncep_* && $forcing != "NCEP" ]]; then
				skip=true
			fi

			if [ "$skip" = true ]; then
				echo "Skipping $name (forcing is $forcing)"
			elif [ -f $name ]; then
				echo $name " exists so no fresh copy made "
			else
				cp $val $name
			fi
		fi
		
		# Copy xml files in a way so changes can be made
		if [[ $name == *".xml" ]]; then
			if [ -f $name ]; then
				echo $name " exists so no fresh copy made "
			else
				cp $val $name
			fi
		fi

		# Copy the executable over, good to keep these.
		if [[ $name == "opa"*$Model ]]; then
			echo "copying " $val $name " for " $Model
			cp $val $name
			ln -s $name opa
		fi
	fi
done < $dataFileFullPath

# Link EMP file
if [ ! -f EMPave_${prevYear}.dat ]; then
	rm -f EMPave_${prevYear}.dat
	ln -fs $EMPaveFile EMPave_${prevYear}.dat
	ln -fs EMPave_${prevYear}.dat EMPave_old.dat
else
	echo "EMPave exists, using existing file!!!"
	ln -fs EMPave_${prevYear}.dat EMPave_old.dat
fi

# ----- Check compiler keys -----
cp $compKey .

grep key_trc_piic $compKey > tmp
if [ -s tmp ]; then
	PIIC=piic
fi

grep key_c14b $compKey > tmp
if [ -s tmp ]; then
	C14=c14
fi

rm tmp
echo ${PIIC^^} ${C14^^}

# ----- Process flags -----
# CO2
rm -f atmco2.dat

echo $CO2
if [ $CO2 == "VARIABLE" ]; then
	ln -s atmco2.dat.variable atmco2.dat
else
	ln -s atmco2.dat.static atmco2.dat
fi

# Forcing
rm -f namelist_ref

echo $forcing
if [ $forcing == "NCEP" ]; then
	forcing_prefix="ncep"
elif [ $forcing == "ERA" ]; then
	forcing_prefix="era"
else
	forcing_prefix="jra"
fi

# Layer 1: Functional symlinks (abstract forcing type)
# - coldstart: nn_rstctl=0, uses nn_date0 for start date
# - restart:   nn_rstctl=2, reads date from restart file, historical forcing
# - cycling:   nn_rstctl=2, reads date from restart file, loops single year forcing
ln -sf namelist_ref_${forcing_prefix}_coldstart namelist_ref_coldstart
ln -sf namelist_ref_${forcing_prefix}_restart namelist_ref_restart
ln -sf namelist_ref_${forcing_prefix}_cycling namelist_ref_cycling

# Automatically correct nn_date0 in namelist_ref_coldstart to match yearStart from setup data
expectedDate="${yearStart}0101"
currentDate=$( grep "nn_date0" namelist_ref_coldstart | head -1 | awk -F'=' '{print $2}' | awk '{print $1}' )

if [ "$currentDate" != "$expectedDate" ]; then
	echo -e "\e[1;33mNOTE\e[0m: Updating nn_date0 from $currentDate to $expectedDate to match yearStart"
	sed -i "s/nn_date0.*=.*/nn_date0    = $expectedDate/" namelist_ref_coldstart
fi

# Layer 2: Temporal symlinks (when each is used)
# - first_year:  always uses coldstart
# - other_years: uses cycling (spinup) or restart (transient)
echo $forcing_mode
ln -sf namelist_ref_coldstart namelist_ref_first_year
if [ $forcing_mode == "spinup" ]; then
	ln -sf namelist_ref_cycling namelist_ref_other_years
else
	ln -sf namelist_ref_restart namelist_ref_other_years
fi

# Layer 3: Final symlink (based on restart file existence)
# Scenarios:
#   | Setup method       | Forcing mode | Has restart | Result   |
#   |--------------------|--------------|-------------|----------|
#   | Fresh              | spinup      | No          | coldstart|
#   | Fresh              | transient    | No          | coldstart|
#   | Continued          | spinup      | Yes         | cycling  |
#   | Continued          | transient    | Yes         | restart  |
#   | From spinup (*)    | spinup      | Yes         | cycling  |
#   | From spinup (*)    | transient    | Yes         | restart  |
#   (*) setup_spin.sh copies restart files then switches namelist_ref -> other_years
#
if [ ! -f restart_0000.nc ]; then
	ln -s namelist_ref_first_year namelist_ref
else
	ln -s namelist_ref_other_years namelist_ref
fi

# Temperature and salinity restoring
TR=$( grep "nn_sstr " namelist_ref | awk -F' ' '{print $3}' )
SR=$( grep "nn_sssr " namelist_ref | awk -F' ' '{print $3}' )
LP=$( grep "ln_lop" namelist_top_ref | awk -F' ' '{print $3}' )

if [ $TR = 1 ]; then
	echo "TEMPERATURE RESTORING"
else
	echo "NO TEMPERATURE RESTORING"
fi

if [ $SR = 1 ]; then
	echo "SALINITY RESTORING"
else
	echo "NO SALINITY RESTORING"
fi

# Check that files for LIMPHY are set correctly
IODEF_PATH=$( grep "^iodef.xml:" $setUpDatafile | awk -F':' '{print $2}' )
KP=$( grep "^keepLimPhy:" $setUpDatafile | awk -F':' '{print $NF}' )
err=0

if [ $LP = ".true." ]; then
	echo "LIMPHY TRUE"

	if [ $KP != 1 ]; then
		echo -e "\e[1;31mWARNING\e[0m : KEEP value for LimPhy not set to 1"
		err=1
	fi
else
	echo "LIMPHY FALSE"

	if [ $KP != 0 ]; then
		echo -e "\e[1;31mWARNING\e[0m : KEEP value for LimPhy not set to 0"
		err=1
	fi
fi

# Check iodef file exists
if [ ! -f "$IODEF_PATH" ]; then
	echo -e "\e[1;31mWARNING\e[0m : IODEF file does not exist: $IODEF_PATH"
	err=1
fi

if [ $err == 1 ]; then
	exit 2
fi

# Get code version
codeVersion=$( grep "opa_*$Model" $setUpDatafile | awk -F'/' '{print$(NF-5)}' )

# ----- Create copies of files used for run -----
# Get NEMO job files
if [ ! -f nemo.job ]; then
	cp ${SCRIPT_DIR}/nemo.job nemo.job
fi
if [ ! -f nemo_compute.job ]; then
	cp ${SCRIPT_DIR}/nemo_compute.job nemo_compute.job
fi

# Get tidying up scripts
ln -fs ${SCRIPT_DIR}/tidyup.sh tidyup.sh
ln -fs ${SCRIPT_DIR}/tidyup.job tidyup.job

# Get analyser scripts
cp ${SCRIPT_DIR}/analyser/analyser*.py .
cp ${SCRIPT_DIR}/analyser/analyser_config.toml .
ln -fs ${SCRIPT_DIR}/shared shared
if [ -f ${SCRIPT_DIR}/iodef_tom12piicc14.xml ]; then
	cp ${SCRIPT_DIR}/iodef_tom12piicc14.xml .
fi

# If analyser_config.toml does not exist (as specified in setUpData file) copy in default
if [ ! -f analyser_config.toml ]; then
	echo "Copying in default script for analyser_config.toml"
	cp ${SCRIPT_DIR}/analyser/analyser_config.toml analyser_config.toml
fi

# Get visualise scripts and files
for file in ${SCRIPT_DIR}/visualise/*; do
	ln -fs $file $(basename $file)
done

# Save parameters needed for creating html file
echo $id $codeVersion $(date '+%d-%b-%Y') $yearStart $yearEnd ${CO2,,} $forcing ${forcing_mode,,} $TR $SR > html_parms

# Get setUpRun script
cp ${SCRIPT_DIR}/setUpRun.sh .

# ----- Setup from spinup model (if provided) -----
if [ -n "$spinupModelId" ]; then
	bash ${SCRIPT_DIR}/setup_spin.sh $id $spinupModelId
	if [ $? -ne 0 ]; then
		echo "Error: Spinup setup failed"
		exit 1
	fi
fi

# ----- Export parameters the nemo.job file will need -----
yearToRun=$yearStart

echo "Exporting " $yearToRun $yearEnd $basedir $modelDir $simulation $Model $forcing_prefix $forcing_mode
export yearToRun yearStart yearEnd basedir modelDir simulation Model forcing_prefix forcing_mode

read -p "Press any key to run it? (cntr+c otherwise)"

# Auto-select job file: use compute if 2+ jobs already on ib
ib_jobs=$(squeue -p ib -u $USER -h 2>/dev/null | wc -l)
if [ "$ib_jobs" -ge 2 ]; then
	echo "IB partition has $ib_jobs jobs, using compute partition"
	sbatch -J${simulation}${yearToRun} < nemo_compute.job
else
	echo "IB partition has $ib_jobs jobs, using ib partition"
	sbatch -J${simulation}${yearToRun} < nemo.job
fi

echo "To check status of job 'squeue | grep <you user name>' "

# ----- Save model details -----
echo $2 "("$(date '+%a %d %b %T %Z %Y')")" >> ${HOME}/scratch/ModelRuns/modelRuns.txt
