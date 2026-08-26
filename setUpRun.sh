#!/bin/bash

# Colors
BOLD='\e[1m'
GREEN='\e[32m'
CYAN='\e[36m'
YELLOW='\e[33m'
RED='\e[1;31m'
DIM='\e[2m'
RESET='\e[0m'

# Log helpers
ok()   { echo -e "  ${GREEN}✓${RESET} $1"; }
skip() { echo -e "  ${YELLOW}⊘${RESET} ${DIM}$1${RESET}"; }
warn() { echo -e "  ${RED}✗${RESET} $1"; }
info() { echo -e "  ${CYAN}→${RESET} $1"; }
section() { echo -e "\n${BOLD}$1${RESET}"; }

confirm_existing_model_dir() {
	local dir=$1
	local reply

	echo ""
	warn "Model directory already exists: $dir"
	echo "  Continuing will reuse this directory and may update setup/config files and links."
	printf 'Continue with existing model_id "%s"? [y/N] ' "$id"
	read -r reply
	case "$reply" in
		y|Y|yes|YES) ;;
		*) echo "aborted."; exit 1 ;;
	esac
}

if [ "$#" -lt 2 ] || [ "$#" -gt 3 ]; then
	echo "Usage: setUpRun <setUpData.dat> <Full Run ID> [SPINUP_MODEL_ID]"
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
	fi
fi

# ----- Meta variables -----
version=$(echo $id | awk -F'_' '{print $1}')
initials=$(echo $id | awk -F'_' '{print $2}')
simulation=$(echo $id | awk -F'_' '{print $3}')

echo ""
echo -e "${BOLD}┌──────────────────────────────────────────┐${RESET}"
echo -e "${BOLD}│${RESET}  NEMO Setup: ${CYAN}${version}_${initials}_${simulation}${RESET}"
echo -e "${BOLD}│${RESET}  $(date '+%a %d %b %Y %H:%M:%S')"
echo -e "${BOLD}│${RESET}  Config: ${DIM}$(basename $setUpDatafile)${RESET}"
echo -e "${BOLD}└──────────────────────────────────────────┘${RESET}"

# ----- Read setup data -----
# Check if setUpDatafile is already an absolute path
if [[ "$setUpDatafile" = /* ]]; then
	dataFileFullPath=$setUpDatafile
else
	dataFileFullPath=$(pwd)"/"$setUpDatafile
fi
[ -f "$dataFileFullPath" ] || { warn "Setup data not found: $dataFileFullPath"; exit 1; }
dataFileFullPath="$(cd "$(dirname "$dataFileFullPath")" && pwd)/$(basename "$dataFileFullPath")"
setUpDatafile=$dataFileFullPath

while IFS= read -r line || [ -n "$line" ]; do
	if [[ -n "$line" && ${line:0:1} != "#" ]]; then

		# Pre-link processing
		name=${line%%:*}
		val=${line#*:}

		if [[ $name != *"."* && $name != "namelist"* ]]; then
			if [[ $name == "yearStart" ]]; then yearStart=$val; fi
			if [[ $name == "yearEnd" ]]; then yearEnd=$val; fi
			if [[ $name == "timestep" ]]; then timestep=$val; fi
			if [[ $name == "stepsPerYear" ]]; then stepsPerYear=$val; fi
			if [[ $name == "CO2" ]]; then CO2=$val; fi
			if [[ $name == "forcing" ]]; then forcing=$val; fi
			if [[ $name == "basedir" ]]; then basedir=$val; fi
			if [[ $name == "archiveDir" ]]; then archiveDir=$val; fi
			if [[ $name == "outputFrequency" ]]; then outputFrequency=$val; fi
			if [[ $name == "EMPaveFile" ]]; then EMPaveFile=$val; fi
			if [[ $name == "model" ]]; then Model=$val; fi
			if [[ $name == "forcing_mode" ]]; then forcing_mode=$val; fi
			if [[ $name == "compilerKey" ]]; then compKey=$val; fi
			if [[ $name == "nemoVersion" ]]; then nemoVersion=$val; fi
			if [[ $name == "executable" ]]; then executable=$val; fi
			if [[ $name == "iceRestartName" ]]; then iceRestartName=$val; fi
			if [[ $name == "nemoCpus" ]]; then nemoCpus=$val; fi
			if [[ $name == "xiosCpus" ]]; then xiosCpus=$val; fi
			if [[ $name == "useXiosServer" ]]; then useXiosServer=$val; fi
			# Retention parameters
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
			if [[ $name == "keepGflux" ]]; then keepGflux=$val; fi
		fi
	fi
done < "$dataFileFullPath"

prevYear=$(($yearStart-1))
nemoVersion=${nemoVersion:-NEMO36}
nemoCpus=${nemoCpus:-48}
xiosCpus=${xiosCpus:-0}
useXiosServer=${useXiosServer:-false}
archiveDir=${archiveDir:-/gpfs/afm/greenocean/software/runs}
outputFrequency=${outputFrequency:-1m}
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
keepGrid_U=${keepGrid_U:-0}
keepGrid_V=${keepGrid_V:-0}
keepGrid_W=${keepGrid_W:-0}
keepLimPhy=${keepLimPhy:-0}
keepGflux=${keepGflux:-0}

if [[ "$nemoVersion" == "NEMO5" ]]; then
	executable=${executable:-nemo}
	iceRestartName=${iceRestartName:-restart_ice}
	if [ "$useXiosServer" = "false" ]; then useXiosServer=true; fi
	if [ "$xiosCpus" = "0" ]; then xiosCpus=12; fi
else
	executable=${executable:-opa}
	iceRestartName=${iceRestartName:-restart_ice_in}
fi

if [ -z "${stepsPerYear:-}" ]; then
	if [[ "$nemoVersion" == "NEMO5" ]]; then stepsPerYear=5840; else stepsPerYear=5475; fi
fi

# ----- Move to or create model directory -----
# Adjust for a possible ~ expansion problem
if [ ${basedir:0:1} == "~" ]; then
	homearea=$(readlink -f ~)
	basedir=$homearea${basedir:1:${#basedir}-1}
fi

modelDir=$basedir$id
if [ -d "$modelDir" ]; then
	confirm_existing_model_dir "$modelDir"
elif [ -e "$modelDir" ]; then
	warn "Model path exists but is not a directory: $modelDir"
	exit 1
else
	mkdir "$modelDir"
fi

# Copy the setUpData file to the directory
cp "$setUpDatafile" "$modelDir"
cd "$modelDir"

# Retention parameters are resolved once below into run.env for all workflow tasks.

# ----- Create links -----
rm -f opa nemo

section "Links & Files"

while IFS= read -r line || [ -n "$line" ]; do
	if [[ -n "$line" && ${line:0:1} != "#" ]]; then

		# Pre-link processing
		name=${line%%:*}
		val=${line#*:}

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
			elif [[ $name == namelist_cfg_era_* && $forcing != "ERA" ]]; then
				skip=true
			elif [[ $name == namelist_ref_jra_* && $forcing != "JRA" ]]; then
				skip=true
			elif [[ $name == namelist_cfg_jra_* && $forcing != "JRA" ]]; then
				skip=true
			elif [[ $name == namelist_ref_ncep_* && $forcing != "NCEP" ]]; then
				skip=true
			elif [[ $name == namelist_cfg_ncep_* && $forcing != "NCEP" ]]; then
				skip=true
			fi

			if [ "$skip" = true ]; then
				skip "$name (forcing is $forcing)"
			elif [ -f $name ]; then
				skip "$name exists"
			else
				cp $val $name
			fi
		fi

		# Copy xml files in a way so changes can be made
		if [[ $name == *".xml" ]]; then
			if [ -f $name ]; then
				skip "$name exists"
			else
				cp $val $name
			fi
		fi

		# Copy the executable over, good to keep these.
		if [[ $name == "opa"*$Model || $name == "nemo"*$Model ]]; then
			if [ -f $name ]; then
				skip "$name exists"
			else
				cp $val $name
				ok "Executable: $name"
			fi
			ln -fs $name $executable
			ln -fs $name opa
		fi
	fi
done < "$dataFileFullPath"

# Link EMP file (freshwater-budget seed for nn_fwb=2).
# Only NEMO3.6 needs this. NEMO5 stores the fwb (a_fwb) in the ocean restart,
# so NEMO5 configs omit EMPaveFile and this block is skipped.
if [ -n "$EMPaveFile" ]; then
	if [ ! -f EMPave_${prevYear}.dat ]; then
		rm -f EMPave_${prevYear}.dat
		ln -fs $EMPaveFile EMPave_${prevYear}.dat
		ln -fs EMPave_${prevYear}.dat EMPave_old.dat
	else
		skip "EMPave exists, using existing file"
		ln -fs EMPave_${prevYear}.dat EMPave_old.dat
	fi
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

section "Configuration"
ok "Compiler keys: ${PIIC^^} ${C14^^}"

# ----- Process flags -----
# CO2
rm -f atmco2.dat

ok "CO2: $CO2"
if [ $CO2 == "VARIABLE" ]; then
	ln -s atmco2.dat.variable atmco2.dat
else
	ln -s atmco2.dat.static atmco2.dat
fi

ok "Forcing: $forcing"
if [ $forcing == "NCEP" ]; then
	forcing_prefix="ncep"
elif [ $forcing == "ERA" ]; then
	forcing_prefix="era"
else
	forcing_prefix="jra"
fi

if [[ "$nemoVersion" == "NEMO5" ]]; then
	control_namelist="namelist_cfg"
else
	control_namelist="namelist_ref"
fi

rm -f $control_namelist

# Layer 1: Functional symlinks (abstract forcing type)
# - coldstart: nn_rstctl=0, uses nn_date0 for start date
# - restart:   nn_rstctl=2, reads date from restart file, historical forcing
# - cycling:   nn_rstctl=2, reads date from restart file, loops single year forcing
ln -sf ${control_namelist}_${forcing_prefix}_coldstart ${control_namelist}_coldstart
ln -sf ${control_namelist}_${forcing_prefix}_restart ${control_namelist}_restart
ln -sf ${control_namelist}_${forcing_prefix}_cycling ${control_namelist}_cycling

# Automatically correct nn_date0 in the coldstart namelist to match yearStart from setup data
expectedDate="${yearStart}0101"
currentDate=$( grep "nn_date0" ${control_namelist}_coldstart | head -1 | awk -F'=' '{print $2}' | awk '{print $1}' )

if [ "$currentDate" != "$expectedDate" ]; then
	info "nn_date0: $currentDate → $expectedDate"
	# --follow-symlinks: write through the symlink to the target file.
	# Without it, sed -i replaces the symlink with a regular file, severing
	# the layer-1 abstraction (${control_namelist}_coldstart -> ${control_namelist}_${forcing}_coldstart).
	sed --follow-symlinks -i "s/nn_date0.*=.*/nn_date0    = $expectedDate/" ${control_namelist}_coldstart
fi

# Layer 2: Temporal symlinks (when each is used)
# - first_year:  always uses coldstart
# - other_years: uses cycling (spinup) or restart (transient)
ok "Forcing mode: $forcing_mode"
ln -sf ${control_namelist}_coldstart ${control_namelist}_first_year
if [ "$forcing_mode" == "spinup" ]; then
	ln -sf ${control_namelist}_cycling ${control_namelist}_other_years
elif [ "$forcing_mode" == "transient" ]; then
	ln -sf ${control_namelist}_restart ${control_namelist}_other_years
else
	warn "Unrecognized forcing_mode '$forcing_mode' (expected 'spinup' or 'transient')"
	exit 1
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
#   (*) setup_spin.sh copies restart files then switches the active namelist -> other_years
#
if [ ! -f restart_0000.nc ]; then
	ln -sf ${control_namelist}_first_year $control_namelist
else
	ln -sf ${control_namelist}_other_years $control_namelist
fi

# Apply the per-run ocean timestep (setUpData timestep:) to the local namelists,
# scaling the dependent counters and stepsPerYear. No-op when already consistent.
if [ -x "${SCRIPT_DIR}/update_timestep.sh" ]; then
	"${SCRIPT_DIR}/update_timestep.sh" . --apply || warn "update_timestep.sh failed; namelists left at their template timestep"
fi

section "Physics"

# Temperature and salinity restoring
TR=$( grep "nn_sstr " $control_namelist 2>/dev/null | awk -F' ' '{print $3}' )
SR=$( grep "nn_sssr " $control_namelist 2>/dev/null | awk -F' ' '{print $3}' )
LP=$( grep "ln_lop" namelist_top_ref 2>/dev/null | awk -F' ' '{print $3}' )
LP=${LP:-.false.}

if [ "$TR" = 1 ]; then
	ok "Temperature restoring: ON"
else
	info "Temperature restoring: OFF"
fi

if [ "$SR" = 1 ]; then
	ok "Salinity restoring: ON"
else
	info "Salinity restoring: OFF"
fi

# Check that files for LIMPHY are set correctly
IODEF_PATH=$( grep "^iodef.xml:" $setUpDatafile | awk -F':' '{print $2}' )
KP=$( grep "^keepLimPhy:" $setUpDatafile | awk -F':' '{print $NF}' )

# Analyser/visualiser config selection (per-run; e.g. NEMO5 vs NEMO3.6 grid).
# Filenames are resolved against analyser/ and visualise/; absolute paths are
# used as-is. These MUST be set in setUpData -- there is no NEMO-version default.
ANALYSER_CONFIG=$( grep "^analyser_config:" $setUpDatafile | awk -F':' '{print $2}' )
VISUALISE_CONFIG=$( grep "^visualise_config:" $setUpDatafile | awk -F':' '{print $2}' )
if [[ "$ANALYSER_CONFIG" = /* ]]; then
	analyserCfgPath="$ANALYSER_CONFIG"
else
	analyserCfgPath="${SCRIPT_DIR}/analyser/${ANALYSER_CONFIG}"
fi
if [[ "$VISUALISE_CONFIG" = /* ]]; then
	visualiseCfgPath="$VISUALISE_CONFIG"
else
	visualiseCfgPath="${SCRIPT_DIR}/visualise/${VISUALISE_CONFIG}"
fi
err=0

if [ "$LP" = ".true." ]; then
	ok "LimPhy: ON"

	if [ $KP != 1 ]; then
		warn "KEEP value for LimPhy not set to 1"
		err=1
	fi
else
	info "LimPhy: OFF"

	if [ $KP != 0 ]; then
		warn "KEEP value for LimPhy not set to 0"
		err=1
	fi
fi

# Check iodef file exists
if [ ! -f "$IODEF_PATH" ]; then
	warn "IODEF file does not exist: $IODEF_PATH"
	err=1
fi

# analyser_config / visualise_config must be set and exist (no version default)
if [ -z "$ANALYSER_CONFIG" ]; then
	warn "analyser_config: not set in $(basename $setUpDatafile)"
	err=1
elif [ ! -f "$analyserCfgPath" ]; then
	warn "analyser config not found: $analyserCfgPath"
	err=1
fi
if [ -z "$VISUALISE_CONFIG" ]; then
	warn "visualise_config: not set in $(basename $setUpDatafile)"
	err=1
elif [ ! -f "$visualiseCfgPath" ]; then
	warn "visualise config not found: $visualiseCfgPath"
	err=1
fi

if [ $err == 1 ]; then
	exit 2
fi

# Get code version
codePath=$( awk -F':' -v model="$Model" '$1 ~ "^(opa|nemo).*" model "$" {print $2; exit}' "$setUpDatafile" )
codeVersion=$( echo "$codePath" | awk -F'/' '{print$(NF-5)}' )

# ----- Create a detached toolkit and workflow snapshot for this run -----
toolkitDir="$modelDir/.planktom_toolkit"
if [ -d "$toolkitDir" ]; then
	skip "Toolkit snapshot exists"
else
	mkdir -p "$toolkitDir"
	cp -R "${SCRIPT_DIR}/analyser" "$toolkitDir/"
	cp -R "${SCRIPT_DIR}/shared" "$toolkitDir/"
	cp -R "${SCRIPT_DIR}/visualise" "$toolkitDir/"
	cp -p "${SCRIPT_DIR}/compute_amoc.sh" "$toolkitDir/"
	ok "Toolkit snapshot: $toolkitDir"
fi

# The report renderer expects this canonical name beside make_html.sh. Keep the
# original named configs too because config_utils resolves them from setUpData.
cp "$visualiseCfgPath" "$toolkitDir/visualise/visualise_config.toml"
cp "$analyserCfgPath" analyser_config.toml
cp "$visualiseCfgPath" visualise_config.toml

workflowFiles="workflow_common.sh run_year.sh checkpoint_year.sh analyse_year.sh archive_year.sh report_run.sh submit_workflow.sh status.sh"
for file in $workflowFiles; do
	cp -p "${SCRIPT_DIR}/workflow/${file}" "$modelDir/${file}"
done

# Remove generated launchers from the former self-resubmitting architecture.
# Existing custom submit.<year> files are deliberately left untouched.
rm -f nemo.job nemo_compute.job nemo5.job nemo5_compute.job tidyup.job tidyup.sh
for legacy_link in analyser*.py shared compute_amoc.sh; do
	[ -L "$legacy_link" ] && rm -f "$legacy_link"
done
for source_file in "${SCRIPT_DIR}"/visualise/*; do
	legacy_link=$(basename "$source_file")
	[ -L "$legacy_link" ] && rm -f "$legacy_link"
done

if [ -f ${SCRIPT_DIR}/iodef_tom12piicc14.xml ]; then
	cp ${SCRIPT_DIR}/iodef_tom12piicc14.xml .
fi

# Save parameters needed for creating html file
echo $id $codeVersion $(date '+%d-%b-%Y') $yearStart $yearEnd ${CO2,,} $forcing ${forcing_mode,,} $TR $SR > html_parms

# Get setUpRun script
cp ${SCRIPT_DIR}/setUpRun.sh .

# ----- Setup from spinup model (if provided) -----
if [ -n "$spinupModelId" ]; then
	section "Spinup Setup (from $spinupModelId)"
	bash ${SCRIPT_DIR}/setup_spin.sh $id $spinupModelId
	if [ $? -ne 0 ]; then
		warn "Spinup setup failed"
		exit 1
	fi
fi

# ----- Write the resolved environment shared by every workflow task -----
runId=$id
MAMBA_EXE=${MAMBA_EXE:-/gpfs/home/vhf24tbu/miniforge3/bin/mamba}
MAMBA_ROOT_PREFIX=${MAMBA_ROOT_PREFIX:-/gpfs/home/vhf24tbu/miniforge3}
analysisEnv=${analysisEnv:-base}

runEnvVars="runId yearStart yearEnd basedir modelDir simulation Model forcing_prefix forcing_mode nemoVersion executable iceRestartName nemoCpus xiosCpus useXiosServer timestep stepsPerYear spinupStart spinupEnd spinupRestartKeepFrequency spinupOutputKeepFrequency runRestartKeepFrequency runOutputKeepFrequency keepGrid_T keepDiad keepPtrc keepIce keepGrid_U keepGrid_V keepGrid_W keepLimPhy keepGflux archiveDir outputFrequency toolkitDir MAMBA_EXE MAMBA_ROOT_PREFIX analysisEnv"
{
	echo "# Generated by setUpRun.sh; source from trusted workflow scripts only."
	for name in $runEnvVars; do
		printf '%s=%q\n' "$name" "${!name}"
	done
} > run.env.tmp
mv -f run.env.tmp run.env
ok "Resolved workflow environment: run.env"

section "Summary"
echo -e "  ${DIM}Years:${RESET}           $yearStart → $yearEnd"
echo -e "  ${DIM}Model dir:${RESET}       $modelDir"
echo -e "  ${DIM}Simulation:${RESET}      $simulation"
echo -e "  ${DIM}Model:${RESET}           $Model"
echo -e "  ${DIM}NEMO version:${RESET}    $nemoVersion"
echo -e "  ${DIM}Forcing:${RESET}         ${forcing_prefix} / ${forcing_mode}"

echo ""
read -p "Press any key to run it? (cntr+c otherwise)"

section "Submitting Job"
if ! "$modelDir/submit_workflow.sh" "$modelDir" "$yearStart" "$yearEnd" auto; then
	warn "Workflow submission failed; inspect state/jobs.tsv before retrying"
	exit 1
fi
ok "Workflow submitted. Check: ${DIM}$modelDir/status.sh $modelDir${RESET}"

# ----- Save model details -----
echo "$id [$(date '+%Y-%m-%d')]" >> "${HOME}/scratch/ModelRuns/modelRuns.org"
