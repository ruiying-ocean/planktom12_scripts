#!/bin/bash
# Rebuild per-processor NEMO restart files into single global files
# and optionally reset the date for starting a new run.
#
# Usage:
#   rebuild_restart.sh <run_dir> <new_date> [output_dir] [timestep] [nprocs]
#
# Arguments:
#   run_dir    Directory containing ORCA2_*_restart_*.nc files
#   new_date   Target start date as YYYYMMDD (e.g. 17500101)
#              The restart will be stamped as the day before (17491231)
#   output_dir Output directory for rebuilt files (default: current directory)
#   timestep   Timestep label (e.g. 01921725). If omitted, auto-detects
#              the latest timestep from restart files in run_dir.
#   nprocs     Number of processors (default: 48)
#
# Examples:
#   rebuild_restart.sh ~/scratch/ModelRuns/TOM12_RY_JRA3 17500101
#   rebuild_restart.sh ~/scratch/ModelRuns/TOM12_RY_JRA3 17500101 ~/Observations/input_data/new_restart
#   rebuild_restart.sh ~/scratch/ModelRuns/TOM12_RY_JRA3 17500101 ~/Observations/input_data/new_restart 01921725 48

set -e

module load netcdf/4.7.4/gcc

REBUILD_NEMO="$HOME/scratch/NEMO3.6-TOM12-MAIN/REBUILD_NEMO/rebuild_nemo"
NCAP2="$HOME/miniforge3/bin/ncap2"

# -----------------------------------------------------------------------------
# Parse arguments
# -----------------------------------------------------------------------------
if [ $# -lt 2 ]; then
    echo "Usage: $0 <run_dir> <new_date_YYYYMMDD> [output_dir] [timestep] [nprocs]"
    echo ""
    echo "  run_dir    Directory containing ORCA2_*_restart_*.nc files"
    echo "  new_date   Desired start date YYYYMMDD (e.g. 17500101)"
    echo "  output_dir Output directory (default: current directory)"
    echo "  timestep   Timestep label (e.g. 01921725, auto-detected if omitted)"
    echo "  nprocs     Number of processors (default: 48)"
    exit 1
fi

RUN_DIR=$(cd "$1" && pwd)
NEW_DATE="$2"
OUTPUT_DIR="${3:-.}"
TIMESTEP="$4"
NPROCS="${5:-48}"

# Create output directory if needed
mkdir -p "${OUTPUT_DIR}"
OUTPUT_DIR=$(cd "${OUTPUT_DIR}" && pwd)

# Auto-detect latest timestep if not provided
if [ -z "${TIMESTEP}" ]; then
    TIMESTEP=$(ls "${RUN_DIR}"/ORCA2_*_restart_0000.nc 2>/dev/null \
        | sed 's/.*ORCA2_//; s/_restart_0000\.nc//' \
        | sort -n \
        | tail -1)
    if [ -z "${TIMESTEP}" ]; then
        echo "Error: No ORCA2_*_restart_0000.nc files found in ${RUN_DIR}"
        exit 1
    fi
    echo "Auto-detected latest timestep: ${TIMESTEP}"
fi

# Compute the day before new_date for the restart stamp
# (restart represents end of previous day)
RESTART_DATE=$(date -d "${NEW_DATE} - 1 day" +%Y%m%d 2>/dev/null || \
               python3 -c "
from datetime import datetime, timedelta
d = datetime.strptime('${NEW_DATE}', '%Y%m%d') - timedelta(days=1)
print(d.strftime('%Y%m%d'))
")

echo "============================================================"
echo "Rebuild NEMO restarts"
echo "============================================================"
echo "  Run directory : ${RUN_DIR}"
echo "  Output dir    : ${OUTPUT_DIR}"
echo "  Timestep      : ${TIMESTEP}"
echo "  Start date    : ${NEW_DATE}"
echo "  Restart stamp : ${RESTART_DATE}"
echo "  Processors    : ${NPROCS}"
echo "============================================================"

# Check that input files exist
for type in restart restart_ice restart_trc; do
    if [ ! -f "${RUN_DIR}/ORCA2_${TIMESTEP}_${type}_0000.nc" ]; then
        echo "Error: ORCA2_${TIMESTEP}_${type}_0000.nc not found in ${RUN_DIR}"
        exit 1
    fi
done

# rebuild_nemo works in the current directory, so we work in output_dir
# and symlink the input files there temporarily
cd "${OUTPUT_DIR}"

for type in restart restart_ice restart_trc; do
    for f in "${RUN_DIR}"/ORCA2_${TIMESTEP}_${type}_*.nc; do
        ln -fs "$f" .
    done
done

# -----------------------------------------------------------------------------
# 1. Rebuild per-processor files into single files
# -----------------------------------------------------------------------------
echo ""
echo "Rebuilding ocean restart..."
${REBUILD_NEMO} ORCA2_${TIMESTEP}_restart ${NPROCS}

echo ""
echo "Rebuilding ice restart..."
${REBUILD_NEMO} ORCA2_${TIMESTEP}_restart_ice ${NPROCS}

echo ""
echo "Rebuilding tracer restart..."
${REBUILD_NEMO} ORCA2_${TIMESTEP}_restart_trc ${NPROCS}

# Clean up temporary symlinks to per-processor files
rm -f ORCA2_${TIMESTEP}_restart_[0-9]*.nc
rm -f ORCA2_${TIMESTEP}_restart_ice_[0-9]*.nc
rm -f ORCA2_${TIMESTEP}_restart_trc_[0-9]*.nc

# -----------------------------------------------------------------------------
# 2. Reset date/time variables in merged files
# -----------------------------------------------------------------------------
echo ""
echo "Setting restart date to ${RESTART_DATE}..."

${NCAP2} -O -s "ndastp=${RESTART_DATE}; kt=0; adatrj=0.0" \
    ORCA2_${TIMESTEP}_restart.nc ORCA2_${TIMESTEP}_restart.nc

${NCAP2} -O -s "ndastp=${RESTART_DATE}; kt=0; adatrj=0.0" \
    ORCA2_${TIMESTEP}_restart_ice.nc ORCA2_${TIMESTEP}_restart_ice.nc

${NCAP2} -O -s "ndastp=${RESTART_DATE}; kt=0; adatrj=0.0; rdttrc1=5760.0" \
    ORCA2_${TIMESTEP}_restart_trc.nc ORCA2_${TIMESTEP}_restart_trc.nc

# -----------------------------------------------------------------------------
# 3. Create symlinks for NEMO
# -----------------------------------------------------------------------------
echo ""
echo "Creating symlinks..."

ln -fs ORCA2_${TIMESTEP}_restart.nc restart.nc
ln -fs ORCA2_${TIMESTEP}_restart_ice.nc restart_ice_in.nc
ln -fs ORCA2_${TIMESTEP}_restart_trc.nc restart_trc.nc

echo ""
echo "============================================================"
echo "Done. Created in ${OUTPUT_DIR}:"
echo "  restart.nc        -> ORCA2_${TIMESTEP}_restart.nc"
echo "  restart_ice_in.nc -> ORCA2_${TIMESTEP}_restart_ice.nc"
echo "  restart_trc.nc    -> ORCA2_${TIMESTEP}_restart_trc.nc"
echo "============================================================"
