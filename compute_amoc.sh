#!/bin/sh
#
# compute_amoc.sh - Compute MOC (Meridional Overturning Circulation) using CDFtools
#
# Usage: bash compute_amoc.sh <grid_V_file> [grid_T_file]
#
# Runs cdfmoc on a grid_V file within a MOC/ subdirectory.
# If grid_T is provided, passes it to cdfmoc with -t for density info.
# Saves output as MOC/moc_{year}.nc
# Gracefully skips if CDFtools binary or grid_V file is not found.

grid_v_file=$1
grid_t_file=$2

# Validate arguments
if [ -z "$grid_v_file" ]; then
	echo "Usage: bash compute_amoc.sh <grid_V_file> [grid_T_file]"
	exit 1
fi

# Check grid_V file exists
if [ ! -f "$grid_v_file" ]; then
	echo "Warning: grid_V file not found: $grid_v_file - skipping AMOC computation"
	exit 0
fi

# CDFtools binary and mask locations
CDFMOC="${HOME}/src/CDFTOOLS/bin/cdfmoc"
MASK_DIR="${HOME}/masks"

# Check CDFtools binary exists
if [ ! -x "$CDFMOC" ]; then
	echo "Warning: cdfmoc not found at $CDFMOC - skipping AMOC computation"
	exit 0
fi

# Extract year from grid_V filename (pattern: ORCA2_1m_YYYY0101_YYYY1231_grid_V.nc)
year=$(echo "$grid_v_file" | sed -n 's/.*_\([0-9]\{4\}\)0101_.*/\1/p')
if [ -z "$year" ]; then
	echo "Warning: could not extract year from filename: $grid_v_file"
	exit 0
fi

# Check mask files exist
MESH_MASK="${MASK_DIR}/mesh_mask3_6.nc"
MASKGLO="${MASK_DIR}/new_maskglo_TOM.nc"

if [ ! -f "$MESH_MASK" ] || [ ! -f "$MASKGLO" ]; then
	echo "Warning: mask files not found in $MASK_DIR - skipping AMOC computation"
	exit 0
fi

# Resolve input paths to absolute before cd
grid_v_file=$(cd "$(dirname "$grid_v_file")" && pwd)/$(basename "$grid_v_file")
if [ -n "$grid_t_file" ] && [ -f "$grid_t_file" ]; then
	grid_t_file=$(cd "$(dirname "$grid_t_file")" && pwd)/$(basename "$grid_t_file")
fi

# NEMO5/XIOS (e.g. KRF12) names grid_V horizontal dims x_grid_V/y_grid_V, but
# cdfmoc (NEMO3.6-era) requires literal x/y. vomecrty/depthv/time_counter already
# match, and geometry comes from the mesh, so only those two dims need renaming.
# Rename into a temp copy (its filename keeps the year so extraction still works).
nemo5_tmp=""
if ncdump -h "$grid_v_file" 2>/dev/null | grep -qE 'x_grid_V[[:space:]]*='; then
	if command -v ncrename >/dev/null 2>&1; then
		nemo5_tmp="${grid_v_file%.nc}_cdfren.nc"
		cp "$grid_v_file" "$nemo5_tmp"
		ncrename -d x_grid_V,x -d y_grid_V,y "$nemo5_tmp" >/dev/null 2>&1
		grid_v_file="$nemo5_tmp"
		echo "  NEMO5 grid_V detected: renamed x_grid_V/y_grid_V -> x/y"
	else
		echo "Warning: NEMO5 grid_V dims (x_grid_V) but ncrename/NCO not on PATH (try 'mamba activate') - skipping AMOC"
		exit 0
	fi
fi

# Create MOC directory and work from there
mkdir -p MOC
cd MOC

echo "Computing MOC for year $year"

# cdfmoc expects mesh_hgr.nc, mesh_zgr.nc, mask.nc, and new_maskglo.nc
# Symlink masks (only if not already present)
for name in mesh_hgr.nc mesh_zgr.nc mask.nc; do
	[ ! -e "$name" ] && ln -sf "$MESH_MASK" "$name"
done
[ ! -e "new_maskglo.nc" ] && ln -sf "$MASKGLO" "new_maskglo.nc"

# Load the system NetCDF lib for CDFtools. Done HERE -- after the conda-based
# ncrename ran above -- so the system libs don't shadow conda's under ncrename
# (matches the manual sequence that worked: rename in conda env, then cdfmoc).
module load netcdf/4.7.4/gcc

# Run cdfmoc (pass grid_T with -t if available)
if [ -n "$grid_t_file" ] && [ -f "$grid_t_file" ]; then
	$CDFMOC -v "$grid_v_file" -t "$grid_t_file"
else
	$CDFMOC -v "$grid_v_file"
fi

# cdfmoc produces moc.nc
if [ -f "moc.nc" ]; then
	mv moc.nc "moc_${year}.nc"
	echo "  MOC saved to MOC/moc_${year}.nc"
else
	echo "Warning: cdfmoc did not produce moc.nc"
fi

# Clean up the temporary renamed grid_V copy (NEMO5 path only)
[ -n "$nemo5_tmp" ] && rm -f "$nemo5_tmp"
