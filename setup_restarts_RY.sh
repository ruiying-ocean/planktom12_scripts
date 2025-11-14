#!/bin/sh

cpus=48
src_model_dir=$1
timestep=$2
target_modeldir=$3

echo "Setting up restart softlinks for timestep: "$timestep
echo "Target directory for symbolic links: "$target_modeldir

# Create the target directory if it doesn't exist
mkdir -p $target_modeldir

# Link the restart file to use for the next year to the one just completed.
for (( i=0; i<$cpus; i++ )); do
	if (( i < 10 )); then
		proc=0$i
	else
		proc=$i
	fi
    
	ln -fs ${src_model_dir}/ORCA2_${timestep}_restart_00${proc}.nc ${target_modeldir}/restart_00${proc}.nc
	ln -fs ${src_model_dir}/ORCA2_${timestep}_restart_ice_00${proc}.nc ${target_modeldir}/restart_ice_in_00${proc}.nc
	ln -fs ${src_model_dir}/ORCA2_${timestep}_restart_trc_00${proc}.nc ${target_modeldir}/restart_trc_00${proc}.nc
done
