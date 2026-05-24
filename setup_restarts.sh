#!/bin/bash

src_model_dir=$1
timestep=$2
target_modeldir=$3
ice_restart_name=${4:-restart_ice_in}
cpus=${5:-${nemoCpus:-48}}

echo "Setting up restart softlinks for timestep: "$timestep
echo "Target directory for symbolic links: "$target_modeldir

# Create the target directory if it doesn't exist
mkdir -p $target_modeldir

# Helper function: create symlink only if source exists
link_if_exists() {
	src=$1
	dst=$2
	if [ ! -f "$src" ]; then
		echo "Error: Source file does not exist: $src"
		exit 1
	fi
	ln -fs "$src" "$dst"
}

# Link the restart files
for (( i=0; i<$cpus; i++ )); do
	proc=$(printf "%02d" $i)

	link_if_exists "${src_model_dir}/ORCA2_${timestep}_restart_00${proc}.nc" "${target_modeldir}/restart_00${proc}.nc"
	link_if_exists "${src_model_dir}/ORCA2_${timestep}_restart_ice_00${proc}.nc" "${target_modeldir}/${ice_restart_name}_00${proc}.nc"
	link_if_exists "${src_model_dir}/ORCA2_${timestep}_restart_trc_00${proc}.nc" "${target_modeldir}/restart_trc_00${proc}.nc"
done

echo "All restart symlinks created successfully"
