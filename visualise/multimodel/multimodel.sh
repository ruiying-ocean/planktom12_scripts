#!/bin/sh

# Load Ferret for map generation
module add ferret/7.6.0
if [ -n "$FER_PATH" ]; then
    source $FER_PATH
fi

# Use miniforge Python instead of module python/3.8
export PATH=~/miniforge3/bin:$PATH


# Create save directory from current date
now=$(date +'%d%m%y-%H%M%S')

# Source central directory where the files and scripts are located
srcDir="/gpfs/home/vhf24tbu/setUpRuns/HALI-DEV/multimodel/"

# Create directory to save files to
curDir=${PWD}
saveDir=${curDir}/${now}/

mkdir ${saveDir}
cp modelsToPlot.csv ${saveDir}
cd ${saveDir}

# Read in runs to compare from csv files
runs=( $( cut -f 1 -d , modelsToPlot.csv | tail -n +2 ) )
desc=( $( cut -f 2 -d , modelsToPlot.csv | tail -n +2 ) )
strt=( $( cut -f 3 -d , modelsToPlot.csv | tail -n +2 ) )
from=( $( cut -f 4 -d , modelsToPlot.csv | tail -n +2 ) )
  to=( $( cut -f 5 -d , modelsToPlot.csv | tail -n +2 ) )
 end=( $( cut -f 6 -d , modelsToPlot.csv | tail -n +2 ) )
locs=( $( cut -f 7 -d , modelsToPlot.csv | tail -n +2 ) )

length=${#runs[@]}
if [ $length -gt 8 ]; then
	echo "Trying to plot too many runs, maximum is 8"
	exit 2
fi

# Copy in the required files
cp ${srcDir}/multimodel.py .

# Copy Python map generation scripts from visualise directory
visualiseDir="/gpfs/home/vhf24tbu/setUpRuns/HALI-DEV/visualise/"
cp ${visualiseDir}/make_maps.py .
cp ${visualiseDir}/map_utils.py .

# Run the python script to generate the plots (with debug to see warnings)
./multimodel.py ${saveDir} --debug

# Create maps for each run for final year using Python
for i in ${!runs[@]}; do
	echo "Creating maps for" ${runs[$i]}

	# Use Python map generation instead of Ferret
	python3 make_maps.py ${runs[$i]} ${to[$i]} ${to[$i]} \
		--basedir ${locs[$i]} \
		--output-dir ${saveDir} \
		--obs-dir /gpfs/home/vhf24tbu/Observations
done

# Copy observation images to save directory
cp ${srcDir}/*.png .

# Generate HTML using Quarto (replaces Python + Jinja2 system)
${srcDir}/createMultimodelHTML.sh ${now} 0

# Save list of models
cd ..
echo "" >> multiplots.txt
echo ${now} >> multiplots.txt

for i in ${!runs[@]}; do    

        old_desc=${desc[$i]}
        new_desc=${old_desc//_/ }

        echo ${runs[$i]} ${new_desc} ${to[$i]} ${locs[$i]} >> multiplots.txt
done

echo "Finished: saved to" ${saveDir}
