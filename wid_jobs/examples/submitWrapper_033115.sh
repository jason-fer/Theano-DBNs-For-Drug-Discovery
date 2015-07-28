#!/bin/bash
# Submit PCSF runs with different params, prizes, and seeds
# Sets up a new configuration file as needed and uses
# environment variables to pass other arguments to PCSF.py

# Set the output directory for PCSF and HTCondor output
export outpath=/mnt/ws/home/agitter/projects/kshv/results/033115
mkdir -p $outpath

# Parameters that do not vary
D=10
# Noise to compute a family of solutions
r=0.01
g=1e-3
prizes=maxProteinScores2KM_tfPrizes
# Parameters fixed during parameter sweep
b=4.5
m=0.02
w=2.5

# When the parameters don't change, neeed this to appear outside
# of the loop below.  Otherwise there are race conditions where
# one job wants to read the file but another removes it.
# Create the conf file, removing any older copies of the file if they exist
filename=conf/conf_w${w}_b${b}_D${D}_m${m}_r${r}_g${g}.txt
rm -f $filename
touch $filename
printf "w = ${w}\n" >> $filename
printf "b = ${b}\n" >> $filename
printf "D = ${D}\n" >> $filename
printf "mu = ${m}\n" >> $filename
printf "r = ${r}\n" >> $filename
printf "g = ${g}\n" >> $filename

# Use different seeds for each run
for s in $(seq 1 1000)
do
	# Set the remaining environment variables
	export conf=$filename
	export beta=$b
	export mu=$m
	export omega=$w
	export prizetype=$prizes
	export seed=$s

	# Submit the job to HTCondor with the configuration file, params, and seed
	condor_submit submitPCSF_033115.sub
done