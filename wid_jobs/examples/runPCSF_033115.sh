#!/bin/bash
# Run PCSF using the paramters set in the wrapper and HTCondor submit file

echo _CONDOR_JOB_IWD $_CONDOR_JOB_IWD
echo Cluster $cluster
echo Process $process
echo RunningOn $runningon

codeD=/mnt/ws/home/agitter/pcsf

prizefile=/mnt/ws/home/agitter/projects/kshv/data/prizes/030615/${prizetype}.txt
edgefile=/mnt/ws/home/agitter/projects/kshv/data/networks/9606.mitab.08122013.MIScore_scored_full_geneSym_convertedOnly_totalScore_unique_4col.dat
dummy=terminals
msgpath=/mnt/ws/home/agitter/bin/msgsteiner9/msgsteiner9
outlabel=${prizetype}_beta${beta}_mu${mu}_omega${omega}_seed${seed}

CMD="python2.7 $codeD/PCSF.py \
	-p $prizefile \
	-e $edgefile \
	-c $conf \
	-d $dummy \
	--msgpath=$msgpath \
	--outpath=$outpath \
	--outlabel=$outlabel \
	--cyto30 \
	-s $seed"

echo $CMD
$CMD
