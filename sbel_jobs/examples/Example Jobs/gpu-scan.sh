#!/bin/sh

#PBS -N gpu-scan
#PBS -l nodes=1:gpus=1,walltime=00:01:00

cd $PBS_O_WORKDIR

$NVSDKCOMPUTE_ROOT/bin/x86_64/linux/release/scan > gpu-scan.out.$PBS_JOBID.`hostname`
