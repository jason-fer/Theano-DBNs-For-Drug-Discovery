#!/bin/bash

#PBS -l nodes=1:ppn=1,walltime=00:01:00
#PBS -t 1-50

echo $PBS_ARRAYID > ~/`hostname`-$PBS_ARRAYID
