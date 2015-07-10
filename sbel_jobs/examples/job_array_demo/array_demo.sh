#!/bin/bash

#PBS -N basic_array_demo
#PBS -l nodes=1:gpus=1,walltime=00:01:00
#PBS -t 1-10

cd $PBS_O_WORKDIR

python job.py /home/cvandenheuve/data-files/$PBS_ARRAYID.txt

