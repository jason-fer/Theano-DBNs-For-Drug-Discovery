#!/bin/bash

#PBS -N mmnt-jason-random-forests-dud_e
#PBS -l nodes=1:gpus=1,walltime=2000:00:00
#PBS -t 0-101

cd $PBS_O_WORKDIR
module load cuda anaconda
python sk_random_forests.py dud_e $PBS_ARRAYID