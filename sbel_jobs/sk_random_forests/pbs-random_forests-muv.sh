#!/bin/bash

#PBS -N mmnt-jason-random-forests-muv
#PBS -l nodes=1:gpus=1,walltime=2000:00:00
#PBS -t 0-33

cd $PBS_O_WORKDIR
module load cuda anaconda
python sk_random_forests.py muv $PBS_ARRAYID