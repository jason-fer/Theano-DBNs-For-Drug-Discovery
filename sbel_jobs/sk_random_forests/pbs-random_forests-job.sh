#!/bin/bash

#PBS -N MMNT-RandomForests
#PBS -l nodes=1:gpus=1,walltime=76:00:00

cd $PBS_O_WORKDIR
module load cuda anaconda
python sk_random_forests.py
