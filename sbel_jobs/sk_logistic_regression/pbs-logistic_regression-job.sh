#!/bin/bash

#PBS -N MMNT-LogisticRegression
#PBS -l nodes=1:gpus=1,walltime=01:00:00

cd $PBS_O_WORKDIR
module load cuda anaconda
python sk_logistic_regression.py
