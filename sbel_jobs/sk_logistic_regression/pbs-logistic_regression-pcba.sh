#!/bin/bash

#PBS -j oe
#PBS -N mmnt-jbf-log-reg-pcba
#PBS -l nodes=1:ncpus=1,mem=6gb,walltime=10:00:00
#PBS -t 0-127

cd $PBS_O_WORKDIR
module load cuda anaconda accelerate
python sk_logistic_regression.py pcba $PBS_ARRAYID