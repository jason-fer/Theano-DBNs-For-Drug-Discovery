#!/bin/bash

#PBS -j oe
#PBS -N mmnt-jbf-log-reg-tox21
#PBS -l nodes=1:ncpus=1,mem=2gb,walltime=10:00:00
#PBS -t 0-11

cd $PBS_O_WORKDIR
module load cuda anaconda accelerate
python sk_logistic_regression.py tox21 $PBS_ARRAYID