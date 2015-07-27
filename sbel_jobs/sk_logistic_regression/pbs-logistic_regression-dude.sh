#!/bin/bash

#PBS -j oe
#PBS -N mmnt-jbf-log-reg-dud_e
#PBS -l nodes=1:ncpus=1,mem=3gb,walltime=10:00:00
#PBS -t 0-101

cd $PBS_O_WORKDIR
module load cuda anaconda accelerate
python sk_logistic_regression.py dud_e $PBS_ARRAYID
