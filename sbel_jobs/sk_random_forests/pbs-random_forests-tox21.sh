#!/bin/bash

#PBS -j oe
#PBS -N mmnt-jbf-rand-forst-tox21
#PBS -l nodes=1:ncpus=1,mem=3gb,walltime=20:00:00
#PBS -t 0-11

cd $PBS_O_WORKDIR
module load cuda anaconda accelerate
python sk_random_forests.py tox21 $PBS_ARRAYID