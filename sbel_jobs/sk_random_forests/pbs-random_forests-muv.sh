#!/bin/bash

#PBS -j oe
#PBS -N mmnt-jbf-rand-forst-muv
#PBS -l nodes=1:ncpus=1,mem=4gb,walltime=20:00:00
#PBS -t 0-16

cd $PBS_O_WORKDIR
module load cuda anaconda accelerate
python sk_random_forests.py muv $PBS_ARRAYID