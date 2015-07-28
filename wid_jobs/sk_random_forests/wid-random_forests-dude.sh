#!/bin/bash
# This has to be executable

echo _CONDOR_JOB_IWD $_CONDOR_JOB_IWD
echo Cluster $cluster
echo Process $process
echo RunningOn $runningon

# $process is your 0-indexed job index that you can use for looping
CMD="python sk_random_forests.py dud_e $process"
echo $CMD
