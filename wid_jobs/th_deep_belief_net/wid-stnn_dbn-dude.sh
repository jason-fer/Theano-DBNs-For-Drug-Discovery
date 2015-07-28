#!/bin/bash
# This has to be executable

echo _CONDOR_JOB_IWD $_CONDOR_JOB_IWD
echo Cluster $cluster
echo Process $process
echo RunningOn $runningon

# $process is your 0-indexed job index that you can use for looping
CMD="python th_deep_belief_net.py dud_e $process"
echo $CMD
