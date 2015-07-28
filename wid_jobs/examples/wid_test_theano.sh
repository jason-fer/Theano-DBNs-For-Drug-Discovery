#!/bin/bash
# Test which machine the job is run on and whether Theano is installed

echo _CONDOR_JOB_IWD $_CONDOR_JOB_IWD
echo Cluster $cluster
echo Process $process
echo RunningOn $runningon

which python

python -c "import theano; print 'Theano version %s' % theano.__version__"