#!/bin/bash
#    Begin PBS directives
#PBS -A csc025ewk
#PBS -N warmup
#PBS -j oe
#PBS -l walltime=0:01:00,size=16
#PBS -l gres=widow2%widow3
#PBS -V
#    End PBS directives and begin shell commands

# replaced with -V.   source ~/jaguar_env.sh


cd /tmp/work/pantc/nscale/bin

date

exit 0
