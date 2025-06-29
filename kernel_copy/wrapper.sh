#!/usr/bin/env bash

LANG=en_US.UTF-8
# load required modules here
module purge
#module load OCI/12.2
module load NeSI
module load cuDNN/8.6.0.163-CUDA-11.8.0 
#module load Anaconda3/2019.03-gimkl-2018b
#module load CDO/1.9.5-GCC-7.1.0
#cd /nesi/project/niwa00004/rampaln
exec /nesi/project/niwa00018/queenle/ml_env_v2/bin/python $@
