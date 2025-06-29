#!/bin/bash -l

# This is the watercare script suite to run for the reports

PATH=/usr/local/sbin:/usr/local/bin:/sbin:/bin:/usr/sbin:/usr/bin
#export DISPLAY=:0.0
# Export the Oracle library to the Top level directory
#export LD_LIBRARY_PATH=/usr/lib/oracle/12.2/client64/lib/${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
module purge
module load NeSI
module load Anaconda3/2019.03-gimkl-2018b
module load cuDNN/8.1.1.33-CUDA-11.2.0
module load CDO/1.9.5-GCC-7.1.0
/nesi/project/niwa00004/rampaln/bin/python "$@"
