#!/bin/bash -l

module use /opt/nesi/modulefiles/
module unuse /opt/niwa/CS500_centos7_skl/modules/all
module unuse /opt/niwa/share/modules/all

export SYSTEM_STRING=CS500
export OS_ARCH_STRING=centos7
export CPUARCH_STRING=skl
export PYTHONUSERBASE=/nesi/project/niwa00018/rampaln/conda_tmp
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib64
#module purge # optional
#module load CDO/1.9.5-GCC-7.1.0
#module load Miniforge3

# Navigate to the code directory
CODE_DIR="/nesi/project/niwa00018/queenle/NIWA-REMS_inference"
cd $CODE_DIR

# Define arrays
ml_models=("NIWA-REMS_tasmax_v050425")
# "NIWA-REMS_v110425_pr"  "NIWA-REMS_sfcWindmax_v050425" "NIWA-REMS_sfcWind_v050425" "NIWA-REMS_tasmin_v050425" 

gcms=("ACCESS-CM2" "EC-Earth3" "NorESM2-MM")
#"ACCESS-CM2" "EC-Earth3" "NorESM2-MM" "AWI-CM-1-1-MR" "CNRM-CM6-1"
variants=("r4i1p1f1" "r1i1p1f1" "r1i1p1f1")

#variant="NA"

variable="tasmax"
ssp="ssp370"
gan="GAN" #GAN, unet

for ml_model in "${ml_models[@]}"; do
    # Loop through GCMs
    for i in "${!gcms[@]}"; do

      gcm="${gcms[i]}"
      variant="${variants[i]}"

      sbatch -J "${ml_model}_${gcm}" apply_emulator_maui.sl "${ssp}" "${gcm}" "${variant}" "${variable}" "${ml_model}" "${gan}"
      echo "$ml_model $gcm $ssp $variable"
      
    done
done
