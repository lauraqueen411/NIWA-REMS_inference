#!/bin/bash -l
#SBATCH --job-name=maui_GPU_job
#SBATCH --partition=niwa_work
#SBATCH --time=00:59:00
#SBATCH --cluster=maui_ancil
#SBATCH --mem=180G
#SBATCH --gpus-per-node=A100:1
#SBATCH --cpus-per-task=2
#SBATCH --account=niwap03712
#SBATCH --mail-type=ALL
#SBATCH --output log/%j-%x.out
#SBATCH --error log/%j-%x.out
#SBATCH --exclude=wmg102

module purge # optional
module load NeSI
module load cuDNN/8.1.1.33-CUDA-11.2.0
nvidia-smi

# directories
code_dir="/nesi/project/niwa00018/queenle/NIWA-REMS_inference"
input_dir="/nesi/project/niwa03712/CMIP6_data/Downscaled_Preprocessed"
output_dir='/nesi/project/niwa00018/queenle/NIWA-REMS_inference/output'
cd $code_dir

# Arguments passed to the script
# $1: Scenario string (e.g., historical, ssp370)
# $2: GCM name (e.g., ACCESS-ESM1-5)
# $3: Realization string (e.g., r37i1p1f1)
# $4: Variable (e.g., tasmax)
# $5: Emulator name (e.g. NIWA-REMS_pr_v280125)
# $6: Emulator type (e.g. GAN, unet)

ssp=$1
gcm=$2
variant=$3
variable=$4
ml_model=$5
gan=$6

# apply to historical period in imperfect framework
/nesi/project/niwa00018/rampaln/envs/ml_env_v2/bin/python run_model.py $gan $variable $input_dir $gcm "historical" $variant $output_dir $ml_model $code_dir "imperfect"

# apply to ssp period in imperfect framework
/nesi/project/niwa00018/rampaln/envs/ml_env_v2/bin/python run_model.py $gan $variable $input_dir $gcm $ssp $variant $output_dir $ml_model $code_dir "imperfect"

# apply to historical+ssp period in perfect framework
/nesi/project/niwa00018/rampaln/envs/ml_env_v2/bin/python run_model.py $gan $variable $input_dir $gcm $ssp $variant $output_dir $ml_model $code_dir "perfect"

# compute metrics
/nesi/project/niwa00018/rampaln/envs/ml_env_v2/bin/python compute_metrics.py $ml_model $gcm $variable $ssp $output_dir $gan