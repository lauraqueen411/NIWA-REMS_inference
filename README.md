# NIWA-REMS Inference

This repository contains the code to apply the latest (as of June 2025) version of the NIWA-REMS RCM emulator and
produce supplementary figures for the manuscript titled, **"Downscaling with AI reveals large role of internal variability in fine-scale projections of climate extremes"**

---

## Quick Start

### Clone the Repository

```bash
git clone https://github.com/lauraqueen411/ML_emulator_temporal_sampling_experiments.git
````

**Note:** This repository is *not fully self-contained*. Essential inference input data are located on the **Maui** system and described below.

---

## Project Structure

This repository is organized into three main components:

1. **Inference** – Emulator application and high-level metric computation
2. **Plotting** – Analysis and figure generation
3. **data_for_Steve** – processed output for Steve's cyclone case studies

---

## Python Environment

All training, inference, and plotting scripts call the following python executable:

```
/nesi/project/niwa00018/rampaln/envs/ml_env_v2/bin/python
```

A reproducible environment file for the `ml_env_v2` Conda environment **with no builds** is included:
```
environment.yml
```

### Jupyter Kernel

All notebooks use the **"jupyter\_env"** kernel:
```
/scale_wlg_persistent/filesets/home/queenle/.local/share/jupyter/kernels/nellys_env/
```

A copy of this kernel is included:
```
kernel_copy/
```

---

## Emulator Inference (/inference)

### Main Inference Scripts

* **submit\_runs.sh**
  * Submits Slurm jobs for inference
  * Note: must manually edit variables to select GCMs, emulators, and model type (GAN/U-Net)
* **apply\_emulator\_maui.sl**
  * Slurm configuration for Maui
* **run\_model.py**
  * Main inference script
* **compute\_metrics.py**
  * Computes climatological metrics after inference
  * Metrics include annual/seasonal means, rx1d, total max
  * Historical (1985-2004) and future (2080-2099) climatologies computed

---

### Inference Inputs

* Emulator config from:
  ```
  /emulators
  ```
* Perfect (coarsened RCM) inputs:
  ```
  /nesi/project/niwa00018/ML_downscaling_CCAM/multi-variate-gan/inputs/predictor_fields/
  ```
* Imperfect (raw GCM) inputs:
  ```
  /nesi/project/niwa03712/CMIP6_data/Downscaled_Preprocessed/
  ```

---

### Inference Outputs (/inference/output/)

* Structure:
  ```
  /{GCM}/{emulator}/
      - NetCDF files of downscaled simulations
      - Separated by historical/ssp370, perfect/imperfect, GAN/U-Net

  /metrics/{GCM}/{emulator}/
      - NetCDF files of metric climatologies
  ```

---

## Plotting (/plotting)

### Figure Notebooks

* **plot_error_maps.ipynb**
    * code for creating the CCAM/Emulator validation supplementary map figures
    * all maps saved to /plotting/maps/
---

