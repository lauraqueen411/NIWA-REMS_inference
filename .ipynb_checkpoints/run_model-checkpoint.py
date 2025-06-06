import os
import sys
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
import tqdm
from functools import partial
import json
import tensorflow as tf
from tensorflow.keras import layers
# AUTOTUNE = tf.data.experimental.AUTOTUNE
from dask.diagnostics import ProgressBar
import pathlib


inp = sys.argv
gan_flag = sys.argv[1]
variable = sys.argv[2]
input_data_path_base = sys.argv[3]
gcm = sys.argv[4]
ssp = sys.argv[5]
variant = sys.argv[6]
output_dir = sys.argv[7]
ml_model_name = sys.argv[8]
code_dir = sys.argv[9]
framework = sys.argv[10]
output_norm =1
emulator_config_dir = '/nesi/project/niwa00018/queenle/NIWA-REMS_inference/emulators'

sys.path.append(f'{code_dir}')
os.chdir(code_dir)
from src.layers import *
from src.models import *
from src.gan import *
from src.src_eval_inference import *

'''
If file exists, skip
'''

if os.path.exists(f'{output_dir}/{gcm}/{ml_model_name}/{gcm}_{variable}_{ssp}_{framework}_framework_{gan_flag}.nc'):
    print('downscaled file already exists')
    sys.exit()


'''
Define config file and directories
'''

print('BEGINNING EMULATOR DOWNSCALING')
print(f"current path: {os.getcwd()}, code_dir: {code_dir}")

print(f'{gcm} --- {ssp} --- {framework}\n')

if framework == 'imperfect':
    if 'ssp' in ssp:
        files = f'{input_data_path_base}/ScenarioMIP/*/{gcm}/{ssp}/{variant}/day/ScenarioMIP_*_{gcm}_{ssp}_{variant}_day*.nc'
    else:
        files = f'{input_data_path_base}/CMIP/*/{gcm}/{ssp}/{variant}/day/CMIP_*_{gcm}_{ssp}_{variant}_day*.nc'
    files = glob.glob(files, recursive =True)
    input_file = files[0]

elif framework == 'perfect':
    input_file = '/nesi/project/niwa00018/ML_downscaling_CCAM/multi-variate-gan/inputs/predictor_fields/predictor_fields_hist_'+ssp+'_merged_updated.nc'

config_file = f'{emulator_config_dir}/{ml_model_name}/config_info.json'

print('config file: ' + config_file)

with open(config_file) as f:
    config = json.load(f)
    
output_means = xr.open_dataset(config["means_output"])
output_stds = xr.open_dataset(config["stds_output"])

# LOAD ML MODEL
print('LOAD ML MODEL\n')
gan, unet, adv_factor = load_model_cascade(ml_model_name, emulator_config_dir, load_unet=True)

# PREP INPUT
print('\nPREP INPUT')
processed_GCM_data, mean_data, variance_data, vegt, orog, he, time_of_year = prepare_ML_inputs(input_file,config,framework,gcm)

# PREP OUTPUT
print('PREP OUTPUT')
output_shape = initialize_output_ds(processed_GCM_data, config)
output_shape = output_shape.rename({"pr": variable})

# APPLY MODEL
print('APPLY ML MODEL')
output = predict_parallel_resid_v280325(gan, unet,
                                        processed_GCM_data.transpose('time', 'lat', 'lon','channel').values, \
                                        output_shape, 64, orog.values, he.values, vegt.values,\
                                        model_type=gan_flag,output_add_factor=output_norm,\
                                        varname = config["output_varname"],\
                                        output_means=output_means, output_stds = output_stds)


output['GCM'] = f'{gcm}'
output.coords['scenario'] = f'{ssp}'

if not os.path.exists(f'{output_dir}/{gcm}/{ml_model_name}/'):
    os.makedirs(f'{output_dir}/{gcm}/{ml_model_name}/')

output.to_netcdf(f'{output_dir}/{gcm}/{ml_model_name}/{gcm}_{variable}_{ssp}_{framework}_framework_{gan_flag}.nc')
print('\n\n')

