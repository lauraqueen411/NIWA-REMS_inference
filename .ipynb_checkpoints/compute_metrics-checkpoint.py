import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import glob
import os
import sys


def get_metric_da(da,period,metric):
    
    period_da = da.sel(time=slice(period[0],period[1]))
    
    if metric == 'rx1d':
        metric_da = period_da.groupby('time.year').max().mean('year').rename(metric+'_' + period[0] + '-' + period[1])
        
    elif metric == 'TXx':
        metric_da = period_da.groupby('time.year').max().mean('year').rename(metric+'_' + period[0] + '-' + period[1])
        
    elif metric == 'TXn':
        metric_da = period_da.groupby('time.year').min().mean('year').rename(metric+'_' + period[0] + '-' + period[1])
        
    elif metric == 'WXx':
        metric_da = period_da.groupby('time.year').max().mean('year').rename(metric+'_' + period[0] + '-' + period[1])

    elif metric == 'total_max':
        metric_da = period_da.max('time').rename(metric+'_' + period[0] + '-' + period[1])

    elif metric == 'annual_mean':
        metric_da = period_da.mean('time').rename(metric+'_' + period[0] + '-' + period[1])

    elif metric == 'SDII':
        metric_da = period_da.where(da>1).mean('time').rename(metric+'_' + period[0] + '-' + period[1])
        
    elif metric in ['DJF_mean','MAM_mean','JJA_mean','SON_mean']:
        s = metric.split('_')[0]
        metric_da = period_da.groupby('time.season').mean('time').sel(season=s).drop('season').rename(metric+'_' + period[0] + '-' + period[1])
            
    return(metric_da)


'''
Load ML output and CCAM 'Ground Truth'
'''

def load_CCAM_data(GCM,var,ssp):
    
    # CCAM dynamical downscaled output
    CCAM_downscaled_ds = xr.open_dataset(CCAM_dir + 'target_fields/target_fields_hist_'+ssp+'_concat.nc')
    CCAM_downscaled_da = CCAM_downscaled_ds.sel(GCM=GCM)[ccam_names[var]]
    if var == 'pr':
        CCAM_downscaled_da = CCAM_downscaled_da*86400 # convert from flux to mm/day
    
    if var == 'tasmin' or var == 'tasmax':
        CCAM_downscaled_da = CCAM_downscaled_da-272.15
        
    return(CCAM_downscaled_da)

def load_emulator_data(GCM,var,ssp,ml_model,ml_output_dir):

    # Emulator - imperfect
    imperfect_hist = xr.open_dataset(f'{ml_output_dir}/{GCM}/{ml_model}/{GCM}_{var}_historical_imperfect_framework_{gan_flag}.nc')[var]
    imperfect_future = xr.open_dataset(f'{ml_output_dir}/{GCM}/{ml_model}/{GCM}_{var}_{ssp}_imperfect_framework_{gan_flag}.nc')[var]
    imperfect_da = xr.concat([imperfect_hist,imperfect_future],dim='time')

    # Emulator - perfect
    perfect_da = xr.open_dataset(f'{ml_output_dir}/{GCM}/{ml_model}/{GCM}_{var}_{ssp}_perfect_framework_{gan_flag}.nc')[var]

    if var == 'tasmin' or var == 'tasmax':
        imperfect_da = imperfect_da-272.15
        perfect_da = perfect_da-272.15
        
    da_dict = {'imperfect':imperfect_da, 'perfect':perfect_da}

    return(da_dict)


def compute_metrics(da, var, base_period,future_period):
    da_list = []
    for period in [base_period,future_period]:
        print('\t\t\t\t- ' + period[0] + '-' + period[1])

        for metric in metrics[var]:
            print('\t\t\t\t\t- ' + metric)

            da_list.append(get_metric_da(da,period,metric))

    merged_ds = xr.merge(da_list)

    return(merged_ds)


def compute_save_CCAM(GCM,var,ml_output_dir):
    
    print('\t\tloading CCAM data')
    ccam_da = load_CCAM_data(GCM,var,ssp)
    print('\t\tcomputing metrics')
    metric_ds = compute_metrics(ccam_da,var,base_period,future_period)

    if not os.path.exists(f'{ml_output_dir}/metrics/{GCM}/CCAM/'):
        os.makedirs(f'{ml_output_dir}/metrics/{GCM}/CCAM/')

    metric_ds.to_netcdf(f'{ml_output_dir}/metrics/{GCM}/CCAM/{GCM}_{var}_metrics.nc')

        
def compute_save_emulator(GCM,var,emulator,ml_output_dir):
    
    print('\t\tloading emulator data: ' + GCM + ' ' + var + ' (' + emulator + ')')
    try:
        da_dict = load_emulator_data(GCM,var,ssp,emulator,ml_output_dir)
    except:
        print('\t\tunable to open all data, skipping')
        return

    for framework in da_dict:
        print('\t\t\tcalculating ' + framework + ' metrics')
        framework_da = da_dict[framework]

        metric_ds = compute_metrics(framework_da,var,base_period,future_period)

        if not os.path.exists(f'{ml_output_dir}/metrics/{GCM}/{emulator}/'):
            os.makedirs(f'{ml_output_dir}/metrics/{GCM}/{emulator}/')

        metric_ds.to_netcdf(f'{ml_output_dir}/metrics/{GCM}/{emulator}/{GCM}_{framework}_{var}_metrics_{gan_flag}.nc')

        
'''
___MAIN___ code

'''

CCAM_dir = '/nesi/project/niwa00018/ML_downscaling_CCAM/multi-variate-gan/inputs/'
static_ds = xr.open_dataset('/nesi/project/niwa00018/ML_downscaling_CCAM/training_GAN/ancil_fields/ERA5_eval_ccam_12km.198110_NZ_Invariant.nc')
land_mask = static_ds.sftlf

base_period = ['1985','2004']
future_period = ['2080','2099']

ccam_names = {'tasmax':'tasmax','tasmin':'tasmin','sfcwind':'sfcWind','sfcwindmax':'sfcWindmax','pr':'pr'}

metrics = {'tasmax':['annual_mean','TXx','DJF_mean','MAM_mean','JJA_mean','SON_mean'],
           'tasmin':['annual_mean','TXn','DJF_mean','MAM_mean','JJA_mean','SON_mean'],
           'sfcwind':['annual_mean','DJF_mean','MAM_mean','JJA_mean','SON_mean'],
           'sfcwindmax':['annual_mean','WXx','DJF_mean','MAM_mean','JJA_mean','SON_mean'],
           'pr':['annual_mean','DJF_mean','MAM_mean','JJA_mean','SON_mean','rx1d','total_max','SDII']}

inp = sys.argv
emulator = sys.argv[1]
gcm = sys.argv[2]
var = sys.argv[3]
ssp = sys.argv[4]
ml_output_dir = sys.argv[5]
gan_flag = sys.argv[6]

print(f'Computing: {emulator} {gcm} {var}')

if not os.path.exists(f'{ml_output_dir}/metrics/{gcm}/CCAM/{gcm}_{var}_metrics.nc'):
    compute_save_CCAM(gcm,var)
else:
    print('\t\tCCAM results exist: skipping')

if not (os.path.exists(f'{ml_output_dir}/metrics/{gcm}/{emulator}/{gcm}_imperfect_{var}_metrics_{gan_flag}.nc') and os.path.exists(f'{ml_output_dir}/metrics/{gcm}/{emulator}/{gcm}_perfect_{var}_metrics_{gan_flag}.nc')):
    compute_save_emulator(gcm,var,emulator,ml_output_dir)

else:
    print('\t\tEmulator results exists: skipping')

    
    