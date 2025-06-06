import xarray as xr
import sys
import matplotlib.pyplot as plt
import os
import tqdm
from tqdm import tqdm
import tqdm
import tensorflow as tf
import numpy as np
from functools import partial

AUTOTUNE = tf.data.experimental.AUTOTUNE
from dask.diagnostics import ProgressBar
import json
import pandas as pd
sys.path.append(os.getcwd())
from src.layers import *
from src.models import *
from src.gan import *
from tensorflow.keras import layers
from src.process_input_training_data import *

def prepare_ML_inputs(GCM_input_path, config, framework, gcm, spectral_filtering = False, method ='normal_means'):
    """Modified on 06/04/25"""
    with ProgressBar():
        ds = xr.open_mfdataset(GCM_input_path).load()
        
        if framework == 'perfect':
            ds = ds.sel(GCM=gcm)

        print('\t- processing GCM input data')
        processed_GCM_data, means, stds = reformat_GCM_data(ds, config, framework, spectral_filtering = spectral_filtering, method = method)
        processed_GCM_data = processed_GCM_data.load()
        means = means.load()
        stds = stds.load()
        print('\t- processing mean, variance, and time data')
        mean_data, variance_data, time_of_year = process_mean_variance_time(ds, config, framework, means, stds)
        mean_data = mean_data.load()
        variance_data = variance_data.load()
        time_of_year = time_of_year.load()
        print('\t- processing static fields')
        vegt, orog, he = prepare_static_fields(config)
        vegt = vegt.load()
        orog = orog.load()
        he = he.load()
        print('\t- calculating time of year array')

    return (processed_GCM_data, mean_data, variance_data, vegt, orog, he, time_of_year)


def reformat_GCM_data(ds, config, framework, spectral_filtering = False, method ='doury'):
    if framework == 'imperfect':
        # Step 1: unstack pressure levels, change variable names
        print('\t\t- unstacking pressure levels')
        ds = unstack_pressure_levels(ds)

    # Step 2: normalize dataset by mean and st. dev.
    print('\t\t- normalizing by mean and standard deviation')
    ds, means, stds = normalize(ds, config, spectral_filtering = spectral_filtering, spectral_threshold=0.48, method =method)

    # Step 3: concatenate variable dimension
    print('\t\t- concatenating variables to channel dimension')
    da = concatenate_variable_dimension(ds, config)

    return da, means, stds


def unstack_pressure_levels(ds):
    unstacked_ds = ds.copy()

    var_name_dict = {'hus': 'q', 'ta': 't', 'ua': 'u', 'va': 'v'}

    for var in ['hus', 'ta', 'ua', 'va']:
        for lev in unstacked_ds.plev.values:
            if lev > 1e4:
                data = unstacked_ds[var].sel(plev=lev)
                lev = int(lev / 100)
            else:
                data = unstacked_ds[var].sel(plev=lev)

            unstacked_ds[f'{var_name_dict[var]}_{int(lev)}'] = data

        unstacked_ds = unstacked_ds.drop([var])

    return (unstacked_ds)


def normalize(ds, config, spectral_filtering = False, spectral_threshold=0.48, method ='normal_means'):
    var_list = config['var_names']
    
    if spectral_filtering:
        print('\t\t\t- Spectral filtering of the input variables')
        ds_new = ds[var_list].copy() * np.nan
        for varname in var_list:
            ds_new[varname].data = low_pass_filter(ds[varname].transpose("time","lat","lon").values, cutoff = spectral_threshold)
            print(f"Spectrally modified variable: {varname}")
        ds = ds_new 
    if method == 'doury':    
        means = ds[var_list].mean(['lat', 'lon'])
        stds = ds[var_list].std(['lat', 'lon'])
        norm_ds = (ds[var_list] - means) / stds
    else:
        means_NORM = xr.open_dataset(config["mean"])
        stds_NORM = xr.open_dataset(config["std"])
        norm_ds = (ds[var_list] - means_NORM.mean(["lat","lon"])) / stds_NORM.mean(["lat","lon"])
        means = ds[var_list].mean(['lat', 'lon'])
        stds = ds[var_list].std(['lat', 'lon'])

    return norm_ds, means, stds



def concatenate_variable_dimension(ds, config):
    var_list = config['var_names']

    # concatenate dataarrays of each variables along dimension 'channel'
    concatenated_da = xr.concat([ds[var] for var in var_list], dim="channel")
    concatenated_da = concatenated_da.rename('GCM_da')

    # name channel dimension values by variables names
    concatenated_da['channel'] = (('channel'), var_list)

    return (concatenated_da)


def process_mean_variance_time(ds, config, framework, GCM_spatial_means, GCM_spatial_stds):
    var_list = config['var_names']

    if framework == 'imperfect':
        ds = unstack_pressure_levels(ds)

    ds = ds[var_list]

    # single values
    predictor_means_mean = xr.open_dataset(config["input_means_means"])
    predictor_means_variance = xr.open_dataset(config["input_means_stds"])

    predictor_stds_mean = xr.open_dataset(config["input_stds_means"])
    predictor_stds_variance = xr.open_dataset(config["input_stds_stds"])

    # time series
    # = ds.mean(['lat', 'lon'])
    #GCM_spatial_stds = ds.std(['lat', 'lon'])

    # stack normalized means
    norm_spatial_means = (GCM_spatial_means - predictor_means_mean) / predictor_means_variance
    norm_spatial_means = xr.concat([norm_spatial_means[i] for i in var_list], dim="channel")
    norm_spatial_means['channel'] = (('channel'), var_list)

    # stack normalized st. deviations
    norm_spatial_stds = (GCM_spatial_stds - predictor_stds_mean) / predictor_stds_variance
    norm_spatial_stds = xr.concat([norm_spatial_stds[i] for i in var_list], dim="channel")
    norm_spatial_stds['channel'] = (('channel'), var_list)

    time_of_year = np.sin(2 * np.pi * norm_spatial_means.time.dt.dayofyear / 365)

    return (norm_spatial_means, norm_spatial_stds, time_of_year)


def prepare_static_fields(config):
    topography_data = xr.open_dataset(config["static_predictors"])

    vegt = topography_data.vegt
    orog = topography_data.orog
    he = topography_data.he

    # normalize to the range [0,1]
    vegt = (vegt - vegt.min()) / (vegt.max() - vegt.min())
    orog = (orog - orog.min()) / (orog.max() - orog.min())
    he = (he - he.min()) / (he.max() - he.min())

    return (vegt, orog, he)


def initialize_output_ds(input_ds, config):
    print('\t- initializing output data structure')

    example_output = xr.open_dataset(config['train_y'])

    try:
        example_output = example_output.isel(GCM=0)[['pr']]
    except:
        example_output = example_output[['pr']]

    output_shape = example_output.isel(time=0).drop(['time'])
    output_shape = output_shape.expand_dims({"time": input_ds.time.size})
    output_shape['time'] = (('time'), input_ds.time.to_index())

    output_shape.pr.values = output_shape.pr.values * 0

    return (output_shape)

def create_output(X, y):
    y = y.isel(time=0).drop("time")
    y = y.expand_dims({"time": X.time.size})
    y['time'] = (('time'), X.time.to_index())
    return y
# changed activation function to hyperbolic tangent


def load_model_cascade(model_name, model_dir, load_unet=True):
    gan = tf.keras.models.load_model(f'{model_dir}/{model_name}/generator_final.h5',
                                     custom_objects={"BicubicUpSampling2D": BicubicUpSampling2D,
                                                     "SymmetricPadding2D": SymmetricPadding2D},
                                     compile=False)
    with open(f'{model_dir}/{model_name}/config_info.json') as f:
        config = json.load(f)

        unet = tf.keras.models.load_model(f'{model_dir}/{model_name}/unet_final.h5',
                                          custom_objects={"BicubicUpSampling2D": BicubicUpSampling2D,
                                                          "SymmetricPadding2D": SymmetricPadding2D}, compile=False)

    return gan, unet, config["ad_loss_factor"]


def load_data_historical(model, output_config_name):
    """
    Returns the predictor variables for inference, and also ground truth data for visualization
    """

    with ProgressBar():
        df = xr.open_dataset(f'{output_config_name["eval_predictor_variables"]}/{model}_histupdated.nc',
                             chunks={"time": 365 * 10}).load()
        df = df.resample(time='1D').mean()
        df['time'] = pd.to_datetime(df.time.dt.strftime("%Y-%m-%d"))
    # df = df.resample(time ='1D').mean()

    # Load and process CCAM outputs
    # y_ssp370 = xr.open_dataset(f'{ccam_output_path}{model}_ssp370_precip.nc')
    y_historical = xr.open_dataset(f'{output_config_name["ground_truth"]}/{model}_historical_precip.nc')
    y_historical['time'] = pd.to_datetime(y_historical.time.dt.strftime("%Y-%m-%d"))
    return df, y_historical


def load_and_normalize_topography_data(filepath):
    # Load the dataset
    topography_data = xr.open_dataset(filepath)

    # Extract variables
    vegt = topography_data.vegt
    orog = topography_data.orog
    he = topography_data.he

    # Print maximum values
    print(f"Max orog: {orog.max().values}, Max he: {he.max().values}, Max vegt: {vegt.max().values}")

    # Normalize the data to the range [0, 1]
    vegt = (vegt - vegt.min()) / (vegt.max() - vegt.min())
    orog = (orog - orog.min()) / (orog.max() - orog.min())
    he = (he - he.min()) / (he.max() - he.min())

    return vegt, orog, he


def normalize_and_stack(concat_dataset, means_filepath, stds_filepath, variables):
    """
    Normalizes specified variables in a dataset with given mean and standard deviation,
    then stacks them along a new 'channel' dimension.

    Parameters:
    concat_dataset (xarray.Dataset): Dataset to normalize.
    means_filepath (str): File path to the dataset containing mean values.
    stds_filepath (str): File path to the dataset containing standard deviation values.
    variables (list): List of variable names to normalize and stack.

    Returns:
    xarray.Dataset: The normalized and stacked dataset.
    """

    # Load mean and standard deviation datasets
    means = xr.open_dataset(means_filepath)
    stds = xr.open_dataset(stds_filepath)

    # Normalize the dataset
    X_norm = (concat_dataset[variables] - means) / stds
    X_norm['time'] = pd.to_datetime(X_norm.time.dt.strftime("%Y-%m-%d"))

    # Stack the variables along a new 'channel' dimension
    stacked_X = xr.concat([X_norm[varname] for varname in variables], dim="channel")
    stacked_X['channel'] = (('channel'), variables)
    stacked_X = stacked_X.transpose("time", "lat", "lon", "channel")

    return stacked_X


@tf.function
def predict_batch(model, latent_vectors, data_batch, orog, he, vegt, model_type):
    if model_type == 'GAN':
        return model([latent_vectors[0], data_batch, orog, he, vegt], training=False)
    else:
        return model(data_batch, training=False)


def expand_conditional_inputs(X, batch_size):
    expanded_image = tf.expand_dims(X, axis=0)  # Shape: (1, 172, 179)

    # Repeat the image to match the desired batch size
    expanded_image = tf.repeat(expanded_image, repeats=batch_size, axis=0)  # Shape: (batch_size, 172, 179)

    # Create a new axis (1) on the last axis
    expanded_image = tf.expand_dims(expanded_image, axis=-1)
    return expanded_image


# Example usage:
@tf.function
def predict_batch_residual(model, unet, latent_vectors, data_batch, orog, he, vegt, model_type):
    if model_type == 'GAN':
        intermediate = unet([latent_vectors[0], data_batch, orog, he, vegt], training=False)
        # intermediate = apply_gaussian_blur(intermediate, size=7, sigma=1.5)
        # max_value = tf.reduce_max(intermediate, axis=(1, 2, 3), keepdims=True)
        # min_value = tf.reduce_min(intermediate, axis=(1, 2, 3), keepdims=True)
        init_prediction = intermediate
        # print(intermediate)
        # intermediate = tf.cast(tf.math.sqrt(tf.clip_by_value(intermediate, clip_value_min=0, clip_value_max=2500)), 'float32')
        return model([latent_vectors[0], data_batch, orog, he, vegt, init_prediction],
                     training=False) + intermediate  # +
    else:
        return unet([latent_vectors[0], data_batch, orog, he, vegt], training=False)

@tf.function
def predict_batch_residual_v2(model, unet, latent_vectors, data_batch, orog, he, vegt, model_type):
    if model_type == 'GAN':
        intermediate = unet([latent_vectors[0], data_batch, orog, he, vegt], training=False)
        # intermediate = apply_gaussian_blur(intermediate, size=7, sigma=1.5)
        # max_value = tf.reduce_max(intermediate, axis=(1, 2, 3), keepdims=True)
        # min_value = tf.reduce_min(intermediate, axis=(1, 2, 3), keepdims=True)
        init_prediction = intermediate
        # print(intermediate)
        # intermediate = tf.cast(tf.math.sqrt(tf.clip_by_value(intermediate, clip_value_min=0, clip_value_max=2500)), 'float32')
        return model([latent_vectors[0], data_batch, orog, he, vegt, init_prediction],
                     training=False)# + intermediate  # +
    else:
        return unet([latent_vectors[0], data_batch, orog, he, vegt], training=False)

def predict_parallel_resid(model, unet, inputs, output_shape, batch_size, orog_vector, he_vector, vegt_vector,model_type='GAN', output_add_factor =1):
    n_iterations = inputs.shape[0] // batch_size
    remainder = inputs.shape[0] - n_iterations * batch_size

    dset = []

    with tqdm.tqdm(total=n_iterations, desc="Predicting", unit="batch") as pbar:
        for i in range(n_iterations):
            data_batch = inputs[i * batch_size: (i + 1) * batch_size]
            random_latent_vectors1 = tf.random.normal(shape=(batch_size,) + tuple(model.inputs[0].shape[1:]))
            #random_latent_vectors1 = tf.repeat(random_latent_vectors1, repeats=batch_size, axis=0)
            orog = expand_conditional_inputs(orog_vector, batch_size)
            he = expand_conditional_inputs(he_vector, batch_size)
            vegt = expand_conditional_inputs(vegt_vector, batch_size)

            output = predict_batch_residual(model, unet, [random_latent_vectors1], data_batch, orog, he, vegt,
                                            model_type)

            dset += (np.exp(output.numpy()[:, :, :, 0]) - output_add_factor).tolist()
            pbar.update(1)  # Update the progress bar

    if remainder != 0:
        random_latent_vectors1 = tf.random.normal(shape=(batch_size,) + tuple(model.inputs[0].shape[1:]))
        #random_latent_vectors1 = tf.repeat(random_latent_vectors1, repeats=batch_size, axis=0)
        orog = expand_conditional_inputs(orog_vector, remainder)
        he = expand_conditional_inputs(he_vector, remainder)
        vegt = expand_conditional_inputs(vegt_vector, remainder)

        output = predict_batch_residual(model, unet, [random_latent_vectors1[:remainder]],
                                        inputs[inputs.shape[0] - remainder:], orog, he, vegt, model_type)

        dset += (np.exp(output.numpy()[:, :, :, 0]) - output_add_factor).tolist()
    output_shape['pr'].values = dset

    return output_shape

@tf.function
def predict_batch_residual_v280325(model, unet, latent_vectors, data_batch, orog, he, vegt, model_type):
    if model_type =='GAN':
        intermediate =unet([data_batch, orog], training=False)
        #intermediate = apply_gaussian_blur(intermediate, size=7, sigma=1.5)
        #max_value = tf.reduce_max(intermediate, axis=(1, 2, 3), keepdims=True)
        #min_value = tf.reduce_min(intermediate, axis=(1, 2, 3), keepdims=True)
        init_prediction = intermediate#(intermediate - min_value)/(max_value - min_value)
        #print(intermediate)
        #intermediate = tf.cast(tf.math.sqrt(tf.clip_by_value(intermediate, clip_value_min=0, clip_value_max=2500)), 'float32')
        return  model([latent_vectors[0], latent_vectors[1],  data_batch, orog, init_prediction], training=False) +intermediate#+
    else:
        return unet([data_batch, orog], training=False)
    
def predict_parallel_resid_v280325(model, unet, inputs, output_shape, batch_size, orog_vector, he_vector, vegt_vector,model_type = 'GAN', output_add_factor = 1, varname = "pr", output_means = None, output_stds = None):
    n_iterations = inputs.shape[0] // batch_size
    remainder = inputs.shape[0] - n_iterations * batch_size

    dset = []

    with tqdm.tqdm(total=n_iterations, desc="Predicting", unit="batch") as pbar:
        for i in range(n_iterations):
            data_batch = inputs[i * batch_size: (i + 1) * batch_size]
            random_latent_vectors1 = tf.random.normal(shape=(batch_size,) + tuple(model.inputs[0].shape[1:]))
            random_latent_vectors2 = tf.random.normal(shape=(batch_size,) + tuple(model.inputs[1].shape[1:]))
            #random_latent_vectors1 = tf.repeat(random_latent_vectors1, repeats=batch_size, axis=0)
            orog = expand_conditional_inputs(orog_vector, batch_size)
            he = expand_conditional_inputs(he_vector, batch_size)
            vegt = expand_conditional_inputs(vegt_vector, batch_size)

            output = predict_batch_residual_v280325(model, unet, [random_latent_vectors1, random_latent_vectors2], data_batch, orog, he, vegt,model_type)
            if varname == "pr":
                dset += (np.exp(output.numpy()[:, :, :, 0]) - output_add_factor).tolist()
            elif "sfcWind" in varname:
                default_name = "sfcWind"
                dset += (output.numpy()[:, :, :, 0] * output_stds[default_name].mean().values +output_means[default_name].mean().values).tolist()
            else:    
                dset += (output.numpy()[:, :, :, 0] * output_stds[varname].mean().values +output_means[varname].mean().values).tolist()
            pbar.update(1)  # Update the progress bar

    if remainder != 0:
        random_latent_vectors1 = tf.random.normal(shape=(batch_size,) + tuple(model.inputs[0].shape[1:]))
        random_latent_vectors2 = tf.random.normal(shape=(batch_size,) + tuple(model.inputs[1].shape[1:]))
        #random_latent_vectors1 = tf.repeat(random_latent_vectors1, repeats=batch_size, axis=0)
        orog = expand_conditional_inputs(orog_vector, remainder)
        he = expand_conditional_inputs(he_vector, remainder)
        vegt = expand_conditional_inputs(vegt_vector, remainder)

        output = predict_batch_residual_v280325(model, unet, [random_latent_vectors1[:remainder], random_latent_vectors2[:remainder]],
                                        inputs[inputs.shape[0] - remainder:], orog, he, vegt, model_type)

        if varname == "pr":
            dset += (np.exp(output.numpy()[:, :, :, 0]) - output_add_factor).tolist()
        elif "sfcWind" in varname:
            default_name = "sfcWind"
            dset += (output.numpy()[:, :, :, 0] * output_stds[default_name].mean().values +output_means[default_name].mean().values ).tolist()
        else:    
            dset += (output.numpy()[:, :, :, 0] * output_stds[varname].mean().values +output_means[varname].mean().values).tolist()
    output_shape[varname].values = dset

    return output_shape

def predict_parallel_resid_t(model, unet, inputs, output_shape, batch_size, orog_vector, he_vector, vegt_vector,
                           model_type='GAN', output_means = None, output_stds = None, config = None):
    n_iterations = inputs.shape[0] // batch_size
    remainder = inputs.shape[0] - n_iterations * batch_size

    dset = []

    with tqdm.tqdm(total=n_iterations, desc="Predicting", unit="batch") as pbar:
        for i in range(n_iterations):
            data_batch = inputs[i * batch_size: (i + 1) * batch_size]
            random_latent_vectors1 = tf.random.normal(shape=(1,) + tuple(model.inputs[0].shape[1:]))
            random_latent_vectors1 = tf.repeat(random_latent_vectors1, repeats=batch_size, axis=0)
            orog = expand_conditional_inputs(orog_vector, batch_size)
            he = expand_conditional_inputs(he_vector, batch_size)
            vegt = expand_conditional_inputs(vegt_vector, batch_size)

            output = predict_batch_residual(model, unet, [random_latent_vectors1], data_batch, orog, he, vegt,
                                            model_type)

            dset += ((output.numpy()[:, :, :, 0]) * (output_stds['tasmax'].values) + output_means[
                'tasmax'].values).tolist()

            pbar.update(1)  # Update the progress bar

    if remainder != 0:
        random_latent_vectors1 = tf.random.normal(shape=(1,) + tuple(model.inputs[0].shape[1:]))
        random_latent_vectors1 = tf.repeat(random_latent_vectors1, repeats=batch_size, axis=0)
        orog = expand_conditional_inputs(orog_vector, remainder)
        he = expand_conditional_inputs(he_vector, remainder)
        vegt = expand_conditional_inputs(vegt_vector, remainder)

        output = predict_batch_residual(model, unet, [random_latent_vectors1[:remainder]],
                                        inputs[inputs.shape[0] - remainder:], orog, he, vegt, model_type)

        dset += ((output.numpy()[:, :, :, 0]) * (output_stds['tasmax'].values) +
                 output_means[
                     'tasmax'].values).tolist()
    output_shape['pr'].values = dset

    return output_shape


class ValidationMetric(object):
    """
    This is a class that computes a wide variety of different metrics for validating a series
    """

    def __init__(self, datasets):
        self.ds = datasets

    def __call__(self, thresh):
        print("Computing Indices.....annual_rainfall")
        annual_rainfall = self.seasonal_rainfall(self.ds)
        print("Computing Indices.....CDD")
        cdd = self.consecutive_dry_days(self.ds, thresh)
        print("Computing Indices.....RX3DAY")
        rx3day = self.rx1day(self.ds, thresh)

        print("Computing R10 Day.....")
        r10day = self.r10day(self.ds)
        merged_df = xr.merge([cdd, rx3day, annual_rainfall, r10day])
        return merged_df

    @staticmethod
    def consecutive_dry_days(ds, thresh=1):
        """
        Compute the number of consecutive dry days in a year for a gridded dataset

        Parameters:
        ds (xarray.Dataset): Gridded dataset with a time dimension
        thresh (float): Threshold value for defining a dry day (default: 0.1)
        time_dim (str): Name of the time dimension in the dataset (default: 'time')

        Returns:
        xarray.DataArray: Number of consecutive dry days in a year
        """

        # Create a function to find consecutive True values in a boolean array
        def find_consecutive_true(arr):
            if ((arr.max() == 1) & (arr.min() == 0)) | (arr.min() == 1):
                arr = np.asarray(arr)
                idx = np.flatnonzero(np.concatenate(([arr[0]],
                                                     arr[:-1] != arr[1:],
                                                     [True])))

                z = np.diff(idx)[::2]
                return np.max(z, axis=0)
            else:
                # this condition implies that there are no CDD throughout a year?
                return 0.0

        test_data = ds.pr
        try:
            test_data = test_data.stack(z=['lat', 'lon']).dropna("z")
        except:
            test_data = test_data.stack(z=['latitude', 'longitude']).dropna("z")
        # fillna(-999)
        bool_arr = (test_data <= thresh).astype('int')
        bool_arr = bool_arr
        with ProgressBar():
            consec_dry_days = xr.apply_ufunc(find_consecutive_true, bool_arr.groupby('time.year'),
                                             input_core_dims=[["time"]], output_core_dims=[[]],
                                             output_dtypes=[int], vectorize=True, dask='parallelized').compute()
        consec_dry_days = consec_dry_days.unstack()
        try:
            consec_dry_days = consec_dry_days.reindex(lat=sorted(consec_dry_days.lat.values))

        except:
            consec_dry_days = consec_dry_days.reindex(longitude=sorted(consec_dry_days.longitude.values))

        output = consec_dry_days

        return output.to_dataset().rename({"pr": "cdd"})

    @staticmethod
    def rx1day(ds, thresh=1):
        """
        Compute the Rx3day index for a gridded dataset

        Parameters:
        ds (xarray.Dataset): Gridded dataset with a time dimension
        thresh (float): Threshold value for defining a wet day (default: 0.1)
        time_dim (str): Name of the time dimension in the dataset (default: 'time')

        Returns:
        xarray.DataArray: Rx3day index for the dataset
        """
        return ds.groupby('time.year').max().rename({"pr": "rx1day"})

    @staticmethod
    def seasonal_rainfall(ds):
        output = ds.groupby('time.season').mean()  # .mean("year")
        output1 = output.sel(season='DJF').drop("season").rename({"pr": "DJF_rainfall"})
        output2 = output.sel(season='JJA').drop("season").rename({"pr": "JJA_rainfall"})
        output = xr.merge([output1, output2])

        return output

    @staticmethod
    def r10day(ds):
        output = (ds > 10).groupby('time.year').sum()  # .mean("year")
        return output.rename({"pr": "r10day"})


def run_experiments(experiments, epoch_list, model_dir,
                    input_predictors, common_times, output_shape, orog, he, vegt, n_members, batch_size=64):
    """
    Runs inference on some predictor fields in/ out-of-sample


    experiments: list of experiment names in the model_dir folder
    input_predictors: stacked netcdf with dims (time, lat, lon, channel) and normalized data
    common_times: common_times between output_shape data and input_predictors
    output_shape: a netcdf (y_true) that is the same shape as the output prediction, it contains the time metadata
    orog, he, vegt: auxiliary files from ccam
    n_member: the number of ensemble members

    """

    # if the epoch list is only a float convert to a list
    if isinstance(epoch_list, int):
        epoch_list = [epoch_list] * len(experiments)

    # creating empty lists to save outputs
    dsets = []
    lambda_var = []
    for i, experiment in enumerate(experiments):
        if 'cascade' in experiment:
            gan, unet, lambdas = load_model_cascade(experiment,
                                                    epoch_list[i], model_dir)
            if i == 0:
                # first instance is always a unet model
                lambdas = 0.0
                preds = xr.concat([predict_parallel_resid(gan, unet,
                                                          input_predictors.sel(time=common_times).values,
                                                          output_shape.sel(time=common_times),
                                                          batch_size, orog, he, vegt, model_type='unet')
                                   for i in range(n_members)],
                                  dim="member")
            else:
                # do not change lambdas value otherwise
                lambdas = lambdas
                preds = xr.concat([predict_parallel_resid(gan, unet,
                                                          input_predictors.sel(time=common_times).values,
                                                          output_shape.sel(time=common_times),
                                                          batch_size, orog, he, vegt, model_type='GAN')
                                   for i in range(n_members)],
                                  dim="member")


        else:
            gan, lambdas = load_model_reg(experiment,
                                          epoch_list[i], model_dir)
            preds = xr.concat([predict_parallel_v1(gan,
                                                   input_predictors.sel(time=common_times).values,
                                                   output_shape.sel(time=common_times),
                                                   batch_size, orog, he, vegt, model_type='GAN')
                               for i in range(n_members)],
                              dim="member")
        lambda_var.append(lambdas)
        dsets.append(preds)
    dsets = xr.concat(dsets, dim="experiment")
    dsets['experiment'] = (('experiment'), lambda_var)
    dsets = dsets.reindex(experiment=sorted(dsets.experiment.values))
    return dsets



