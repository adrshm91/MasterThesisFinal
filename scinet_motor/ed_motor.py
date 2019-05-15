import numpy as np
import _pickle as cPickle
import gzip
import scinet_motor.io_nn as io
import pandas as pd
import glob
from tqdm import tqdm
import tensorflow as tf
import matplotlib.pyplot as plt
import os

data_dir = "D:\\MeasurementsNew\\"
files = glob.glob(data_dir + "*.txt")
number_of_cols = 9
number_of_rows = 64000 * 20


def create_motor_dataset_all(series_length, N=None, file_name=None):
    np_array_list = []
    for file in tqdm(files):
        df = pd.read_csv(file)
        index = pd.date_range('1/1/2000', periods=df.shape[0], freq='0.0000625S')
        df.set_index(index, inplace=True)
        np_array = df.to_numpy()
        np_array_list.append(np_array)
    np_array_all = np.array(np_array_list).reshape((len(np_array_list)*number_of_rows, number_of_cols))
    np_array = np_array_all[:, [1, 3, 7]]
    np_array_split = np.array(np.array_split(np_array, np_array.shape[0] // series_length))
    speed = np_array_split[:, :, 0]
    current = np_array_split[:, :, 1]
    torque = np_array_split[:, :, 2]
    speed_torque = np.reshape(np.dstack([speed, torque]), (speed.shape[0], series_length*2))
    current_input = current[:, 0].reshape((current.shape[0],1))
    data = [current_input, speed_torque[:, :2], speed_torque[:, 2:series_length*2]]
    result = (data, current, speed, torque)
    if file_name is not None:
        f = gzip.open(io.data_path + file_name + ".plk.gz", 'wb')
        cPickle.dump(result, f, protocol=2)
        f.close()
    return result


def create_motor_dataset(series_length, N=None, file_name=None):
    file = data_dir + "latent space\\" + "M7VerschleissStangeS4Nm.txt"
    df = pd.read_csv(file)
    index = pd.date_range('1/1/2000', periods=df.shape[0], freq='0.0000625S')
    df.set_index(index, inplace=True)
    np_array_all = df.to_numpy()
    np_array = np_array_all[:, [1, 3, 7]]
    np_array_split = np.array(np.array_split(np_array, np_array.shape[0] // series_length))
    speed = np_array_split[:, :, 0]
    current = np_array_split[:, :, 1]
    torque = np_array_split[:, :, 2]
    speed_torque = np.reshape(np.dstack([speed, torque]), (speed.shape[0], series_length*2))
    current_input = current[:, 0].reshape((current.shape[0],1))
    data = [current_input, speed_torque[:, :2], speed_torque[:, 2:series_length*2]]
    result = (data, current, speed, torque)
    if file_name is not None:
        f = gzip.open(io.data_path + file_name + ".plk.gz", 'wb')
        cPickle.dump(result, f, protocol=2)
        f.close()
    return result


def create_motor_dataset_low_freq(series_length, N=None, file_name=None):
    file = data_dir + "M2LagerschadS4Nm.txt"
    df = pd.read_csv(file)
    speed = []
    torque = []
    for i in range(20):
        df_sample = df[0 + i * 64000:64000 + i * 64000]
        index = pd.date_range('1/1/2000', periods=64000, freq='0.0000625S')
        df_sample.set_index(index, inplace=True)
        df_downsampled = df_sample.asfreq('0.08S')
        speed_i = df_downsampled['Drehzahl'].to_numpy()
        torque_i = df_downsampled['Drehmoment'].to_numpy()
        speed.append(speed_i)
        torque.append(torque_i)
        print('step' + str(i))
    speed = np.vstack(speed)
    torque = np.vstack(torque)
    data = np.dstack([speed, torque])
    data = np.reshape(data, [N, 2 * series_length], order='C')
    result = (data, speed, torque)
    if file_name is not None:
        f = gzip.open(io.data_path + file_name + ".plk.gz", 'wb')
        cPickle.dump(result, f, protocol=2)
        f.close()
    return result


def load(validation_size_p, file_name):
    """
    Params:
    validation_size_p: percentage of data to be used for validation
    file_name (str): File containing the data
    """
    f = gzip.open(io.data_path + file_name + ".plk.gz", 'rb')
    data, speed, torque = cPickle.load(f)
    train_val_separation = int(len(data) * (1 - validation_size_p / 100.))
    training_data = data[:train_val_separation]
    validation_data = data[train_val_separation:]
    f.close()
    return training_data, validation_data


def load_speed_torque_current(validation_size_p, file_name):
    """
    Params:
    validation_size_p: percentage of data to be used for validation
    file_name (str): File containing the data
    """
    f = gzip.open(io.data_path + file_name + ".plk.gz", 'rb')
    data, current, speed, torque = cPickle.load(f)
    train_val_separation = int(len(data[0]) * (1 - validation_size_p / 100.))
    training_data = [data[i][:train_val_separation] for i in [0, 1, 2]]
    validation_data = [data[i][train_val_separation:] for i in [0, 1, 2]]
    f.close()
    return training_data, validation_data
