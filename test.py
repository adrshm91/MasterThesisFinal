import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scinet_motor import model_new as nn
import scinet_motor.ed_motor as edm
import pandas as pd
from tqdm import *
from sklearn.metrics import mean_squared_error
from math import sqrt
import scinet_motor.io_nn as io_nn
import os

series_length = 100
N = 20
L = 64000

net = nn.Network.from_saved('motor_speed_torque', change_params={'load_file': 'model_motor_speed_torque_epoch_195'})

# data, speed, torque = edm.create_motor_dataset(series_length, N)

data, current, speed, torque = edm.create_motor_dataset(series_length, N, file_name='motor_speed_torque_test')


b = net.run(data, net.decoded_list)


speed_given = np.empty((L, 1))
speed_given[:] = np.nan
for i, speed_i in enumerate(speed[:L//series_length, 0]):
    speed_given[i*series_length] = speed_i
current_given = np.empty((L, 1))
current_given[:] = np.nan
for i, current_i in enumerate(current[:L//series_length, 0]):
    current_given[i*series_length] = current_i
torque_given = np.empty((L, 1))
torque_given[:] = np.nan
for i, torque_i in enumerate(torque[:L//series_length, 0]):
    torque_given[i*series_length] = torque_i
speed_actual = speed.ravel()[:L]
torque_actual = torque.ravel()[:L]
speed_predicted = np.transpose(np.array([x[:, 0] for x in b])).ravel()[:L]
torque_predicted = np.transpose(np.array([x[:, 1] for x in b])).ravel()[:L]

time_series_plot = 5000

rmse = sqrt(mean_squared_error(speed_actual[:time_series_plot], speed_predicted[:time_series_plot]))
print("RMSE Error is " + str(rmse))

rmse = sqrt(mean_squared_error(torque_actual[:time_series_plot], torque_predicted[:time_series_plot]))
print("RMSE Error is " + str(rmse))

tt_given = np.linspace(0, 4, L)

time_series_plot = 5000
plt.subplot(3,1,1)
plt.plot(tt_given[:time_series_plot], speed_actual[:time_series_plot], 'r', label = 'Actual')
plt.plot(tt_given[:time_series_plot], speed_predicted[:time_series_plot], 'b', label = 'Predicted')
plt.legend(loc='upper right')
plt.xlabel('time(seconds)')
plt.ylabel('speed(LU/s)')
plt.title('Speed predicted vs actual')
plt.subplot(3,1,2)
plt.plot(tt_given[:time_series_plot], current_given[:time_series_plot], 'xr')
plt.xlabel('time(s)')
plt.ylabel('current(A)')
plt.title('Current given')
plt.subplot(3,1,3)
plt.plot(tt_given[:time_series_plot], speed_given[:time_series_plot], 'or')
plt.xlabel('time(seconds)')
plt.ylabel('speed(LU/s)')
plt.title('Speed given')
plt.tight_layout()

tt_given = np.linspace(0, 4, L)

time_series_plot = 5000
plt.subplot(3,1,1)
plt.plot(tt_given[:time_series_plot], torque_actual[:time_series_plot], 'r', label = 'Actual')
plt.plot(tt_given[:time_series_plot], torque_predicted[:time_series_plot], 'b', label = 'Predicted')
plt.legend(loc='upper right')
plt.xlabel('time(s)')
plt.ylabel('torque(Nm)')
plt.title('Torque predicted vs actual')
plt.subplot(3,1,2)
plt.plot(tt_given[:time_series_plot], current_given[:time_series_plot], 'xr')
plt.xlabel('time(s)')
plt.ylabel('current(A)')
plt.title('Current given')
plt.subplot(3,1,3)
plt.plot(tt_given[:time_series_plot], torque_given[:time_series_plot], 'or')
plt.xlabel('time(s)')
plt.ylabel('torque(Nm)')
plt.title('Torque given')
plt.tight_layout()

print("hello")