import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scinet_motor import model_test as nn
import scinet_motor.ed_motor as edm
import tqdm
import scinet_motor.io_nn as io
series_length = 50
N = 20

net = nn.Network.from_saved('motor_speed', change_params={'load_file': 'model_motor_speed_epoch_155'})

# data, speed, torque = edm.create_motor_dataset(series_length, N)

data, speed = edm.create_motor_dataset_speed(series_length, N)

input = data[0, 0]

input = np.reshape(input, [1, 1])

input_step = np.zeros((1280), dtype=np.float32)
speed = np.zeros((64000), dtype=np.float32)
for step in tqdm.tqdm(range(1280)):
    out = net.run(np.reshape(data[step, 0], [1, 1]), net.decoded_list)
    speed_step = np.transpose(np.reshape(np.array(out), (50)))
    speed[step * 50: (step * 50) + 50] = speed_step
    input_step[step] = data[step, 0]

speed_actual = np.zeros((64000), dtype=np.float32)
for step in tqdm.tqdm(range(1280)):
    speed_actual[step * 50: (step * 50) + 50] = data[step]

blue_color='#000cff'
orange_color='#ff7700'
green = "#00ff00"
tt_given = np.linspace(0, 1, 64000)
fig = plt.figure(figsize=(3.4, 2.1))
ax = fig.add_subplot(111)
ax.plot(tt_given, speed_actual, color=orange_color, label='True time evolution')
ax.plot(tt_given, speed, '--', color=blue_color, label='Predicted time evolution')
#ax.plot(tt_input, speed_step, '--', color=green, label='Input time evolution')
ax.set_xlabel(r'$t$ [$s$]')
ax.set_ylabel(r'$x$ [$m$]')
handles, labels = ax.get_legend_handles_labels()
lgd=ax.legend(handles, labels,loc='upper center', bbox_to_anchor=(0.6, 1.3), shadow=True, ncol=1)
fig.tight_layout()
plt.show()
print("hello")
# plt.savefig(io.tf_save_path + 'Plot.png', dpi=300)