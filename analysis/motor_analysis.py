import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scinet_motor import model as nn
import scinet_motor.ed_motor as edm


series_length = 50
N = 20

net = nn.Network.from_saved('motor', change_params={'name': 'motor'})

data, speed, torque = edm.motor_data(series_length, N)

b = net.run(data, net.decoded_list)

rms_loss = np.sqrt(net.run(data, net.recon_loss))
print('Loss: ' + str(rms_loss))

speed_Predicted = []
torque_Predicted = []
for i in range(len(b)):
    speed_Predicted.append(b[i][:, 0])
    torque_Predicted.append(b[i][:, 1])
speed_Predicted = np.transpose(np.array(speed_Predicted))
torque_Predicted = np.transpose(np.array(torque_Predicted))

c = torque[0]
d = speed[0]

for i in range(5):
    blue_color='#000cff'
    orange_color='#ff7700'
    print('speed given:' + str(speed[i][0]))
    print('speed predicted:' + str(speed_Predicted[i]))
    print('speed actual:' + str(speed[i]))
    print('torque given:' + str(torque[i][0]))
    print('torque predicted:' + str(torque_Predicted[i]))
    print('torque actual:' + str(torque[i]))
    tt_given = np.linspace(0, 1, series_length)
    fig = plt.figure(figsize=(3.4, 2.1))
    ax = fig.add_subplot(111)
    ax.plot(tt_given, speed[i], color=orange_color, label='True time evolution')
    ax.plot(tt_given, speed_Predicted[i], '--', color=blue_color, label='Predicted time evolution')
    ax.set_xlabel(r'$t$ [$s$]')
    ax.set_ylabel(r'$x$ [$m$]')
    handles, labels = ax.get_legend_handles_labels()
    lgd=ax.legend(handles, labels,loc='upper center', bbox_to_anchor=(0.6, 1.3), shadow=True, ncol=1)
    fig.tight_layout()
    plt.show()

