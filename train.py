# import scinet_motor.model as nn
# import scinet_motor.ed_motor as edm
from scinet_motor import model_new as nn
from scinet_motor import ed_motor as edm
import scinet_motor.io_nn as io_nn
import os

series_length = 50
N = 20

model_id = "latent4_100"
input_output = io_nn.Io_nn(model_id, path=os.path.realpath(__file__))

# data, current, speed, torque = edm.create_motor_dataset(series_length, N, file_name='motor_speed_torque_50')

net = nn.Network(latent_size=4, input_size=1, encoder_num_units=[100, 100], decoder_num_units=[100, 100], name='motor_speed_torque_50',
                 output_size=2, time_series_length=series_length, io_nn=input_output)

training_data, validation_data = edm.load_speed_torque_current(20, 'motor_speed_torque_50')

net.train(epoch_num=2000, batch_size=1000, learning_rate=1e-4, training_data=training_data, validation_data=validation_data,
          test_step=10)

net.save('motor_speed_torque_50')