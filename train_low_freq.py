import scinet_motor.model as nn
import scinet_motor.ed_motor as edm

series_length = 50
N = 20

data, speed, torque = edm.create_motor_dataset_low_freq(series_length, N, file_name='motor_low_freq')

net = nn.Network(latent_size=2, input_size=1, encoder_num_units=[100, 100], decoder_num_units=[100, 100], name='motor_low_freq',
                 output_size=2, time_series_length=series_length)

training_data, validation_data = edm.load(20, 'motor_low_freq')

net.train(epoch_num=200, batch_size=1, learning_rate=1e-3, training_data=training_data, validation_data=validation_data,
          test_step=50)

net.save('motor_low_freq')


