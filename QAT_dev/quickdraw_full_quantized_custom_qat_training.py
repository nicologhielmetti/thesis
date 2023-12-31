import json
import sys
from datetime import datetime
from functools import partial
from json import JSONEncoder

from tensorflow import keras

sys.path.extend(
    ['/data1/home/ghielmetti/thesis', '/data1/home/ghielmetti/thesis/PTQ_dev', '/data1/home/ghielmetti/thesis/QAT_dev',
     '/data1/home/ghielmetti/thesis/models_and_data'])

from common import Common
from custom_flopo_analyzer_keras import CustomFloPoAnalyzerKeras

import numpy as np
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from models_and_data import ModelsAndData
from quantized_float import quantized_float, quantized_float_tanh, quantized_float_sigmoid, quantized_float_softmax

X_train = np.load('../models_and_data/quickdraw_dataset/X_train.npy', allow_pickle=True)
y_train = np.load('../models_and_data/quickdraw_dataset/y_train.npy', allow_pickle=True)
X_test = np.load('../models_and_data/quickdraw_dataset/X_test.npy', allow_pickle=True)
y_test = np.load('../models_and_data/quickdraw_dataset/y_test.npy', allow_pickle=True)

# class EpochCallbacks(keras.callbacks.Callback):
#     def __init__(self, epoch):
#         self.epoch = epoch
# 
#     def on_epoch_end(self, epoch, logs=None):
#         self.epoch.set_epochs(epoch)
#         # print("End epoch {} of training; got log keys: {}".format(epoch, keys))
#
# 
# class Epoch:
#     def __init__(self, counter):
#         self.counter = counter
# 
#     def set_epochs(self, new_counter):
#         self.counter = new_counter
# 
#     def get_epochs(self):
#         return self.counter


# epoch = Epoch(0)

keras_model = keras.models.load_model('../models_and_data/saved_quickdraw_model/quickdraw_not_quantized.h5')
ds_len = 10
activations_file_name = keras_model.name + '_' + str(ds_len)
weights_file_name = keras_model.name

activation_analyzer = CustomFloPoAnalyzerKeras(keras_model, activations_file_name,
                                               partial(Common.get_activations_keras, keras_model,
                                                       X_test[:ds_len]),
                                               'activations')

data_activations = activation_analyzer.analyze(profile_timing=True)
activations_analysis = activation_analyzer.mantissa_exponent_analysis()
# activation_analyzer.make_plots()

weight_analyzer = CustomFloPoAnalyzerKeras(keras_model, weights_file_name,
                                           partial(Common.get_weights_keras, keras_model),
                                           'weights')

data_weights = weight_analyzer.analyze(profile_timing=True)
weight_analysis = weight_analyzer.mantissa_exponent_analysis()
# weight_analyzer.make_plots()

del data_weights
del data_activations
del activation_analyzer
del weight_analyzer

act_dense_6_bias = activations_analysis['layer_data']['dense_6_activation']['exact_values']['bias']
act_lstm_1_bias = activations_analysis['layer_data']['lstm_1_activation']['exact_values']['bias']
state_lstm_1_bias = activations_analysis['layer_data']['lstm_1_state_activation']['exact_values']['bias']
act_softmax_bias = activations_analysis['layer_data']['softmax_activation']['exact_values']['bias']
act_dense_5_bias = activations_analysis['layer_data']['dense_5_activation']['exact_values']['bias']
act_dense_3_bias = activations_analysis['layer_data']['dense_3_activation']['exact_values']['bias']

act_dense_6_exp = activations_analysis['layer_data']['dense_6_activation']['exact_values']['min_exp_bit']
act_lstm_1_exp = activations_analysis['layer_data']['lstm_1_activation']['exact_values']['min_exp_bit']
state_lstm_1_exp = activations_analysis['layer_data']['lstm_1_state_activation']['exact_values']['min_exp_bit']
act_softmax_exp = activations_analysis['layer_data']['softmax_activation']['exact_values']['min_exp_bit']
act_dense_5_exp = activations_analysis['layer_data']['dense_5_activation']['exact_values']['min_exp_bit']
act_dense_3_exp = activations_analysis['layer_data']['dense_3_activation']['exact_values']['min_exp_bit']

act_dense_6_man = activations_analysis['layer_data']['dense_6_activation']['exact_values']['min_man_bit']
act_lstm_1_man = activations_analysis['layer_data']['lstm_1_activation']['exact_values']['min_man_bit']
state_lstm_1_man = activations_analysis['layer_data']['lstm_1_state_activation']['exact_values']['min_man_bit']
act_softmax_man = activations_analysis['layer_data']['softmax_activation']['exact_values']['min_man_bit']
act_dense_5_man = activations_analysis['layer_data']['dense_5_activation']['exact_values']['min_man_bit']
act_dense_3_man = activations_analysis['layer_data']['dense_3_activation']['exact_values']['min_man_bit']

wei_dense_6_bias = weight_analysis['layer_data']['dense_6_w']['exact_values']['bias']
wei_lstm_1_bias = weight_analysis['layer_data']['lstm_1_w']['exact_values']['bias']
rw_lstm_1_bias = weight_analysis['layer_data']['lstm_1_rw']['exact_values']['bias']
wei_dense_5_bias = weight_analysis['layer_data']['dense_5_w']['exact_values']['bias']
wei_dense_3_bias = weight_analysis['layer_data']['dense_3_w']['exact_values']['bias']

wei_dense_6_exp = weight_analysis['layer_data']['dense_6_w']['exact_values']['min_exp_bit']
wei_lstm_1_exp = weight_analysis['layer_data']['lstm_1_w']['exact_values']['min_exp_bit']
rw_lstm_1_exp = weight_analysis['layer_data']['lstm_1_rw']['exact_values']['min_exp_bit']
wei_dense_5_exp = weight_analysis['layer_data']['dense_5_w']['exact_values']['min_exp_bit']
wei_dense_3_exp = weight_analysis['layer_data']['dense_3_w']['exact_values']['min_exp_bit']

wei_dense_6_man = min(weight_analysis['layer_data']['dense_6_w']['exact_values']['min_man_bit'], 4)
wei_lstm_1_man = min(weight_analysis['layer_data']['lstm_1_w']['exact_values']['min_man_bit'], 4)
rw_lstm_1_man = min(weight_analysis['layer_data']['lstm_1_rw']['exact_values']['min_man_bit'], 4)
wei_dense_5_man = min(weight_analysis['layer_data']['dense_5_w']['exact_values']['min_man_bit'], 4)
wei_dense_3_man = min(weight_analysis['layer_data']['dense_3_w']['exact_values']['min_man_bit'], 4)

b_dense_6_bias = weight_analysis['layer_data']['dense_6_b']['exact_values']['bias']
b_lstm_1_bias = weight_analysis['layer_data']['lstm_1_b']['exact_values']['bias']
b_dense_5_bias = weight_analysis['layer_data']['dense_5_b']['exact_values']['bias']
b_dense_3_bias = weight_analysis['layer_data']['dense_3_b']['exact_values']['bias']

b_dense_6_exp = weight_analysis['layer_data']['dense_6_b']['exact_values']['min_exp_bit']
b_lstm_1_exp = weight_analysis['layer_data']['lstm_1_b']['exact_values']['min_exp_bit']
b_dense_5_exp = weight_analysis['layer_data']['dense_5_b']['exact_values']['min_exp_bit']
b_dense_3_exp = weight_analysis['layer_data']['dense_3_b']['exact_values']['min_exp_bit']

b_dense_6_man = min(weight_analysis['layer_data']['dense_6_b']['exact_values']['min_man_bit'], 4)
b_lstm_1_man = min(weight_analysis['layer_data']['lstm_1_b']['exact_values']['min_man_bit'], 4)
b_dense_5_man = min(weight_analysis['layer_data']['dense_5_b']['exact_values']['min_man_bit'], 4)
b_dense_3_man = min(weight_analysis['layer_data']['dense_3_b']['exact_values']['min_man_bit'], 4)


quantizer_dict = \
    {
        'quantized_input':
            {
                'activation': quantized_float(12, 0)
            },
        'lstm_1':
            {
                'activation': quantized_float_tanh(act_lstm_1_exp, act_lstm_1_man, act_lstm_1_bias, use_est_bias=1),
                'recurrent_activation': quantized_float_sigmoid(state_lstm_1_exp, state_lstm_1_man, state_lstm_1_bias,
                                                                use_est_bias=1),
                'kernel_quantizer': quantized_float(wei_lstm_1_exp, wei_lstm_1_man, wei_lstm_1_bias, use_est_bias=1),
                'recurrent_quantizer': quantized_float(rw_lstm_1_exp, rw_lstm_1_man, rw_lstm_1_bias, use_est_bias=1),
                'bias_quantizer': quantized_float(b_lstm_1_exp, b_lstm_1_man, b_lstm_1_bias, use_est_bias=1),
                'state_quantizer': quantized_float(state_lstm_1_exp, state_lstm_1_man, state_lstm_1_bias, use_est_bias=1)
            },
        'dense_3':
            {
                'activation': quantized_float(act_dense_3_exp, act_dense_3_man, act_dense_3_bias, use_est_bias=1),
                'kernel_quantizer': quantized_float(wei_dense_3_exp, wei_dense_3_man, wei_dense_3_bias, use_est_bias=1),
                'bias_quantizer': quantized_float(b_dense_3_exp, b_dense_3_man, b_dense_3_bias, use_est_bias=1)
            },
        'dense_5':
            {
                'activation': quantized_float(act_dense_5_exp, act_dense_5_man, act_dense_5_bias, use_est_bias=1),
                'kernel_quantizer': quantized_float(wei_dense_5_exp, wei_dense_5_man, wei_dense_5_bias, use_est_bias=1),
                'bias_quantizer': quantized_float(b_dense_5_exp, b_dense_5_man, b_dense_5_bias, use_est_bias=1)
            },
        'dense_6':
            {
                'activation': quantized_float(act_dense_6_exp, act_dense_6_man, act_dense_6_bias, use_est_bias=1),
                'kernel_quantizer': quantized_float(wei_dense_6_exp, wei_dense_6_man, wei_dense_6_bias, use_est_bias=1),
                'bias_quantizer': quantized_float(b_dense_6_exp, b_dense_6_man, b_dense_6_bias, use_est_bias=1)
            },
        'softmax':
            {
                'activation': quantized_float_softmax(act_softmax_exp, act_softmax_man, act_softmax_bias,
                                                      use_est_bias=1)
            }
    }

model_id = 'quickdraw_full_quantized_custom'

with open(model_id + '_quantizer_dict.json', 'w') as json_file:
    json.dump(quantizer_dict, json_file, default=vars, indent=4)

quickdraw_quantized = ModelsAndData.get_quickdraw_quantized_all_quantized(quantizer_dict=quantizer_dict)

time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
# ec = EpochCallbacks(epoch)
mc = ModelCheckpoint('ckpt_' + model_id + '_' + time_str + '/' + model_id + '_{epoch:02d}_{val_loss:.2f}.h5',
                     verbose=2, monitor='val_loss', mode='min', save_best_only=True)
es = EarlyStopping(monitor='val_loss', mode='min', verbose=2, patience=5)
tb = TensorBoard(
    log_dir='tensorboard_logs/' + model_id + '_' + time_str,
    histogram_freq=1,
    write_graph=False,
    write_images=False,
    update_freq=1,
    profile_batch=0
)
quickdraw_quantized.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = quickdraw_quantized.fit(X_train, y_train, batch_size=512, epochs=50,
                                  validation_split=0.2, shuffle=True, callbacks=[mc, es, tb],
                                  use_multiprocessing=True, workers=28 * 2)

test_perf = quickdraw_quantized.evaluate(x=X_test, y=y_test, verbose=1, workers=28 * 2, use_multiprocessing=True,
                                         return_dict=True, callbacks=[mc, es, tb])
with open('ckpt_' + model_id + '_' + time_str + '/' + model_id + '_test_performance.json', 'w') as json_file:
    json.dump(test_perf, json_file)
