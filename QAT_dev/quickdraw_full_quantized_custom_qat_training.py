import json
import sys
from datetime import datetime
from functools import partial

from tensorflow import keras

sys.path.extend(
    ['/data1/home/ghielmetti/thesis', '/data1/home/ghielmetti/thesis/PTQ_dev', '/data1/home/ghielmetti/thesis/QAT_dev',
     '/data1/home/ghielmetti/thesis/models'])

from common import Common
from custom_flopo_analyzer_keras import CustomFloPoAnalyzerKeras

import numpy as np
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from models_and_data import ModelsAndData
from quantized_float_tf import quantized_float, quantized_float_tanh, quantized_float_sigmoid, quantized_float_softmax

X_train = np.load('../models/quickdraw_dataset/X_train.npy', allow_pickle=True)
y_train = np.load('../models/quickdraw_dataset/y_train.npy', allow_pickle=True)
X_test = np.load('../models/quickdraw_dataset/X_test.npy', allow_pickle=True)
y_test = np.load('../models/quickdraw_dataset/y_test.npy', allow_pickle=True)

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

keras_model = keras.models.load_model('../models/saved_quickdraw_model/quickdraw_not_quantized.h5')
ds_len = 1000
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

act_dense_6_bias = activations_analysis['layer_data']['dense_6']['exact_values']['exact_bias']
act_lstm_1_bias = activations_analysis['layer_data']['lstm_1']['exact_values']['exact_bias']
act_softmax_bias = activations_analysis['layer_data']['softmax']['exact_values']['exact_bias']
act_dense_5_bias = activations_analysis['layer_data']['dense_5']['exact_values']['exact_bias']
act_dense_3_bias = activations_analysis['layer_data']['dense_3']['exact_values']['exact_bias']

act_dense_6_exp = activations_analysis['layer_data']['dense_6']['exact_values']['min_exponent_bits']
act_lstm_1_exp = activations_analysis['layer_data']['lstm_1']['exact_values']['min_exponent_bits']
act_softmax_exp = activations_analysis['layer_data']['softmax']['exact_values']['min_exponent_bits']
act_dense_5_exp = activations_analysis['layer_data']['dense_5']['exact_values']['min_exponent_bits']
act_dense_3_exp = activations_analysis['layer_data']['dense_3']['exact_values']['min_exponent_bits']

act_dense_6_man = min(activations_analysis['layer_data']['dense_6']['exact_values']['min_mantissa_bit'], 4)
act_lstm_1_man = min(activations_analysis['layer_data']['lstm_1']['exact_values']['min_mantissa_bit'], 4)
act_softmax_man = min(activations_analysis['layer_data']['softmax']['exact_values']['min_mantissa_bit'], 4)
act_dense_5_man = min(activations_analysis['layer_data']['dense_5']['exact_values']['min_mantissa_bit'], 4)
act_dense_3_man = min(activations_analysis['layer_data']['dense_3']['exact_values']['min_mantissa_bit'], 4)

wei_dense_6_bias = weight_analysis['layer_data']['dense_6']['exact_values']['exact_bias']
wei_lstm_1_bias = weight_analysis['layer_data']['lstm_1']['exact_values']['exact_bias']
wei_dense_5_bias = weight_analysis['layer_data']['dense_5']['exact_values']['exact_bias']
wei_dense_3_bias = weight_analysis['layer_data']['dense_3']['exact_values']['exact_bias']

wei_dense_6_exp = weight_analysis['layer_data']['dense_6']['exact_values']['min_exponent_bits']
wei_lstm_1_exp = weight_analysis['layer_data']['lstm_1']['exact_values']['min_exponent_bits']
wei_dense_5_exp = weight_analysis['layer_data']['dense_5']['exact_values']['min_exponent_bits']
wei_dense_3_exp = weight_analysis['layer_data']['dense_3']['exact_values']['min_exponent_bits']

wei_dense_6_man = min(weight_analysis['layer_data']['dense_6']['exact_values']['min_mantissa_bit'], 4)
wei_lstm_1_man = min(weight_analysis['layer_data']['lstm_1']['exact_values']['min_mantissa_bit'], 4)
wei_dense_5_man = min(weight_analysis['layer_data']['dense_5']['exact_values']['min_mantissa_bit'], 4)
wei_dense_3_man = min(weight_analysis['layer_data']['dense_3']['exact_values']['min_mantissa_bit'], 4)


quantizer_dict = \
    {
        'quantized_input':
            {
                'activation': quantized_float(4, 4)
            },
        'lstm_1':
            {
                'activation': quantized_float_tanh(act_lstm_1_exp, act_lstm_1_man, act_lstm_1_bias, use_est_bias=1),
                'recurrent_activation': quantized_float_sigmoid(wei_lstm_1_exp, wei_lstm_1_man, wei_lstm_1_bias, use_est_bias=1),
                'kernel_quantizer': quantized_float(wei_lstm_1_exp, wei_lstm_1_man, wei_lstm_1_bias, use_est_bias=1),
                'recurrent_quantizer': quantized_float(wei_lstm_1_exp, wei_lstm_1_man, wei_lstm_1_bias, use_est_bias=1),
                'bias_quantizer': quantized_float(wei_lstm_1_exp, wei_lstm_1_man, wei_lstm_1_bias, use_est_bias=1),
                'state_quantizer': quantized_float(wei_lstm_1_exp, wei_lstm_1_man, wei_lstm_1_bias, use_est_bias=1)
            },
        'dense_3':
            {
                'activation': quantized_float(act_dense_3_exp, act_dense_3_man, act_dense_3_bias, use_est_bias=1),
                'kernel_quantizer': quantized_float(wei_dense_3_exp, wei_dense_3_man, wei_dense_3_bias, use_est_bias=1),
                'bias_quantizer': quantized_float(wei_dense_3_exp, wei_dense_3_man, wei_dense_3_bias, use_est_bias=1)
            },
        'dense_5':
            {
                'activation': quantized_float(act_dense_5_exp, act_dense_5_man, act_dense_5_bias, use_est_bias=1),
                'kernel_quantizer': quantized_float(wei_dense_5_exp, wei_dense_5_man, wei_dense_5_bias, use_est_bias=1),
                'bias_quantizer': quantized_float(wei_dense_5_exp, wei_dense_5_man, wei_dense_5_bias, use_est_bias=1)
            },
        'dense_6':
            {
                'activation': quantized_float(act_dense_6_exp, act_dense_6_man, act_dense_6_bias, use_est_bias=1),
                'kernel_quantizer': quantized_float(wei_dense_6_exp, wei_dense_6_man, wei_dense_6_bias, use_est_bias=1),
                'bias_quantizer': quantized_float(wei_dense_6_exp, wei_dense_6_man, wei_dense_6_bias, use_est_bias=1)
            },
        'softmax':
            {
                'activation': quantized_float_softmax(act_softmax_exp, act_softmax_man, act_softmax_bias, use_est_bias=1)
            }
    }
quickdraw_quantized = ModelsAndData.get_quickdraw_quantized_all_quantized(quantizer_dict=quantizer_dict)

model_id = 'quickdraw_full_quantized_custom'
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
                                  use_multiprocessing=True, workers=28*2)

test_perf = quickdraw_quantized.evaluate(x=X_test, y=y_test, verbose=1, workers=28*2, use_multiprocessing=True,
                                         return_dict=True, callbacks=[mc, es, tb])
with open('ckpt_' + model_id + '_' + time_str + '/' + model_id + '_test_performance.json', 'w') as json_file:
    json.dump(test_perf, json_file)
