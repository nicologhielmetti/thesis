import json
import sys
from datetime import datetime
from functools import partial
from json import JSONEncoder

import qkeras.utils
from tensorflow import keras


def _default(self, obj):
    return getattr(obj.__class__, "to_json", _default.default)(obj)


_default.default = JSONEncoder().default
JSONEncoder.default = _default

sys.path.extend(
    ['/data1/home/ghielmetti/thesis', '/data1/home/ghielmetti/thesis/PTQ_dev', '/data1/home/ghielmetti/thesis/QAT_dev',
     '/data1/home/ghielmetti/thesis/models_and_data'])

from common import Common
from custom_flopo_analyzer_keras import CustomFloPoAnalyzerKeras

import numpy as np
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from models_and_data import ModelsAndData
from quantized_float import quantized_float, quantized_float_tanh, quantized_float_sigmoid, quantized_float_softmax

X_train = np.load('../models_and_data/quickdraw_dataset/X_train.npy', allow_pickle=True).astype(np.float32)
y_train = np.load('../models_and_data/quickdraw_dataset/y_train.npy', allow_pickle=True).astype(np.float32)
X_test = np.load('../models_and_data/quickdraw_dataset/X_test.npy', allow_pickle=True).astype(np.float32)
y_test = np.load('../models_and_data/quickdraw_dataset/y_test.npy', allow_pickle=True).astype(np.float32)

# keras_model = keras.models_and_data.load_model('../models_and_data/saved_quickdraw_model/quickdraw_not_quantized.h5')
# ds_len = 2000
# activations_file_name = keras_model.name + '_' + str(ds_len)
# weights_file_name = keras_model.name
# 
# activation_analyzer = CustomFloPoAnalyzerKeras(keras_model, activations_file_name,
#                                                partial(Common.get_activations_keras, keras_model,
#                                                        X_test[:ds_len]),
#                                                'activations')
# 
# data_activations = activation_analyzer.analyze(profile_timing=True)
# activations_analysis = activation_analyzer.mantissa_exponent_analysis()
# activation_analyzer.make_plots()
# 
# weight_analyzer = CustomFloPoAnalyzerKeras(keras_model, weights_file_name,
#                                            partial(Common.get_weights_keras, keras_model),
#                                            'weights')
# 
# data_weights = weight_analyzer.analyze(profile_timing=True)
# weight_analysis = weight_analyzer.mantissa_exponent_analysis()
# weight_analyzer.make_plots()

# del data_weights
# del data_activations
# del activation_analyzer
# del weight_analyzer
# 

w_path = '../QAT_dev/analysis_report/weights_quickdraw_PTQ_analysis.json'
a_path = '../QAT_dev/analysis_report/activations_quickdraw_2000_PTQ_analysis.json'

with open(w_path, 'r') as f:
    weight_analysis = json.load(f)

with open(a_path, 'r') as f:
    activations_analysis = json.load(f)

act_dense_6_bias = activations_analysis['layer_data']['dense_6_activation']['statistical_values']['bias']
act_lstm_1_bias = activations_analysis['layer_data']['lstm_1_activation']['statistical_values']['bias']
state_lstm_1_bias = activations_analysis['layer_data']['lstm_1_state_activation']['statistical_values']['bias']
act_softmax_bias = activations_analysis['layer_data']['softmax_activation']['statistical_values']['bias']
act_dense_5_bias = activations_analysis['layer_data']['dense_5_activation']['statistical_values']['bias']
act_dense_3_bias = activations_analysis['layer_data']['dense_3_activation']['statistical_values']['bias']

act_dense_6_exp = activations_analysis['layer_data']['dense_6_activation']['statistical_values']['min_exp_bit']
act_lstm_1_exp = activations_analysis['layer_data']['lstm_1_activation']['statistical_values']['min_exp_bit']
state_lstm_1_exp = activations_analysis['layer_data']['lstm_1_state_activation']['statistical_values']['min_exp_bit']
act_softmax_exp = activations_analysis['layer_data']['softmax_activation']['statistical_values']['min_exp_bit']
act_dense_5_exp = activations_analysis['layer_data']['dense_5_activation']['statistical_values']['min_exp_bit']
act_dense_3_exp = activations_analysis['layer_data']['dense_3_activation']['statistical_values']['min_exp_bit']

act_dense_6_man = activations_analysis['layer_data']['dense_6_activation']['statistical_values']['min_man_bit'][4]
act_lstm_1_man = activations_analysis['layer_data']['lstm_1_activation']['statistical_values']['min_man_bit'][4]
state_lstm_1_man = activations_analysis['layer_data']['lstm_1_state_activation']['statistical_values']['min_man_bit'][4]
act_softmax_man = activations_analysis['layer_data']['softmax_activation']['statistical_values']['min_man_bit'][4]
act_dense_5_man = activations_analysis['layer_data']['dense_5_activation']['statistical_values']['min_man_bit'][4]
act_dense_3_man = activations_analysis['layer_data']['dense_3_activation']['statistical_values']['min_man_bit'][4]

wei_dense_6_bias = weight_analysis['layer_data']['dense_6_w']['statistical_values']['bias']
wei_lstm_1_bias = weight_analysis['layer_data']['lstm_1_w']['statistical_values']['bias']
rw_lstm_1_bias = weight_analysis['layer_data']['lstm_1_rw']['statistical_values']['bias']
wei_dense_5_bias = weight_analysis['layer_data']['dense_5_w']['statistical_values']['bias']
wei_dense_3_bias = weight_analysis['layer_data']['dense_3_w']['statistical_values']['bias']

wei_dense_6_exp = weight_analysis['layer_data']['dense_6_w']['statistical_values']['min_exp_bit']
wei_lstm_1_exp = weight_analysis['layer_data']['lstm_1_w']['statistical_values']['min_exp_bit']
rw_lstm_1_exp = weight_analysis['layer_data']['lstm_1_rw']['statistical_values']['min_exp_bit']
wei_dense_5_exp = weight_analysis['layer_data']['dense_5_w']['statistical_values']['min_exp_bit']
wei_dense_3_exp = weight_analysis['layer_data']['dense_3_w']['statistical_values']['min_exp_bit']

wei_dense_6_man = weight_analysis['layer_data']['dense_6_w']['statistical_values']['min_man_bit'][4]
wei_lstm_1_man = weight_analysis['layer_data']['lstm_1_w']['statistical_values']['min_man_bit'][4]
rw_lstm_1_man = weight_analysis['layer_data']['lstm_1_rw']['statistical_values']['min_man_bit'][4]
wei_dense_5_man = weight_analysis['layer_data']['dense_5_w']['statistical_values']['min_man_bit'][4]
wei_dense_3_man = weight_analysis['layer_data']['dense_3_w']['statistical_values']['min_man_bit'][4]

b_dense_6_bias = weight_analysis['layer_data']['dense_6_b']['statistical_values']['bias']
b_lstm_1_bias = weight_analysis['layer_data']['lstm_1_b']['statistical_values']['bias']
b_dense_5_bias = weight_analysis['layer_data']['dense_5_b']['statistical_values']['bias']
b_dense_3_bias = weight_analysis['layer_data']['dense_3_b']['statistical_values']['bias']

b_dense_6_exp = weight_analysis['layer_data']['dense_6_b']['statistical_values']['min_exp_bit']
b_lstm_1_exp = weight_analysis['layer_data']['lstm_1_b']['statistical_values']['min_exp_bit']
b_dense_5_exp = weight_analysis['layer_data']['dense_5_b']['statistical_values']['min_exp_bit']
b_dense_3_exp = weight_analysis['layer_data']['dense_3_b']['statistical_values']['min_exp_bit']

b_dense_6_man = weight_analysis['layer_data']['dense_6_b']['statistical_values']['min_man_bit'][4]
b_lstm_1_man = weight_analysis['layer_data']['lstm_1_b']['statistical_values']['min_man_bit'][4]
b_dense_5_man = weight_analysis['layer_data']['dense_5_b']['statistical_values']['min_man_bit'][4]
b_dense_3_man = weight_analysis['layer_data']['dense_3_b']['statistical_values']['min_man_bit'][4]

quantizer_dict = \
    {
        'quantized_input':
            {
                'activation_quantizer': quantized_float(4, 4)
            },
        'lstm_1':
            {
                'activation_quantizer': quantized_float_tanh(act_lstm_1_exp, act_lstm_1_man, act_lstm_1_bias,
                                                             use_exp_offset=1),
                'recurrent_activation': quantized_float_sigmoid(state_lstm_1_exp, state_lstm_1_man, state_lstm_1_bias,
                                                                use_exp_offset=1),
                'kernel_quantizer': quantized_float(wei_lstm_1_exp, wei_lstm_1_man, wei_lstm_1_bias, use_exp_offset=1),
                'recurrent_quantizer': quantized_float(rw_lstm_1_exp, rw_lstm_1_man, rw_lstm_1_bias, use_exp_offset=1),
                'bias_quantizer': quantized_float(b_lstm_1_exp, b_lstm_1_man, b_lstm_1_bias, use_exp_offset=1),
                'state_quantizer': quantized_float(state_lstm_1_exp, state_lstm_1_man, state_lstm_1_bias,
                                                   use_exp_offset=1)
            },
        'dense_3':
            {
                'activation_quantizer': quantized_float(act_dense_3_exp, act_dense_3_man, act_dense_3_bias,
                                                        use_exp_offset=1),
                'kernel_quantizer': quantized_float(wei_dense_3_exp, wei_dense_3_man, wei_dense_3_bias,
                                                    use_exp_offset=1),
                'bias_quantizer': quantized_float(b_dense_3_exp, b_dense_3_man, b_dense_3_bias, use_exp_offset=1)
            },
        'dense_5':
            {
                'activation_quantizer': quantized_float(act_dense_5_exp, act_dense_5_man, act_dense_5_bias,
                                                        use_exp_offset=1),
                'kernel_quantizer': quantized_float(wei_dense_5_exp, wei_dense_5_man, wei_dense_5_bias,
                                                    use_exp_offset=1),
                'bias_quantizer': quantized_float(b_dense_5_exp, b_dense_5_man, b_dense_5_bias, use_exp_offset=1)
            },
        'dense_6':
            {
                'activation_quantizer': quantized_float(act_dense_6_exp, act_dense_6_man, act_dense_6_bias,
                                                        use_exp_offset=1),
                'kernel_quantizer': quantized_float(wei_dense_6_exp, wei_dense_6_man, wei_dense_6_bias,
                                                    use_exp_offset=1),
                'bias_quantizer': quantized_float(b_dense_6_exp, b_dense_6_man, b_dense_6_bias, use_exp_offset=1)
            },
        'softmax':
            {
                'activation_quantizer': quantized_float_softmax(act_softmax_exp, act_softmax_man, act_softmax_bias,
                                                                use_exp_offset=1)
            }
    }
model_id = 'quickdraw_full_quantized_custom_stat'

with open(model_id + '_quantizer_dict.json', 'w') as json_file:
    json.dump(quantizer_dict, json_file, default=vars, indent=4)

keras_model = keras.models.load_model('../models_and_data/saved_quickdraw_model/Quickdraw5ClassLSTMFinL.h5')
co = \
    {
        'quantized_float': quantized_float,
        'quantized_float_softmax': quantized_float_softmax,
        'quantized_float_sigmoid': quantized_float_sigmoid,
        'quantized_float_tanh': quantized_float_tanh
    }

# quantized_model.load_weights('../models_and_data/saved_quickdraw_model/quickdraw_not_quantized.h5')
# quantized_weights = qkeras.utils.model_save_quantized_weights(
#     quantized_model,
#     '../models_and_data/saved_quickdraw_weights/' + model_id + '.h5')

quantized_model = qkeras.utils.model_quantize(keras_model, quantizer_dict, 0, co, transfer_weights=True)

# quickdraw = ModelsAndData.get_quickdraw()

# zw = quickdraw.get_weights()
# zwq = quickdraw_quantized.get_weights()
# s = [abs(a - b) for a, b in zip(zw, zwq)]
# quickdraw_quantized.load_weights('../models_and_data/saved_quickdraw_model/quickdraw_not_quantized.h5')

time_str = datetime.now().strftime("%Y%m%d_%H%M%S")

tb = TensorBoard(
    log_dir='tensorboard_logs/' + model_id + '_' + time_str,
    histogram_freq=1,
    write_graph=False,
    write_images=False,
    update_freq=1,
    profile_batch=0
)
quantized_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
test_perf = quantized_model.evaluate(x=X_test, y=y_test, verbose=1, workers=28 * 2, use_multiprocessing=True,
                                     return_dict=True, callbacks=[tb])
with open('../models_and_data/saved_quickdraw_weights/' + model_id + '_test_performance.json', 'w') as json_file:
    json.dump(test_perf, json_file)
