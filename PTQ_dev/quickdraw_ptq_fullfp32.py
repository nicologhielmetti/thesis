import json
import sys
from datetime import datetime
from functools import partial

import qkeras.utils
from tensorflow import keras

from json import JSONEncoder

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

X_train = np.load('../models_and_data/quickdraw_dataset/X_train.npy', allow_pickle=True)
y_train = np.load('../models_and_data/quickdraw_dataset/y_train.npy', allow_pickle=True)
X_test = np.load('../models_and_data/quickdraw_dataset/X_test.npy', allow_pickle=True)
y_test = np.load('../models_and_data/quickdraw_dataset/y_test.npy', allow_pickle=True)

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

quantizer_dict = \
    {
        'quantized_input':
            {
                'activation_quantizer': quantized_float(8, 23)
            },
        'lstm_1':
            {
                'activation_quantizer': quantized_float_tanh(8, 23),
                'recurrent_activation_quantizer': quantized_float_sigmoid(8, 23),
                'kernel_quantizer': quantized_float(8, 23),
                'recurrent_quantizer': quantized_float(8, 23),
                'bias_quantizer': quantized_float(8, 23),
                'state_quantizer': quantized_float(8, 23)
            },
        'dense_3':
            {
                'activation_quantizer': quantized_float(8, 23),
                'kernel_quantizer': quantized_float(8, 23),
                'bias_quantizer': quantized_float(8, 23)
            },
        'dense_5':
            {
                'activation_quantizer': quantized_float(8, 23),
                'kernel_quantizer': quantized_float(8, 23),
                'bias_quantizer': quantized_float(8, 23)
            },
        'dense_6':
            {
                'activation_quantizer': quantized_float(8, 23),
                'kernel_quantizer': quantized_float(8, 23),
                'bias_quantizer': quantized_float(8, 23)
            },
        'softmax':
            {
                'activation_quantizer': quantized_float_softmax(8, 23)
            }
    }
model_id = 'quickdraw_full_quantized_full_precision'

with open(model_id + '_quantizer_dict.json', 'w') as json_file:
    json.dump(quantizer_dict, json_file, default=vars, indent=4)

keras_model = keras.models.load_model('../models_and_data/saved_quickdraw_model/quickdraw_not_quantized.h5')
co = \
    {
        'quantized_float': quantized_float,
        'quantized_float_softmax': quantized_float_softmax,
        'quantized_float_sigmoid': quantized_float_sigmoid,
        'quantized_float_tanh': quantized_float_tanh
    }

qmodel = qkeras.utils.model_quantize(keras_model, quantizer_dict, 0, co, True)

# quantized_model = ModelsAndData.get_quickdraw_quantized_all_quantized(quantizer_dict=quantizer_dict)
# quantized_model.load_weights('../models_and_data/saved_quickdraw_model/quickdraw_not_quantized.h5')
quantized_weights = qkeras.utils.model_save_quantized_weights(
    qmodel,
    '../models_and_data/saved_quickdraw_weights/' + model_id + '_q.h5')
qmodel.load_weights('../models_and_data/saved_quickdraw_weights/' + model_id + '_q.h5')

# quantized_model = qkeras.utils.model_quantize(keras_model, quantizer_dict, 0, co)

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
qmodel.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
test_perf = qmodel.evaluate(x=X_test, y=y_test, verbose=1, workers=28 * 2, use_multiprocessing=True,
                                     return_dict=True, callbacks=[tb])
with open('../models_and_data/saved_quickdraw_weights/' + model_id + '_test_performance.json', 'w') as json_file:
    json.dump(test_perf, json_file)
