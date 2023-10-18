from tensorflow import keras
from custom_flopo_analyzer_keras import CustomFloPoAnalyzerKeras
import numpy as np
from common import Common
from functools import partial

# input_5 None
# lstm_4 <function tanh at 0x7ffb2cbf55e0>
# dropout_8 None
# dense_8 <function linear at 0x7ffb2cbf5ca0>
# dropout_9 None
# dense_9 <function linear at 0x7ffb2cbf5ca0>
# rnn_densef <function softmax at 0x7ffb2cbee820>

model = keras.models.load_model('../../models/quickdraw-model/Quickdraw5ClassLSTMFinL.h5')

X_test = np.load('../../data/quickdraw-dataset/X_test_format.npy')
y_test = np.load('../../data/quickdraw-dataset/y_test_format.npy')

ds_len = 1000
ulp_activations_file_name = '../profiling_data/ulp-activations-quickdraw-' + str(ds_len) + '.pkl'
exp_activations_file_name = '../profiling_data/exp-activations-quickdraw-' + str(ds_len) + '.pkl'

ulp_weights_file_name = '../profiling_data/ulp-weights-quickdraw.pkl'
exp_weights_file_name = '../profiling_data/exp-weights-quickdraw.pkl'


activation_analyzer = CustomFloPoAnalyzerKeras(
    model, ulp_activations_file_name,
    exp_activations_file_name,
    partial(Common.get_activations_keras, model, X_test[:ds_len]), 'activations'
)

data_activations = activation_analyzer.analyze(profile_timing=True)
man_exp_activations = activation_analyzer.mantissa_exponent_analysis()
# activation_analyzer.plot_value_frequency()

weight_analyzer = CustomFloPoAnalyzerKeras(
    model, ulp_weights_file_name,
    exp_weights_file_name,
    partial(Common.get_weights_keras, model), 'weights'
)

data_weights = weight_analyzer.analyze(profile_timing=True)
man_exp_weights = weight_analyzer.mantissa_exponent_analysis()
# weight_analyzer.plot_value_frequency()
