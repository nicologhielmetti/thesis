from tensorflow import keras
from custom_flopo_analyzer_keras import CustomFloPoAnalyzerKeras
import numpy as np
from common import Common
from functools import partial

from models_and_data.models import Models

# input_5 None
# lstm_4 <function tanh at 0x7ffb2cbf55e0>
# dropout_8 None
# dense_8 <function linear at 0x7ffb2cbf5ca0>
# dropout_9 None
# dense_9 <function linear at 0x7ffb2cbf5ca0>
# rnn_densef <function softmax at 0x7ffb2cbee820>

model = keras.models.load_model('../../models/saved_quickdraw_model/Quickdraw5ClassLSTMFinL.h5')
model._name = 'quickdraw_original'
X_test = np.load('../data/quickdraw-dataset/X_test_format.npy')
y_test = np.load('../data/quickdraw-dataset/y_test_format.npy')

ds_len = 1000
activations_file_name = model.name + '-' + str(ds_len)

weights_file_name = model.name

# activation_analyzer = CustomFloPoAnalyzerKeras(model, activations_file_name,
#                                                partial(Common.get_activations_keras, model, X_test[:ds_len]),
#                                                'activations')
# 
# data_activations = activation_analyzer.analyze(profile_timing=True)
# activations_analysis = activation_analyzer.mantissa_exponent_analysis()
# activation_analyzer.make_plots()
# 
# weight_analyzer = CustomFloPoAnalyzerKeras(model, weights_file_name, partial(Common.get_weights_keras, model),
#                                            'weights')
# 
# data_weights = weight_analyzer.analyze(profile_timing=True)
# weight_analysis = weight_analyzer.mantissa_exponent_analysis()
# weight_analyzer.make_plots()

######################################

model_fixed = Models.get_quickdraw()
model_fixed.load_weights('../../models/saved_quickdraw_model/Quickdraw5ClassLSTMFinL.h5')
model_fixed.predict()

activations_file_name_fix = model_fixed.name + '-' + str(ds_len)

weights_file_name_fix = model_fixed.name

activation_analyzer_fixed = CustomFloPoAnalyzerKeras(model_fixed, activations_file_name_fix,
                                                     partial(Common.get_activations_keras, model_fixed,
                                                             X_test[:ds_len]),
                                                     'activations')

data_activations_fixed = activation_analyzer_fixed.analyze(profile_timing=True)
activations_analysis_fixed = activation_analyzer_fixed.mantissa_exponent_analysis()
activation_analyzer_fixed.make_plots()

weight_analyzer_fixed = CustomFloPoAnalyzerKeras(model_fixed, weights_file_name_fix,
                                                 partial(Common.get_weights_keras, model_fixed),
                                                 'weights')

data_weights_fixed = weight_analyzer_fixed.analyze(profile_timing=True)
weight_analysis_fixed = weight_analyzer_fixed.mantissa_exponent_analysis()
weight_analyzer_fixed.make_plots()
