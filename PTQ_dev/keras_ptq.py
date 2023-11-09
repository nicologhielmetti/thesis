import sys
sys.path.extend(
    ['/data1/home/ghielmetti/thesis', '/data1/home/ghielmetti/thesis/PTQ_dev', '/data1/home/ghielmetti/thesis/QAT_dev',
     '/data1/home/ghielmetti/thesis/models_and_data', '/home/nicolo/thesis/QAT_dev', '/home/nicolo/thesis/models_and_data',
     '/home/nicolo/thesis/PTQ_dev'])
from tensorflow import keras
from custom_flopo_analyzer_keras import CustomFloPoAnalyzerKeras
import numpy as np
from common import Common
from functools import partial
from models_and_data import ModelsAndData

model = keras.models.load_model('../../models_and_data/saved_quickdraw_model/Quickdraw5ClassLSTMFinL.h5')
model._name = 'quickdraw_original'
X_test = np.load('../data/quickdraw_dataset/X_test_format.npy')
y_test = np.load('../data/quickdraw_dataset/y_test_format.npy')

ds_len = 1000
activations_file_name = model.name + '-' + str(ds_len)

weights_file_name = model.name

# activation_analyzer = CustomFloPoAnalyzerKeras(qmodel, activations_file_name,
#                                                partial(Common.get_activations_keras, qmodel, X_test[:ds_len]),
#                                                'activations')
# 
# data_activations = activation_analyzer.analyze(profile_timing=True)
# activations_analysis = activation_analyzer.mantissa_exponent_analysis()
# activation_analyzer.make_plots()
# 
# weight_analyzer = CustomFloPoAnalyzerKeras(qmodel, weights_file_name, partial(Common.get_weights_keras, qmodel),
#                                            'weights')
# 
# data_weights = weight_analyzer.analyze(profile_timing=True)
# weight_analysis = weight_analyzer.mantissa_exponent_analysis()
# weight_analyzer.make_plots()

######################################

model_fixed = Models.get_quickdraw()
model_fixed.load_weights('../../models_and_data/saved_quickdraw_model/Quickdraw5ClassLSTMFinL.h5')
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
