import sys
from functools import partial
sys.path.extend(
    ['/data1/home/ghielmetti/thesis', '/data1/home/ghielmetti/thesis/PTQ_dev', '/data1/home/ghielmetti/thesis/QAT_dev',
     '/data1/home/ghielmetti/thesis/models_and_data'])

import numpy as np
from tensorflow import keras

from shared_definitions import SharedDefinitons
from common import Common
from custom_flopo_analyzer_keras import CustomFloPoAnalyzerKeras

names = SharedDefinitons('quickdraw')
model_file_path, model_name = names.get_flopo32_model_names()

try:
    model = keras.models.load_model(model_file_path)
except:
    print('file: ' + model_file_path + ' does not exist')
    sys.exit(1)
    
X_test = np.load('../models_and_data/quickdraw_dataset/X_test.npy', allow_pickle=True).astype(np.float32)
y_test = np.load('../models_and_data/quickdraw_dataset/y_test.npy', allow_pickle=True).astype(np.float32)

ds_len = 1000

activations_file_name = model_name + '_' + str(ds_len)
weights_file_name = model_name

activation_analyzer_fixed = CustomFloPoAnalyzerKeras(model, activations_file_name,
                                                     partial(Common.get_activations_keras, model,
                                                             X_test[:ds_len]),
                                                     'activations',
                                                     min_value_filter_ulp=0.05,
                                                     min_value_filter_exp=0.05,
                                                     ulp_percentiles=[40, 50, 60, 70, 80])

data_activations_fixed = activation_analyzer_fixed.analyze(analyze_ulp=True, analyze_exp=True, profile_timing=True)
activations_analysis_fixed = activation_analyzer_fixed.mantissa_exponent_analysis()
activation_analyzer_fixed.make_plots()

weight_analyzer_fixed = CustomFloPoAnalyzerKeras(model, weights_file_name,
                                                 partial(Common.get_weights_keras, model),
                                                 'weights',
                                                 min_value_filter_ulp=0.05,
                                                 min_value_filter_exp=0.05,
                                                 ulp_percentiles=[40, 50, 60, 70, 80])

data_weights_fixed = weight_analyzer_fixed.analyze(analyze_ulp=True, analyze_exp=True, profile_timing=True)
weight_analysis_fixed = weight_analyzer_fixed.mantissa_exponent_analysis()
weight_analyzer_fixed.make_plots()
