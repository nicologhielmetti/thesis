import sys
from functools import partial

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

X_test = np.load('models_and_data/quickdraw_dataset/X_test.npy', allow_pickle=True).astype(np.float32)
y_test = np.load('models_and_data/quickdraw_dataset/y_test.npy', allow_pickle=True).astype(np.float32)

ds_len = 1000

activation_analyzer = CustomFloPoAnalyzerKeras(model, model_name,
                                                     partial(Common.get_activations_keras, model,
                                                             X_test[:ds_len]),
                                                     'activations',
                                                     min_value_filter_ulp=0.10,
                                                     min_value_filter_exp=0.10,
                                                     ulp_percentiles=[40, 50, 60, 70, 80])

data_activations_fixed = activation_analyzer.analyze(analyze_ulp=True, analyze_exp=True, profile_timing=True)
activations_analysis = activation_analyzer.mantissa_exponent_analysis()
activation_analyzer.make_plots()

weight_analyzer = CustomFloPoAnalyzerKeras(model, model_name,
                                           partial(Common.get_weights_keras, model),
                                                 'weights',
                                           min_value_filter_ulp=0.10,
                                           min_value_filter_exp=0.10,
                                           ulp_percentiles=[40, 50, 60, 70, 80])

data_weights_fixed = weight_analyzer.analyze(analyze_ulp=True, analyze_exp=True, profile_timing=True)
weight_analysis_fixed = weight_analyzer.mantissa_exponent_analysis()
weight_analyzer.make_plots()
