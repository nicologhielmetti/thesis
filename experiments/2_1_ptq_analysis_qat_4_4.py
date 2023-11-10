import sys
from functools import partial
from json import JSONEncoder

from models_and_data import ModelsAndData


def _default(self, obj):
    return getattr(obj.__class__, "to_json", _default.default)(obj)


_default.default = JSONEncoder().default
JSONEncoder.default = _default
import keras.models
import qkeras.utils

import numpy as np

from shared_definitions import SharedDefinitons
from common import Common
from custom_flopo_analyzer_keras import CustomFloPoAnalyzerKeras

names = SharedDefinitons('quickdraw')
model_file_path, model_name = names.get_homo_quantized_model_names(4, 4, 0)

model = ModelsAndData.get_quickdraw()

qmodel = qkeras.utils.model_quantize(model, names.get_config_4_4(), 0, names.get_custom_objects(),
                                              True)

# except Exception as e:
#     print('file: ' + model_file_path + ' does not exist')
#     sys.exit(1)

X_test = np.load('models_and_data/quickdraw_dataset/X_test.npy', allow_pickle=True).astype(np.float32)
y_test = np.load('models_and_data/quickdraw_dataset/y_test.npy', allow_pickle=True).astype(np.float32)

ds_len = 1000
#
# activation_analyzer = CustomFloPoAnalyzerKeras(qmodel, model_name,
#                                                partial(Common.get_activations_keras, qmodel,
#                                                        X_test[:ds_len]),
#                                                'activations',
#                                                min_value_filter_ulp=0.10,
#                                                min_value_filter_exp=0.10,
#                                                ulp_percentiles=[40, 50, 60, 70, 80])
#
# data_activations = activation_analyzer.analyze(analyze_ulp=True, analyze_exp=True, profile_timing=True)
# activations_analysis_fixed = activation_analyzer.mantissa_exponent_analysis()
# activation_analyzer.make_plots()

weight_analyzer = CustomFloPoAnalyzerKeras(qmodel, model_name,
                                           partial(Common.get_weights_keras, qmodel),
                                           'weights',
                                           min_value_filter_ulp=0.10,
                                           min_value_filter_exp=0.10,
                                           ulp_percentiles=[40, 50, 60, 70, 80])

data_weights_fixed = weight_analyzer.analyze(analyze_ulp=True, analyze_exp=True, profile_timing=True)
weight_analysis_fixed = weight_analyzer.mantissa_exponent_analysis()
weight_analyzer.make_plots()
