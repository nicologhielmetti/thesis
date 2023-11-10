from collections import defaultdict

from hls4ml.model import profiling
import numpy as np

class Common:
    @staticmethod
    def get_activations_keras(model, X, datatype=np.float32):
        return profiling.activations_keras(model, X, type=datatype)

    @staticmethod
    def get_weights_keras(model):
        return profiling.weights_keras(model)

