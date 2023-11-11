import pandas
from hls4ml.model import profiling
import numpy as np
from tensorflow import keras


class Common:
    @staticmethod
    def get_activations_keras(model, X, datatype=np.float32):
        def get_activations(model, X_test):
            inp = model.input
            res = {'values': [], 'layer_name': [], 'activation_name': []}
            for l in model.layers:
                if not isinstance(l, keras.layers.InputLayer) and not isinstance(l, keras.layers.Dropout):
                    i_model = keras.models.Model(inputs=[inp], outputs=[l.output])
                    y = i_model.predict(X_test).flatten()
                    res['values'].extend(y)
                    res['layer_name'].extend([l.name] * len(y))
                    res['activation_name'].extend(['activation'] * len(y))
            df = pandas.DataFrame(res)
            return df

        df = get_activations(model, X.astype(datatype))
        return df

        # return profiling.activations_keras(model, X, type=datatype)

    @staticmethod
    def get_weights_keras(model):
        return profiling.weights_keras(model)
