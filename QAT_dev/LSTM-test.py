from keras import Model
from tensorflow import keras
from custom_flopo_analyzer_keras import CustomFloPoAnalyzerKeras
import numpy as np
from common import Common
from functools import partial
import tensorflow as tf

from models_and_data import ModelsAndData


# input_5 None
# lstm_4 <function tanh at 0x7ffb2cbf55e0>
# dropout_8 None
# dense_8 <function linear at 0x7ffb2cbf5ca0>
# dropout_9 None
# dense_9 <function linear at 0x7ffb2cbf5ca0>
# rnn_densef <function softmax at 0x7ffb2cbee820>

class LogIntermediateActivationsWeights(keras.callbacks.Callback):
    def on_predict_begin(self, logs=None):
        keys = list(logs.keys())
        print("Start predicting; got log keys: {}".format(keys))

    def on_predict_batch_end(self, batch, logs=None):
        keys = list(logs.keys())
        print("...Predicting: end of batch {}; got log keys: {}".format(batch, keys))


model = keras.models.load_model('../../models/saved-quickdraw-model/Quickdraw5ClassLSTMFinL.h5')
model._name = 'quickdraw_original'

model_fixed = Models.get_quickdraw()
model_fixed.load_weights('../../models/saved-quickdraw-model/Quickdraw5ClassLSTMFinL.h5')

X_test = np.load('../data/quickdraw-dataset/X_test_format.npy')
y_test = np.load('../data/quickdraw-dataset/y_test_format.npy')

# y_o = model.predict(X_test)
y_o = model.predict(x=X_test[:10], batch_size=1, callbacks=[LogIntermediateActivationsWeights()])
# y_f = model_fixed.predict(X_test)
# 
# m1 = Model(model.inputs, model.layers[1].output)
# m1_f = Model(model_fixed.inputs, model_fixed.layers[2].output)
# y1 = m1.predict(X_test)
# y1_f = m1_f.predict(X_test)
