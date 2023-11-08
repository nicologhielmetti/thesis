import json
import sys
from datetime import datetime
from json import JSONEncoder
def _default(self, obj):
    return getattr(obj.__class__, "to_json", _default.default)(obj)


_default.default = JSONEncoder().default
JSONEncoder.default = _default
sys.path.extend(
    ['/data1/home/ghielmetti/thesis', '/data1/home/ghielmetti/thesis/PTQ_dev', '/data1/home/ghielmetti/thesis/QAT_dev',
     '/data1/home/ghielmetti/thesis/models_and_data'])

import numpy as np
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from models_and_data import ModelsAndData
from quantized_float import quantized_float, quantized_float_tanh, quantized_float_sigmoid, quantized_float_softmax

X_train = np.load('models_and_data/quickdraw_dataset/X_train.npy', allow_pickle=True).astype(np.float32)
y_train = np.load('models_and_data/quickdraw_dataset/y_train.npy', allow_pickle=True).astype(np.float32)
X_test = np.load('models_and_data/quickdraw_dataset/X_test.npy', allow_pickle=True).astype(np.float32)
y_test = np.load('models_and_data/quickdraw_dataset/y_test.npy', allow_pickle=True).astype(np.float32)

# class EpochCallbacks(keras.callbacks.Callback):
#     def __init__(self, epoch):
#         self.epoch = epoch
# 
#     def on_epoch_end(self, epoch, logs=None):
#         self.epoch.set_epochs(epoch)
#         # print("End epoch {} of training; got log keys: {}".format(epoch, keys))

quantizer_dict = \
    {
        'input_linear':
            {
                'activation_quantizer': quantized_float(4, 4)
            },
        'lstm_1':
            {
                'activation_quantizer': quantized_float_tanh(4, 4),
                'recurrent_activation_quantizer': quantized_float_sigmoid(4, 4),
                'kernel_quantizer': quantized_float(4, 4),
                'recurrent_quantizer': quantized_float(4, 4),
                'bias_quantizer': quantized_float(4, 4),
                'state_quantizer': quantized_float(4, 4)
            },
        'dense_3':
            {
                'activation_quantizer': quantized_float(4, 4),
                'kernel_quantizer': quantized_float(4, 4),
                'bias_quantizer': quantized_float(4, 4)
            },
        'dense_5':
            {
                'activation_quantizer': quantized_float(4, 4),
                'kernel_quantizer': quantized_float(4, 4),
                'bias_quantizer': quantized_float(4, 4)
            },
        'dense_6':
            {
                'activation_quantizer': quantized_float(4, 4),
                'kernel_quantizer': quantized_float(4, 4),
                'bias_quantizer': quantized_float(4, 4)
            },
        'softmax':
            {
                'activation_quantizer': quantized_float_softmax(4, 4)
            }
    }
quickdraw_quantized = ModelsAndData.get_quickdraw_quantized(quantizer_dict=quantizer_dict)


model_id = 'quickdraw_full_quantized_4_4'

with open(model_id + '_quantizer_dict.json', 'w') as json_file:
    json.dump(quantizer_dict, json_file, indent=4)

time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
# ec = EpochCallbacks(epoch)
mc = ModelCheckpoint('ckpt_' + model_id + '_' + time_str + '/' + model_id + '_{epoch:02d}_{val_loss:.2f}.h5',
                     verbose=2, monitor='val_loss', mode='min', save_best_only=True)
es = EarlyStopping(monitor='val_loss', mode='min', verbose=2, patience=5)
tb = TensorBoard(
    log_dir='tensorboard_logs/' + model_id + '_' + time_str,
    histogram_freq=1,
    write_graph=False,
    write_images=False,
    update_freq=1,
    profile_batch=0
)
quickdraw_quantized.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = quickdraw_quantized.fit(X_train, y_train, batch_size=512, epochs=50,
                                  validation_split=0.2, shuffle=True, callbacks=[mc, es, tb],
                                  use_multiprocessing=True, workers=28)

test_perf = quickdraw_quantized.evaluate(x=X_test, y=y_test, verbose=1, workers=28, use_multiprocessing=True,
                                         return_dict=True, callbacks=[mc, es, tb])
with open('ckpt_' + model_id + '_' + time_str + '/' + model_id + '_test_performance.json', 'w') as json_file:
    json.dump(test_perf, json_file)
