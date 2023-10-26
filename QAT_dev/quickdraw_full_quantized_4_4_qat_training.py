import sys
sys.path.extend(['/data1/home/ghielmetti/thesis', '/data1/home/ghielmetti/thesis/PTQ-dev', '/data1/home/ghielmetti/thesis/QAT_dev', '/data1/home/ghielmetti/thesis/models'])

import numpy as np
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from matplotlib import pyplot as plt
from models_and_data import ModelsAndData
from quantized_float_tf import quantized_float, quantized_float_tanh, quantized_float_sigmoid, quantized_float_softmax

X_train = np.load('../models/quickdraw_dataset/X_train.npy', allow_pickle=True)
y_train = np.load('../models/quickdraw_dataset/y_train.npy', allow_pickle=True)
X_test = np.load('../models/quickdraw_dataset/X_test.npy', allow_pickle=True)
y_test = np.load('../models/quickdraw_dataset/y_test.npy', allow_pickle=True)

quantizer_dict = \
    {
        'quantized_input':
            {
                'activation': quantized_float(4, 4)
            },
        'lstm_1':
            {
                'activation': quantized_float_tanh(4, 4),
                'recurrent_activation': quantized_float_sigmoid(4, 4),
                'kernel_quantizer': quantized_float(4, 4),
                'recurrent_quantizer': quantized_float(4, 4),
                'bias_quantizer': quantized_float(4, 4),
                'state_quantizer': quantized_float(4, 4)
            },
        'dense_3':
            {
                'activation': quantized_float(4, 4),
                'kernel_quantizer': quantized_float(4, 4),
                'bias_quantizer': quantized_float(4, 4)
            },
        'dense_5':
            {
                'activation': quantized_float(4, 4),
                'kernel_quantizer': quantized_float(4, 4),
                'bias_quantizer': quantized_float(4, 4)
            },
        'dense_6':
            {
                'activation': quantized_float(4, 4),
                'kernel_quantizer': quantized_float(4, 4),
                'bias_quantizer': quantized_float(4, 4)
            },
        'softmax':
            {
                'activation': quantized_float_softmax(4, 4)
            }
    }
quickdraw_quantized = ModelsAndData.get_quickdraw_quantized_all_quantized(quantizer_dict=quantizer_dict)

mc = ModelCheckpoint('ckpt_quckdraw_full_quantized_4_4/quickdraw_full_quantized_4_4.h5',
                     verbose=1, monitor='val_loss', mode='min')
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
tb = TensorBoard(
    log_dir='tensorboard_logs_best_model_full_quantized_4_4',
    histogram_freq=1,
    write_graph=True,
    write_images=True,
    update_freq='batch',
)
quickdraw_quantized.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = quickdraw_quantized.fit(X_train, y_train, batch_size=512, epochs=50,
                                  validation_split=0.2, shuffle=True, callbacks=[mc, es, tb],
                                  use_multiprocessing=True, workers=28)
