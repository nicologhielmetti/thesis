import numpy as np
from matplotlib import pyplot as plt

from models_and_data import ModelsAndData
from quantized_float_tf import quantized_float


def learning_curve(history):
    plt.figure(figsize=(10, 8))
    plt.plot(history.history['loss'], linewidth=1)
    plt.plot(history.history['val_loss'], linewidth=1)
    plt.title('Model Loss over Epochs')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['training sample loss', 'validation sample loss'])
    plt.savefig('Learning_curve.pdf')
    plt.show()
    plt.close()


X_train = np.load('../models/quickdraw_dataset/X_train.npy', allow_pickle=True)
y_train = np.load('../models/quickdraw_dataset/y_train.npy', allow_pickle=True)
X_test = np.load('../models/quickdraw_dataset/X_test.npy', allow_pickle=True)
y_test = np.load('../models/quickdraw_dataset/y_test.npy', allow_pickle=True)

quantizer_dict = \
    {
        'lstm_1':
            {
                'kernel_quantizer': quantized_float(4, 4),
                'recurrent_quantizer': quantized_float(4, 4),
                'bias_quantizer': quantized_float(4, 4),
                'state_quantizer': quantized_float(4, 4)
            },
        'dense_3':
            {
                'kernel_quantizer': quantized_float(4, 4),
                'bias_quantizer': quantized_float(4, 4)
            },
        'dense_5':
            {
                'kernel_quantizer': quantized_float(4, 4),
                'bias_quantizer': quantized_float(4, 4)
            },
        'dense_6':
            {
                'kernel_quantizer': quantized_float(4, 4),
                'bias_quantizer': quantized_float(4, 4)
            }
    }
quickdraw_quantized = ModelsAndData.get_quickdraw_quantized(quantizer_dict=quantizer_dict)

quickdraw_quantized.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = quickdraw_quantized.fit(X_train, y_train, batch_size=512, epochs=50,
                                  validation_split=0.2, shuffle=True, callbacks=None,
                                  use_multiprocessing=True, workers=12)

learning_curve(history)
