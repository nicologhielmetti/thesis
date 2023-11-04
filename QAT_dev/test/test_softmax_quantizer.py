import keras.activations
import numpy as np
import tensorflow

from quantized_float import quantized_float_softmax

x = np.random.rand(1_000_000, 5)
x = tensorflow.convert_to_tensor(x, dtype='float32')
q = quantized_float_softmax(4, 4, ret_inf_on_ovf=0)


yq = q(x)
y = keras.activations.softmax(x)

err = abs(y.numpy()-yq.numpy())

max_err = max(err.flatten())

amax_err = np.argmax(abs(y.numpy()-yq.numpy()))