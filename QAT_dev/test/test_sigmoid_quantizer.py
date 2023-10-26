import keras.activations
import numpy as np
import tensorflow

from quantized_float_tf import quantized_float_sigmoid

x = np.random.normal(0, 5, 1_000_000)
x = tensorflow.convert_to_tensor(x, dtype='float32')
q = quantized_float_sigmoid(4, 4, ret_inf_on_ovf=0)

yq = q(x)
y = keras.activations.hard_sigmoid(x)

err = abs(y.numpy() - yq.numpy())

max_err = max(err)

amax_err = np.argmax(abs(y.numpy() - yq.numpy()))
