import keras.activations
import numpy as np

from quantized_float_tf import quantized_float_tanh

x = np.random.normal(0, 0.1, 1_000_000)

q = quantized_float_tanh(4, 4, ret_inf_on_ovf=0)


yq = q(x)
y = keras.activations.tanh(x)

err = abs(y.numpy()-yq.numpy())

max_err = max(err)

amax_err = np.argmax(abs(y.numpy()-yq.numpy()))