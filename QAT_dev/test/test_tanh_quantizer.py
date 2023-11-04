import tensorflow as tf
import numpy as np

from quantized_float_tf import quantized_float_tanh

x = np.random.normal(0, 1000, 1_000_000).astype(np.float32)

q = quantized_float_tanh(8, 23, ret_inf_on_ovf=0)


yq = q(x)
y = tf.tanh(x)

err = abs(y.numpy()-yq.numpy())

max_err = max(err)

amax_err = np.argmax(abs(y.numpy()-yq.numpy()))