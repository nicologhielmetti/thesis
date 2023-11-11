import tensorflow as tf
import numpy as np

from quantized_float import quantized_float_tanh

x = np.load('test.npy')

q = quantized_float_tanh(4, 23, exp_offset=-14, use_exp_offset=1, ret_inf_on_ovf=0)


yq = q(x)
y = tf.tanh(x)

err = abs(y.numpy()-yq.numpy())

max_err = max(err)

amax_err = np.argmax(abs(y.numpy()-yq.numpy()))