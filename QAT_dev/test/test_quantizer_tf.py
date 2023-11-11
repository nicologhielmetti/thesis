import keras.activations
import numpy as np
import tensorflow
from qkeras import QActivation
from tensorflow.python.framework.ops import disable_eager_execution

from quantized_float import quantized_float

# x = np.random.normal(2**2, 5, 1_000_000).astype(np.float32)
x = np.load('test.npy')
q = quantized_float(4, 23, exp_offset=-14, use_exp_offset=1, ret_inf_on_ovf=0)

xq = q(x)

err = abs(x - xq.numpy())

max_err = max(err)

amax_err = np.argmax(abs(x - xq.numpy()))
