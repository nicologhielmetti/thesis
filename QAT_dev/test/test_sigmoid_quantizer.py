import keras.activations
import numpy as np
import tensorflow
from qkeras import QActivation
from tensorflow.python.framework.ops import disable_eager_execution

from quantized_float import quantized_float_sigmoid

x = np.random.normal(0, 5, 1_000_000).astype(np.float32)

q = quantized_float_sigmoid(8, 23, ret_inf_on_ovf=0)

yq = q(x)

y = keras.activations.sigmoid(x)

err = abs(y.numpy() - yq.numpy())

max_err = max(err)

amax_err = np.argmax(abs(y.numpy() - yq.numpy()))
