import numpy as np
from quantized_float_tf import quantized_float

x = np.random.rand(100_000_000)

q = quantized_float(4, 4, ret_inf_on_ovf=1)

xq = q(x)
