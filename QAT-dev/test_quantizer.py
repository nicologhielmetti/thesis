import numpy as np
from quantized_float import quantized_float

x = np.random.rand(1_000_000)

q = quantized_float(4, 4)

xq = q(x)
