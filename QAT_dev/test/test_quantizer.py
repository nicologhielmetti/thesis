import numpy as np
from quantized_float import quantized_float

x = np.random.uniform(low=(np.float32(-1*2**-((2**7)-1)*(sum([1/(2**i) for i in range(0,24)])))),
                      high=(np.float32(2**((2**7)-1)*(sum([1/(2**i) for i in range(0,24)])))), size=50_000_000)
x = x.astype(np.float32)
q = quantized_float(8, 23, ret_inf_on_ovf=0, exp_offset=0, use_exp_offset=0)

xq = q(x).numpy()

d = abs(x-xq)
