import threading
import multiprocessing as mp

import numpy as np
from qkeras import BaseQuantizer
import tensorflow.keras.backend as K
import tensorflow as tf
from itertools import chain
import fpquantizer


class quantized_float(BaseQuantizer):
    def __init__(self, exponent_bits, mantissa_bits):
        super(quantized_float, self).__init__()
        self.exponent_bits = exponent_bits
        self.mantissa_bits = mantissa_bits
        self.bits = exponent_bits + mantissa_bits + 1
        self.n_cores = mp.cpu_count()

    def __call__(self, x):
        x = K.cast_to_floatx(x)
        xq = fpquantizer.compute_nogil(x.flatten(), self.exponent_bits, self.mantissa_bits)
        return x + tf.stop_gradient(-x + xq)

        # def thread_compute(chunk, j, exp_bits, man_bits):
        #     r[j] = fpquantizer.compute_nogil(chunk, exp_bits, man_bits)
        # 
        # threads = []
        # r = [None] * self.n_cores
        # x = K.cast_to_floatx(x)
        # chunks = [e[0] if e.shape[0] == 1 else e for e in np.array_split(x, self.n_cores)]
        # for i, c in enumerate(chunks):
        #     thread = threading.Thread(target=thread_compute,
        #                               args=(c.tolist(), i, self.exponent_bits, self.mantissa_bits))
        #     threads.append(thread)
        #     thread.start()
        # 
        # for thread in threads:
        #     thread.join()
        # xq = np.array(list(chain(*r)), dtype=np.float32).flatten().reshape(x.shape)
        # return x + tf.stop_gradient(-x + xq)

    def max(self):
        return fpquantizer.max(self.exponent_bits, self.mantissa_bits)

    def min(self):
        return fpquantizer.min(self.exponent_bits, self.mantissa_bits)
