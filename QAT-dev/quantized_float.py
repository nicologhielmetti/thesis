import threading
import multiprocessing as mp

import numpy as np
from qkeras import BaseQuantizer
from keras import backend as K
import tensorflow as tf
from itertools import chain
import fpquantizer


class quantized_float(BaseQuantizer):
    def __init__(self, exponent_bits, mantissa_bits):
        super(quantized_float, self).__init__()
        self.exponent_bits = exponent_bits
        self.mantissa_bits = mantissa_bits
        self.n_cores = mp.cpu_count()

    def __call__(self, x):
        def thread_compute(chunk, j, exp_bits, man_bits):
            r[j] = fpquantizer.compute_nogil(chunk, exp_bits, man_bits)

        threads = []
        r = [None] * self.n_cores
        x = K.cast_to_floatx(x)
        chunks = np.array_split(x, self.n_cores)
        for i, c in enumerate(chunks):
            thread = threading.Thread(target=thread_compute,
                                      args=(c.tolist(), i, self.exponent_bits, self.mantissa_bits))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()
        xq = np.array(list(chain(*r)), dtype=np.float32).flatten()
        return x + tf.stop_gradient(-x + xq)
