from qkeras import BaseQuantizer
import tensorflow.keras.backend as K
import tensorflow as tf
import pyb11_fpquantizer.fpquantizer as fpquantizer

tf_fpquantizer = tf.load_op_library('tf_fpquantizer/tf_fpquantizer.so')


class quantized_float(BaseQuantizer):
    def __init__(self, exponent_bits, mantissa_bits):
        super(quantized_float, self).__init__()
        self.exponent_bits = exponent_bits
        self.mantissa_bits = mantissa_bits
        self.bits = exponent_bits + mantissa_bits + 1

    def __call__(self, x):
        x = K.cast_to_floatx(x)
        xq = tf_fpquantizer.quantize(x, e_bits=self.exponent_bits, m_bits=self.mantissa_bits)
        return x + tf.stop_gradient(-x + xq)

    def max(self):
        return fpquantizer.max(self.exponent_bits, self.mantissa_bits)

    def min(self):
        return fpquantizer.min(self.exponent_bits, self.mantissa_bits)
