from qkeras import BaseQuantizer
import tensorflow.keras.backend as K
import tensorflow as tf
import pyb11_fpquantizer.fpquantizer as fpquantizer

tf_fpquantizer = tf.load_op_library('tf_fpquantizer/tf_fpquantizer.so')


class quantized_float(BaseQuantizer):
    def __init__(self, exponent_bits, mantissa_bits, est_bias=0, use_est_bias=0, ret_inf_on_ovf=0):
        super(quantized_float, self).__init__()
        self.exponent_bits = exponent_bits
        self.mantissa_bits = mantissa_bits
        self.est_bias = est_bias
        self.use_est_bias = use_est_bias
        self.ret_inf_on_ovf = ret_inf_on_ovf
        # self.bits = exponent_bits + mantissa_bits + 1

    def __call__(self, x):
        x = K.cast_to_floatx(x)
        xq = tf_fpquantizer.quantize(x, m_bits=self.mantissa_bits, e_bits=self.exponent_bits, est_bias=self.est_bias,
                                     use_est_bias=self.use_est_bias, ret_inf_on_ovf=self.ret_inf_on_ovf)
        return x + tf.stop_gradient(-x + xq)

    def max(self):
        # return self._delta() / 2**-self.bits
        return fpquantizer.max(self.exponent_bits, self.mantissa_bits)

    def min(self):
        return fpquantizer.min(self.exponent_bits, self.mantissa_bits)

    def delta(self):
        return fpquantizer.delta(self.exponent_bits, self.mantissa_bits)


class quantized_float_tanh(BaseQuantizer):
    def __init__(self, exponent_bits, mantissa_bits, est_bias=0, use_est_bias=0, ret_inf_on_ovf=0):
        super(quantized_float_tanh, self).__init__()
        self.exponent_bits = exponent_bits
        self.mantissa_bits = mantissa_bits
        self.est_bias = est_bias
        self.use_est_bias = use_est_bias
        self.ret_inf_on_ovf = ret_inf_on_ovf

    def __call__(self, x):
        x = K.cast_to_floatx(x)
        xq = tf_fpquantizer.quantize(x, m_bits=self.mantissa_bits, e_bits=self.exponent_bits, est_bias=self.est_bias,
                                     use_est_bias=self.use_est_bias, ret_inf_on_ovf=self.ret_inf_on_ovf)
        yq = K.tanh(xq)
        y = K.tanh(x)
        return y + tf.stop_gradient(-y + yq)


class quantized_float_sigmoid(BaseQuantizer):
    def __init__(self, exponent_bits, mantissa_bits, est_bias=0, use_est_bias=0, ret_inf_on_ovf=0):
        super(quantized_float_sigmoid, self).__init__()
        self.exponent_bits = exponent_bits
        self.mantissa_bits = mantissa_bits
        self.est_bias = est_bias
        self.use_est_bias = use_est_bias
        self.ret_inf_on_ovf = ret_inf_on_ovf

    def __call__(self, x):
        x = K.cast_to_floatx(x)
        xq = tf_fpquantizer.quantize(x, m_bits=self.mantissa_bits, e_bits=self.exponent_bits, est_bias=self.est_bias,
                                     use_est_bias=self.use_est_bias, ret_inf_on_ovf=self.ret_inf_on_ovf)
        yq = K.hard_sigmoid(xq)
        y = K.hard_sigmoid(x)
        return y + tf.stop_gradient(-y + yq)


class quantized_float_softmax(BaseQuantizer):
    def __init__(self, exponent_bits, mantissa_bits, est_bias=0, use_est_bias=0, ret_inf_on_ovf=0):
        super(quantized_float_softmax, self).__init__()
        self.exponent_bits = exponent_bits
        self.mantissa_bits = mantissa_bits
        self.est_bias = est_bias
        self.use_est_bias = use_est_bias
        self.ret_inf_on_ovf = ret_inf_on_ovf

    def __call__(self, x):
        x = K.cast_to_floatx(x)
        xq = tf_fpquantizer.quantize(x, m_bits=self.mantissa_bits, e_bits=self.exponent_bits, est_bias=self.est_bias,
                                     use_est_bias=self.use_est_bias, ret_inf_on_ovf=self.ret_inf_on_ovf)
        yq = K.softmax(xq)
        y = K.softmax(x)
        return y + tf.stop_gradient(-y + yq)
