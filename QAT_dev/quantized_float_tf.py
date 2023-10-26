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

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def get_config(self):
        config = {
            "exponent_bits": self.exponent_bits,
            "mantissa_bits": self.mantissa_bits,
            "est_bias": self.est_bias,
            "use_est_bias": self.use_est_bias,
            "ret_inf_on_ovf": self.ret_inf_on_ovf
        }
        return config


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
        # not quantized for STE
        y = K.tanh(x)
        # quantized input and output
        xq = tf_fpquantizer.quantize(x, m_bits=self.mantissa_bits, e_bits=self.exponent_bits, est_bias=self.est_bias,
                                     use_est_bias=self.use_est_bias, ret_inf_on_ovf=self.ret_inf_on_ovf)
        yx = K.tanh(xq)
        yq = tf_fpquantizer.quantize(yx, m_bits=self.mantissa_bits, e_bits=self.exponent_bits, est_bias=self.est_bias,
                                     use_est_bias=self.use_est_bias, ret_inf_on_ovf=self.ret_inf_on_ovf)

        return y + tf.stop_gradient(-y + yq)

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def get_config(self):
        config = {
            "exponent_bits": self.exponent_bits,
            "mantissa_bits": self.mantissa_bits,
            "est_bias": self.est_bias,
            "use_est_bias": self.use_est_bias,
            "ret_inf_on_ovf": self.ret_inf_on_ovf
        }
        return config


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
        y = K.hard_sigmoid(x)

        xq = tf_fpquantizer.quantize(x, m_bits=self.mantissa_bits, e_bits=self.exponent_bits, est_bias=self.est_bias,
                                     use_est_bias=self.use_est_bias, ret_inf_on_ovf=self.ret_inf_on_ovf)
        yx = K.hard_sigmoid(xq)
        yq = tf_fpquantizer.quantize(yx, m_bits=self.mantissa_bits, e_bits=self.exponent_bits, est_bias=self.est_bias,
                                     use_est_bias=self.use_est_bias, ret_inf_on_ovf=self.ret_inf_on_ovf)
        return y + tf.stop_gradient(-y + yq)

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def get_config(self):
        config = {
            "exponent_bits": self.exponent_bits,
            "mantissa_bits": self.mantissa_bits,
            "est_bias": self.est_bias,
            "use_est_bias": self.use_est_bias,
            "ret_inf_on_ovf": self.ret_inf_on_ovf
        }
        return config


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
        y = K.softmax(x)
        xq = tf_fpquantizer.quantize(x, m_bits=self.mantissa_bits, e_bits=self.exponent_bits, est_bias=self.est_bias,
                                     use_est_bias=self.use_est_bias, ret_inf_on_ovf=self.ret_inf_on_ovf)
        yx = K.softmax(xq)
        yq = tf_fpquantizer.quantize(yx, m_bits=self.mantissa_bits, e_bits=self.exponent_bits, est_bias=self.est_bias,
                                     use_est_bias=self.use_est_bias, ret_inf_on_ovf=self.ret_inf_on_ovf)
        return y + tf.stop_gradient(-y + yq)

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def get_config(self):
        config = {
            "exponent_bits": self.exponent_bits,
            "mantissa_bits": self.mantissa_bits,
            "est_bias": self.est_bias,
            "use_est_bias": self.use_est_bias,
            "ret_inf_on_ovf": self.ret_inf_on_ovf
        }
        return config
