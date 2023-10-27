import json
from json import JSONEncoder

from qkeras import BaseQuantizer
import tensorflow.keras.backend as K
import tensorflow as tf
import pyb11_fpquantizer.fpquantizer as fpquantizer

tf_fpquantizer = tf.load_op_library('tf_fpquantizer/tf_fpquantizer.so')


class quantized_float(BaseQuantizer):
    def __init__(self, exponent_bits, mantissa_bits, est_bias=0, use_est_bias=0, ret_inf_on_ovf=0, epoch=None):
        super(quantized_float, self).__init__()
        self.exponent_bits = exponent_bits
        self.mantissa_bits = mantissa_bits
        self.est_bias = est_bias
        self.use_est_bias = use_est_bias
        self.ret_inf_on_ovf = ret_inf_on_ovf
        self.epoch = epoch
        # self.bits = exponent_bits + mantissa_bits + 1

    def __call__(self, x):
        x = K.cast_to_floatx(x)
        # tf.summary.histogram(x.name + '_' + quantized_float.__name__, x, description="Before quantization", step=self.epoch.get_epochs())
        xq = tf_fpquantizer.quantize(x, m_bits=self.mantissa_bits, e_bits=self.exponent_bits, est_bias=self.est_bias,
                                     use_est_bias=self.use_est_bias, ret_inf_on_ovf=self.ret_inf_on_ovf)
        # tf.summary.histogram(xq.name + '_' + quantized_float.__name__, x, description="After quantization", step=self.epoch.get_epochs())
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

    class QuantizerEncoder(JSONEncoder):
        def default(self, o):
            return o.__dict__.update({'quantizer_name': self.__class__.__name__})


class quantized_float_tanh(BaseQuantizer):
    def __init__(self, exponent_bits, mantissa_bits, est_bias=0, use_est_bias=0, ret_inf_on_ovf=0, epoch=None):
        super(quantized_float_tanh, self).__init__()
        self.exponent_bits = exponent_bits
        self.mantissa_bits = mantissa_bits
        self.est_bias = est_bias
        self.use_est_bias = use_est_bias
        self.ret_inf_on_ovf = ret_inf_on_ovf
        self.epoch = epoch

    def __call__(self, x):
        x = K.cast_to_floatx(x)
        # tf.summary.histogram(x.name + '_' + quantized_float.__name__, x, description="Input before quantization", step=self.epoch.get_epochs())
        # not quantized for STE
        y = K.tanh(x)
        # tf.summary.histogram(y.name + '_' + quantized_float.__name__, x, description="Output no quantization", step=self.epoch.get_epochs())
        # quantized input and output
        xq = tf_fpquantizer.quantize(x, m_bits=self.mantissa_bits, e_bits=self.exponent_bits, est_bias=self.est_bias,
                                     use_est_bias=self.use_est_bias, ret_inf_on_ovf=self.ret_inf_on_ovf)
        # tf.summary.histogram(xq.name + '_' + quantized_float.__name__, x, description="Input after quantization", step=self.epoch.get_epochs())
        yx = K.tanh(xq)
        # tf.summary.histogram(yx.name + '_' + quantized_float.__name__, x, description="Input after quantization and "
        #                                                                               "function", step=self.epoch.get_epochs())
        yq = tf_fpquantizer.quantize(yx, m_bits=self.mantissa_bits, e_bits=self.exponent_bits, est_bias=self.est_bias,
                                     use_est_bias=self.use_est_bias, ret_inf_on_ovf=self.ret_inf_on_ovf)
        # tf.summary.histogram(yq.name + '_' + quantized_float.__name__, x, description="Output after quantization and "
        #                                                                               "function", step=self.epoch.get_epochs())

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

    class QuantizerEncoder(JSONEncoder):
        def default(self, o):
            return o.__dict__.update({'quantizer_name': self.__class__.__name__})


class quantized_float_sigmoid(BaseQuantizer):
    def __init__(self, exponent_bits, mantissa_bits, est_bias=0, use_est_bias=0, ret_inf_on_ovf=0, epoch=None):
        super(quantized_float_sigmoid, self).__init__()
        self.exponent_bits = exponent_bits
        self.mantissa_bits = mantissa_bits
        self.est_bias = est_bias
        self.use_est_bias = use_est_bias
        self.ret_inf_on_ovf = ret_inf_on_ovf
        self.epoch = epoch

    def __call__(self, x):
        x = K.cast_to_floatx(x)
        # tf.summary.histogram(x.name + '_' + quantized_float.__name__, x, description="Input before quantization", step=self.epoch.get_epochs())
        y = K.hard_sigmoid(x)
        # tf.summary.histogram(y.name + '_' + quantized_float.__name__, x, description="Output no quantization", step=self.epoch.get_epochs())
        xq = tf_fpquantizer.quantize(x, m_bits=self.mantissa_bits, e_bits=self.exponent_bits, est_bias=self.est_bias,
                                     use_est_bias=self.use_est_bias, ret_inf_on_ovf=self.ret_inf_on_ovf)
        # tf.summary.histogram(xq.name + '_' + quantized_float.__name__, x, description="Input after quantization", step=self.epoch.get_epochs())
        yx = K.hard_sigmoid(xq)
        # tf.summary.histogram(yx.name + '_' + quantized_float.__name__, x, description="Input after quantization and "
        #                                                                               "function", step=self.epoch.get_epochs())
        yq = tf_fpquantizer.quantize(yx, m_bits=self.mantissa_bits, e_bits=self.exponent_bits, est_bias=self.est_bias,
                                     use_est_bias=self.use_est_bias, ret_inf_on_ovf=self.ret_inf_on_ovf)
        # tf.summary.histogram(yq.name + '_' + quantized_float.__name__, x, description="Output after quantization and "
        #                                                                               "function", step=self.epoch.get_epochs())
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

    class QuantizerEncoder(JSONEncoder):
        def default(self, o):
            return o.__dict__.update({'quantizer_name': self.__class__.__name__})


class quantized_float_softmax(BaseQuantizer):
    def __init__(self, exponent_bits, mantissa_bits, est_bias=0, use_est_bias=0, ret_inf_on_ovf=0, epoch=None):
        super(quantized_float_softmax, self).__init__()
        self.exponent_bits = exponent_bits
        self.mantissa_bits = mantissa_bits
        self.est_bias = est_bias
        self.use_est_bias = use_est_bias
        self.ret_inf_on_ovf = ret_inf_on_ovf
        self.epoch = epoch

    def __call__(self, x):
        x = K.cast_to_floatx(x)
        # tf.summary.histogram(x.name + '_' + quantized_float.__name__, x, description="Input before quantization", step=self.epoch.get_epochs())
        y = K.softmax(x)
        # tf.summary.histogram(y.name + '_' + quantized_float.__name__, x, description="Output no quantization", step=self.epoch.get_epochs())
        xq = tf_fpquantizer.quantize(x, m_bits=self.mantissa_bits, e_bits=self.exponent_bits, est_bias=self.est_bias,
                                     use_est_bias=self.use_est_bias, ret_inf_on_ovf=self.ret_inf_on_ovf)
        # tf.summary.histogram(xq.name + '_' + quantized_float.__name__, x, description="Input after quantization", step=self.epoch.get_epochs())
        yx = K.softmax(xq)
        # tf.summary.histogram(yx.name + '_' + quantized_float.__name__, x, description="Input after quantization and "
        #                                                                               "function", step=self.epoch.get_epochs())
        yq = tf_fpquantizer.quantize(yx, m_bits=self.mantissa_bits, e_bits=self.exponent_bits, est_bias=self.est_bias,
                                     use_est_bias=self.use_est_bias, ret_inf_on_ovf=self.ret_inf_on_ovf)
        # tf.summary.histogram(yq.name + '_' + quantized_float.__name__, x, description="Output after quantization and "
        #                                                                               "function", step=self.epoch.get_epochs())
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

    class QuantizerEncoder(JSONEncoder):
        def default(self, o):
            return o.__dict__.update({'quantizer_name': self.__class__.__name__})
