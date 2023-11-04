import json
from json import JSONEncoder

from qkeras import BaseQuantizer
import tensorflow.keras.backend as K
import tensorflow as tf
import python_bindings11_flopo_quantizer.fpquantizer as fpquantizer

tf_fpquantizer = tf.load_op_library('../quantizers/tensorflow_flopo_quantizer/tf_fpquantizer.so')


class quantized_float(BaseQuantizer):
    def __init__(self, exponent_bits, mantissa_bits, exp_offset=0, use_exp_offset=0, ret_inf_on_ovf=0, debug=0,
                 quantizer_name=None, alpha=None):
        super(quantized_float, self).__init__()
        if quantizer_name is None:
            self.quantizer_name = self.__class__.__name__
        else:
            self.quantizer_name = quantizer_name
        self.exponent_bits = exponent_bits
        self.mantissa_bits = mantissa_bits
        self.exp_offset = exp_offset
        self.use_exp_offset = use_exp_offset
        self.ret_inf_on_ovf = ret_inf_on_ovf
        self.debug = debug
        if alpha is None:
            self.alpha = 1.0
        else:
            self.alpha = alpha

    def __call__(self, x):
        x = K.cast_to_floatx(x)
        xq = tf_fpquantizer.quantize(x, m_bits=self.mantissa_bits, e_bits=self.exponent_bits,
                                     exp_offset=self.exp_offset,
                                     use_exp_offset=self.use_exp_offset, ret_inf_on_ovf=self.ret_inf_on_ovf,
                                     debug=self.debug)
        return x + tf.stop_gradient(-x + xq)

    def max(self):
        if self.use_exp_offset:
            return fpquantizer.max(self.exp_offset, self.exponent_bits, self.mantissa_bits)
        else:
            return fpquantizer.max(self.exponent_bits, self.mantissa_bits)

    def min(self):
        if self.use_exp_offset:
            return fpquantizer.min(self.exp_offset, self.exponent_bits, self.mantissa_bits)
        else:
            return fpquantizer.min(self.exponent_bits, self.mantissa_bits)

    def delta(self):
        if self.use_exp_offset:
            return fpquantizer.delta(self.exp_offset, self.exponent_bits, self.mantissa_bits)
        else:
            return fpquantizer.delta(self.exponent_bits, self.mantissa_bits)

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def get_config(self):
        config = {
            "quantizer_name": self.quantizer_name,
            "exponent_bits": self.exponent_bits,
            "mantissa_bits": self.mantissa_bits,
            "exp_offset": self.exp_offset,
            "use_exp_offset": self.use_exp_offset,
            "ret_inf_on_ovf": self.ret_inf_on_ovf,
            "debug": self.debug,
            "alpha": self.alpha
        }
        return config

    def to_json(self):
        cfg = {'class_name': self.__class__.__name__, 'config': self.get_config()}
        return cfg


class quantized_float_tanh(BaseQuantizer):
    def __init__(self, exponent_bits, mantissa_bits, exp_offset=0, use_exp_offset=0, ret_inf_on_ovf=0, debug=0,
                 quantizer_name=None, alpha=None):
        super(quantized_float_tanh, self).__init__()
        if quantizer_name is None:
            self.quantizer_name = self.__class__.__name__
        else:
            self.quantizer_name = quantizer_name
        self.exponent_bits = exponent_bits
        self.mantissa_bits = mantissa_bits
        self.exp_offset = exp_offset
        self.use_exp_offset = use_exp_offset
        self.ret_inf_on_ovf = ret_inf_on_ovf
        self.debug = debug
        if alpha is None:
            self.alpha = 1.0
        else:
            self.alpha = alpha

    def __call__(self, x):
        x = K.cast_to_floatx(x)

        # not quantized for STE
        y = K.tanh(x)
        # quantized input and output
        xq = tf_fpquantizer.quantize(x, m_bits=self.mantissa_bits, e_bits=self.exponent_bits,
                                     exp_offset=self.exp_offset,
                                     use_exp_offset=self.use_exp_offset, ret_inf_on_ovf=self.ret_inf_on_ovf,
                                     debug=self.debug)
        yx = K.tanh(xq)

        yq = tf_fpquantizer.quantize(yx, m_bits=self.mantissa_bits, e_bits=self.exponent_bits,
                                     exp_offset=self.exp_offset,
                                     use_exp_offset=self.use_exp_offset, ret_inf_on_ovf=self.ret_inf_on_ovf,
                                     debug=self.debug)

        return y + tf.stop_gradient(-y + yq)

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def get_config(self):
        config = {
            "quantizer_name": self.quantizer_name,
            "exponent_bits": self.exponent_bits,
            "mantissa_bits": self.mantissa_bits,
            "exp_offset": self.exp_offset,
            "use_exp_offset": self.use_exp_offset,
            "ret_inf_on_ovf": self.ret_inf_on_ovf,
            "debug": self.debug,
            "alpha": self.alpha
        }
        return config

    def to_json(self):
        cfg = {'class_name': self.__class__.__name__, 'config': self.get_config()}
        return cfg


class quantized_float_sigmoid(BaseQuantizer):
    def __init__(self, exponent_bits, mantissa_bits, exp_offset=0, use_exp_offset=0, ret_inf_on_ovf=0, debug=0,
                 quantizer_name=None, alpha=None):
        super(quantized_float_sigmoid, self).__init__()
        if quantizer_name is None:
            self.quantizer_name = self.__class__.__name__
        else:
            self.quantizer_name = quantizer_name
        self.exponent_bits = exponent_bits
        self.mantissa_bits = mantissa_bits
        self.exp_offset = exp_offset
        self.use_exp_offset = use_exp_offset
        self.ret_inf_on_ovf = ret_inf_on_ovf
        self.debug = debug
        if alpha is None:
            self.alpha = 1.0
        else:
            self.alpha = alpha

    def __call__(self, x):
        x = K.cast_to_floatx(x)
        y = K.sigmoid(x)
        xq = tf_fpquantizer.quantize(x, m_bits=self.mantissa_bits, e_bits=self.exponent_bits,
                                     exp_offset=self.exp_offset,
                                     use_exp_offset=self.use_exp_offset, ret_inf_on_ovf=self.ret_inf_on_ovf,
                                     debug=self.debug)
        yx = K.sigmoid(xq)
        yq = tf_fpquantizer.quantize(yx, m_bits=self.mantissa_bits, e_bits=self.exponent_bits,
                                     exp_offset=self.exp_offset,
                                     use_exp_offset=self.use_exp_offset, ret_inf_on_ovf=self.ret_inf_on_ovf,
                                     debug=self.debug)
        return y + tf.stop_gradient(-y + yq)

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def get_config(self):
        config = {
            "quantizer_name": self.quantizer_name,
            "exponent_bits": self.exponent_bits,
            "mantissa_bits": self.mantissa_bits,
            "exp_offset": self.exp_offset,
            "use_exp_offset": self.use_exp_offset,
            "ret_inf_on_ovf": self.ret_inf_on_ovf,
            "debug": self.debug,
            "alpha": self.alpha
        }
        return config

    def to_json(self):
        cfg = {'class_name': self.__class__.__name__, 'config': self.get_config()}
        return cfg


class quantized_float_softmax(BaseQuantizer):
    def __init__(self, exponent_bits, mantissa_bits, exp_offset=0, use_exp_offset=0, ret_inf_on_ovf=0, debug=0,
                 quantizer_name=None, alpha=None):
        super(quantized_float_softmax, self).__init__()
        if quantizer_name is None:
            self.quantizer_name = self.__class__.__name__
        else:
            self.quantizer_name = quantizer_name
        self.exponent_bits = exponent_bits
        self.mantissa_bits = mantissa_bits
        self.exp_offset = exp_offset
        self.use_exp_offset = use_exp_offset
        self.ret_inf_on_ovf = ret_inf_on_ovf
        self.debug = debug
        if alpha is None:
            self.alpha = 1.0
        else:
            self.alpha = alpha

    def __call__(self, x):
        x = K.cast_to_floatx(x)
        y = K.softmax(x)
        xq = tf_fpquantizer.quantize(x, m_bits=self.mantissa_bits, e_bits=self.exponent_bits,
                                     exp_offset=self.exp_offset,
                                     use_exp_offset=self.use_exp_offset, ret_inf_on_ovf=self.ret_inf_on_ovf,
                                     debug=self.debug)
        yx = K.softmax(xq)
        yq = tf_fpquantizer.quantize(yx, m_bits=self.mantissa_bits, e_bits=self.exponent_bits,
                                     exp_offset=self.exp_offset,
                                     use_exp_offset=self.use_exp_offset, ret_inf_on_ovf=self.ret_inf_on_ovf,
                                     debug=self.debug)
        return y + tf.stop_gradient(-y + yq)

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def get_config(self):
        config = {
            "quantizer_name": self.quantizer_name,
            "exponent_bits": self.exponent_bits,
            "mantissa_bits": self.mantissa_bits,
            "exp_offset": self.exp_offset,
            "use_exp_offset": self.use_exp_offset,
            "ret_inf_on_ovf": self.ret_inf_on_ovf,
            "debug": self.debug,
            "alpha": self.alpha
        }
        return config

    def to_json(self):
        cfg = {'class_name': self.__class__.__name__, 'config': self.get_config()}
        return cfg
