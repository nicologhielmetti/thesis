from keras.layers import Softmax, Activation
from tensorflow.keras.layers import Input, LSTM, Dropout, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import GlorotUniform, Orthogonal, Zeros, LecunUniform, RandomUniform
from qkeras import QDense, QActivation, QLSTM
from quantized_float_tf import quantized_float


class ModelsAndData:
    @staticmethod
    def get_simple_dense(input_shape=(1,), units=10):
        input_layer = Input(shape=input_shape, dtype='float32', name='input')
        dense_layer = Dense(units=units, activation='linear',
                            kernel_initializer=RandomUniform(seed=1),
                            bias_initializer=RandomUniform(seed=1))(input_layer)
        output_layer = Activation(activation='softmax')(dense_layer)
        return Model(inputs=input_layer, outputs=output_layer, name='simple_dense')

    @staticmethod
    def get_simple_dense_quantized(input_shape=(1,), units=10):
        input_layer = Input(shape=input_shape, dtype='float32', name='input')
        dense_layer = QDense(units=units, activation='linear',
                             kernel_quantizer=quantized_float(4, 4),
                             bias_quantizer=quantized_float(4, 4),
                             kernel_initializer=RandomUniform(seed=1),
                             bias_initializer=RandomUniform(seed=1)
                             )(input_layer)
        output_layer = Activation(activation='softmax')(dense_layer)
        return Model(inputs=input_layer, outputs=output_layer, name='simple_dense_quantized')

    @staticmethod
    def get_quickdraw():
        # Input layer
        input_layer = Input(shape=(100, 3), dtype='float32', name='input_layer')

        input_linear = Activation(activation='linear', name='input_linear')(input_layer)

        # LSTM layer
        lstm_layer = LSTM(units=128, activation='tanh', recurrent_activation='sigmoid', use_bias=True,
                          kernel_initializer=GlorotUniform(seed=1),
                          recurrent_initializer=Orthogonal(gain=1.0, seed=1),
                          bias_initializer=Zeros(), dtype='float32', name='lstm_1')(input_linear)

        # Dropout layer
        dropout_1 = Dropout(rate=0.6, name='dropout_2', dtype='float32')(lstm_layer)

        # Dense layer
        dense_1 = Dense(units=256, activation='linear', use_bias=True,
                        kernel_initializer=GlorotUniform(seed=1), bias_initializer=Zeros(), name='dense_3',
                        dtype='float32')(
            dropout_1)

        # Dropout layer
        dropout_2 = Dropout(rate=0.5, name='dropout_4', dtype='float32')(dense_1)

        # Dense layer
        dense_2 = Dense(units=128, activation='linear', use_bias=True,
                        kernel_initializer=GlorotUniform(seed=1), bias_initializer=Zeros(), name='dense_5',
                        dtype='float32')(
            dropout_2)

        # Dense output layer
        dense_3 = Dense(units=5, use_bias=True,
                        kernel_initializer=LecunUniform(seed=1), bias_initializer=Zeros(), name='dense_6',
                        dtype='float32')(dense_2)

        output_layer = Activation(activation='softmax', dtype='float32', name='softmax')(dense_3)

        # Create the Keras model
        model = Model(inputs=input_layer, outputs=output_layer, name='quickdraw')
        return model

    @staticmethod
    def get_quickdraw_quantized(quantizer_dict):
        # Input layer
        input_layer = Input(shape=(100, 3), dtype='float32', name='input_layer')

        quantized_input = QActivation(activation=quantizer_dict['input_linear']['activation_quantizer'])(input_layer)

        # LSTM layer
        lstm_layer = QLSTM(units=128, activation=quantizer_dict['lstm_1']['activation_quantizer'],
                           recurrent_activation=quantizer_dict['lstm_1']['recurrent_activation_quantizer'],
                           use_bias=True,
                           kernel_initializer=GlorotUniform(seed=1),
                           recurrent_initializer=Orthogonal(gain=1.0, seed=1),
                           bias_initializer=Zeros(),
                           kernel_quantizer=quantizer_dict['lstm_1']['kernel_quantizer'],
                           recurrent_quantizer=quantizer_dict['lstm_1']['recurrent_quantizer'],
                           bias_quantizer=quantizer_dict['lstm_1']['bias_quantizer'],
                           state_quantizer=quantizer_dict['lstm_1']['state_quantizer'],
                           name='lstm_1')(quantized_input)

        # Dropout layer
        dropout_1 = Dropout(rate=0.6, name='dropout_2')(lstm_layer)

        # Dense layer
        dense_1 = QDense(units=256, activation=quantizer_dict['dense_3']['activation_quantizer'], use_bias=True,
                         kernel_initializer=GlorotUniform(seed=1), bias_initializer=Zeros(),
                         kernel_quantizer=quantizer_dict['dense_3']['kernel_quantizer'],
                         bias_quantizer=quantizer_dict['dense_3']['bias_quantizer'],
                         name='dense_3')(
            dropout_1)

        # Dropout layer
        dropout_2 = Dropout(rate=0.5, name='dropout_4')(dense_1)

        # Dense layer
        dense_2 = QDense(units=128, activation=quantizer_dict['dense_5']['activation_quantizer'], use_bias=True,
                         kernel_initializer=GlorotUniform(seed=1), bias_initializer=Zeros(),
                         kernel_quantizer=quantizer_dict['dense_5']['kernel_quantizer'],
                         bias_quantizer=quantizer_dict['dense_5']['bias_quantizer'],
                         name='dense_5')(
            dropout_2)

        # Dense output layer
        dense_3 = QDense(units=5, use_bias=True,
                         activation=quantizer_dict['dense_6']['activation_quantizer'],
                         kernel_initializer=LecunUniform(seed=1), bias_initializer=Zeros(),
                         kernel_quantizer=quantizer_dict['dense_6']['kernel_quantizer'],
                         bias_quantizer=quantizer_dict['dense_6']['bias_quantizer'],
                         name='dense_6')(dense_2)

        output_layer = QActivation(activation=quantizer_dict['softmax']['activation_quantizer'], name='softmax')(dense_3)

        # Create the Keras model
        model = Model(inputs=input_layer, outputs=output_layer, name='quickdraw_full_quantized')
        return model
