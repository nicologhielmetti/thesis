import keras.activations
import numpy as np
import tensorflow
from qkeras import QActivation
from tensorflow.python.framework.ops import disable_eager_execution

from quantized_float_tf import quantized_float

x = np.random.normal(0, 5, 1_000_000)
x = tensorflow.convert_to_tensor(x, dtype='float32')
q = quantized_float(4, 4, ret_inf_on_ovf=0)

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

disable_eager_execution()

# Define input shape and units
input_shape = 1  # Replace with your input shape
# hidden_units = 10  # Replace with the desired number of hidden units
output_units = 1  # Replace with the number of output units for your task

# Define the input layer
input_layer = Input(shape=(input_shape,))

# Define the hidden layer with ReLU activation
# hidden_layer = Dense(units=hidden_units, activation='relu')(input_layer)

# Define the output layer with softmax activation
output_layer = QActivation(activation=quantized_float(4, 4, ret_inf_on_ovf=0))(input_layer)

# Create the Keras model
model = Model(inputs=input_layer, outputs=output_layer)

yq = model(x)
y = keras.activations.hard_sigmoid(x)

err = abs(y.numpy() - yq.numpy())

max_err = max(err)

amax_err = np.argmax(abs(y.numpy() - yq.numpy()))
