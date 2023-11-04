from quantized_float import quantized_float
import tensorflow as tf

from models_and_data import ModelsAndData

x = tf.Variable([1.0, 2.1, 2.52, 1.2234])

q = quantized_float(6, 6)

with tf.GradientTape() as tape:
    y = q(x)

dy_dx = tape.gradient(y, x)
