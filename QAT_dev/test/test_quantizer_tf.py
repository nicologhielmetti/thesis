import tensorflow as tf
fpquantizer = tf.load_op_library('../tf_fpquantizer/tf_fpquantizer.so.old')
print(fpquantizer.quantize([[1.0, 2.0], [3.0, 4.0]], m_bits=0, e_bits=1).numpy())