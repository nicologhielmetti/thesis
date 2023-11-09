import tensorflow as tf
fpquantizer = tf.load_op_library('../../quantizers/tensorflow_flopo_quantizer/tf_fpquantizer.so.no_zero')
for i in range(0, 100):
    print(fpquantizer.quantize([[1.0+i, 2.0], [3.0, 4.0]], m_bits=8, e_bits=8, exp_offset=0, use_exp_offset=0, ret_inf_on_ovf=0, debug=0).numpy())