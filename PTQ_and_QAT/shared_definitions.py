import os

from quantized_float import quantized_float, quantized_float_tanh, quantized_float_sigmoid, quantized_float_softmax


class SharedDefinitons:
    def __init__(self, base_name):
        self.base_name = base_name
        if not os.path.exists('saved_models'):
            os.makedirs('saved_models')

    def get_activation_and_weight_file_names(self):
        pass

    def get_activation_and_weight_analysis(self):
        pass

    def get_base_name(self):
        return self.base_name

    def get_homo_quantized_model_names(self, exp, man, exp_offset):
        if len(self.base_name.split('_')) == 1:
            self.base_name = self.base_name + '_homo_e' + str(exp) + '_m' + str(man) + '_o' + str(exp_offset)
        if not os.path.exists(self.base_name):
            os.makedirs(self.base_name)
        return 'saved_models/' + self.base_name + '.h5', self.base_name

    def get_hetero_quantized_model_names(self, tag):
        if len(self.base_name.split('_')) == 1:
            self.base_name = self.base_name + '_hetero_' + tag
        if not os.path.exists(self.base_name):
            os.makedirs(self.base_name)
        return 'saved_models/' + self.base_name + '.h5', self.base_name

    def get_flopo32_model_names(self):
        if len(self.base_name.split('_')) == 1:
            self.base_name = self.base_name + '_flopo32'
        if not os.path.exists(self.base_name):
            os.makedirs(self.base_name)
        return 'saved_models/' + self.base_name + '.h5', self.base_name

    def get_config_4_4(self):
        return \
            {
                'input_linear':
                    {
                        'activation_quantizer': quantized_float(4, 4)
                    },
                'lstm_1':
                    {
                        'activation_quantizer': quantized_float_tanh(4, 4),
                        'recurrent_activation_quantizer': quantized_float_sigmoid(4, 4),
                        'kernel_quantizer': quantized_float(4, 4),
                        'recurrent_quantizer': quantized_float(4, 4),
                        'bias_quantizer': quantized_float(4, 4),
                        'state_quantizer': quantized_float(4, 4)
                    },
                'dense_3':
                    {
                        'activation_quantizer': quantized_float(4, 4),
                        'kernel_quantizer': quantized_float(4, 4),
                        'bias_quantizer': quantized_float(4, 4)
                    },
                'dense_5':
                    {
                        'activation_quantizer': quantized_float(4, 4),
                        'kernel_quantizer': quantized_float(4, 4),
                        'bias_quantizer': quantized_float(4, 4)
                    },
                'dense_6':
                    {
                        'activation_quantizer': quantized_float(4, 4),
                        'kernel_quantizer': quantized_float(4, 4),
                        'bias_quantizer': quantized_float(4, 4)
                    },
                'softmax':
                    {
                        'activation_quantizer': quantized_float_softmax(4, 4)
                    }
            }

    def get_custom_objects(self):
        return \
            {
                'quantized_float': quantized_float,
                'quantized_float_softmax': quantized_float_softmax,
                'quantized_float_sigmoid': quantized_float_sigmoid,
                'quantized_float_tanh': quantized_float_tanh
            }

    def get_config_from_act_and_weight_config(self, weight_config, activation_config, kind_of_analysis, percentile_idx):
        if kind_of_analysis not in ['statistical_values', 'exact_values']:
            raise ValueError('kind_of_analysis has to be one of `statistical_values` or `exact_values`')
        act_dense_6_exp_offset = activation_config['layer_data']['dense_6_activation'][kind_of_analysis][
            'exponent_offset']
        act_lstm_1_exp_offset = activation_config['layer_data']['lstm_1_activation'][kind_of_analysis][
            'exponent_offset']
        state_lstm_1_exp_offset = activation_config['layer_data']['lstm_1_state_activation'][kind_of_analysis][
            'exponent_offset']
        act_softmax_exp_offset = activation_config['layer_data']['softmax_activation'][kind_of_analysis][
            'exponent_offset']
        act_dense_5_exp_offset = activation_config['layer_data']['dense_5_activation'][kind_of_analysis][
            'exponent_offset']
        act_dense_3_exp_offset = activation_config['layer_data']['dense_3_activation'][kind_of_analysis][
            'exponent_offset']

        act_dense_6_exp = activation_config['layer_data']['dense_6_activation'][kind_of_analysis]['min_exp_bit']
        act_lstm_1_exp = activation_config['layer_data']['lstm_1_activation'][kind_of_analysis]['min_exp_bit']
        state_lstm_1_exp = activation_config['layer_data']['lstm_1_state_activation'][kind_of_analysis][
            'min_exp_bit']
        act_softmax_exp = activation_config['layer_data']['softmax_activation'][kind_of_analysis]['min_exp_bit']
        act_dense_5_exp = activation_config['layer_data']['dense_5_activation'][kind_of_analysis]['min_exp_bit']
        act_dense_3_exp = activation_config['layer_data']['dense_3_activation'][kind_of_analysis]['min_exp_bit']

        act_dense_6_man = activation_config['layer_data']['dense_6_activation'][kind_of_analysis]['min_man_bit'][
            percentile_idx]
        act_lstm_1_man = activation_config['layer_data']['lstm_1_activation'][kind_of_analysis]['min_man_bit'][
            percentile_idx]
        state_lstm_1_man = \
            activation_config['layer_data']['lstm_1_state_activation'][kind_of_analysis]['min_man_bit'][percentile_idx]
        act_softmax_man = activation_config['layer_data']['softmax_activation'][kind_of_analysis]['min_man_bit'][
            percentile_idx]
        act_dense_5_man = activation_config['layer_data']['dense_5_activation'][kind_of_analysis]['min_man_bit'][
            percentile_idx]
        act_dense_3_man = activation_config['layer_data']['dense_3_activation'][kind_of_analysis]['min_man_bit'][
            percentile_idx]

        wei_dense_6_exp_offset = weight_config['layer_data']['dense_6_w'][kind_of_analysis]['exponent_offset']
        wei_lstm_1_exp_offset = weight_config['layer_data']['lstm_1_w'][kind_of_analysis]['exponent_offset']
        rw_lstm_1_exp_offset = weight_config['layer_data']['lstm_1_rw'][kind_of_analysis]['exponent_offset']
        wei_dense_5_exp_offset = weight_config['layer_data']['dense_5_w'][kind_of_analysis]['exponent_offset']
        wei_dense_3_exp_offset = weight_config['layer_data']['dense_3_w'][kind_of_analysis]['exponent_offset']

        wei_dense_6_exp = weight_config['layer_data']['dense_6_w'][kind_of_analysis]['min_exp_bit']
        wei_lstm_1_exp = weight_config['layer_data']['lstm_1_w'][kind_of_analysis]['min_exp_bit']
        rw_lstm_1_exp = weight_config['layer_data']['lstm_1_rw'][kind_of_analysis]['min_exp_bit']
        wei_dense_5_exp = weight_config['layer_data']['dense_5_w'][kind_of_analysis]['min_exp_bit']
        wei_dense_3_exp = weight_config['layer_data']['dense_3_w'][kind_of_analysis]['min_exp_bit']

        wei_dense_6_man = weight_config['layer_data']['dense_6_w'][kind_of_analysis]['min_man_bit'][percentile_idx]
        wei_lstm_1_man = weight_config['layer_data']['lstm_1_w'][kind_of_analysis]['min_man_bit'][percentile_idx]
        rw_lstm_1_man = weight_config['layer_data']['lstm_1_rw'][kind_of_analysis]['min_man_bit'][percentile_idx]
        wei_dense_5_man = weight_config['layer_data']['dense_5_w'][kind_of_analysis]['min_man_bit'][percentile_idx]
        wei_dense_3_man = weight_config['layer_data']['dense_3_w'][kind_of_analysis]['min_man_bit'][percentile_idx]

        b_dense_6_exp_offset = weight_config['layer_data']['dense_6_b'][kind_of_analysis]['exponent_offset']
        b_lstm_1_exp_offset = weight_config['layer_data']['lstm_1_b'][kind_of_analysis]['exponent_offset']
        b_dense_5_exp_offset = weight_config['layer_data']['dense_5_b'][kind_of_analysis]['exponent_offset']
        b_dense_3_exp_offset = weight_config['layer_data']['dense_3_b'][kind_of_analysis]['exponent_offset']

        b_dense_6_exp = weight_config['layer_data']['dense_6_b'][kind_of_analysis]['min_exp_bit']
        b_lstm_1_exp = weight_config['layer_data']['lstm_1_b'][kind_of_analysis]['min_exp_bit']
        b_dense_5_exp = weight_config['layer_data']['dense_5_b'][kind_of_analysis]['min_exp_bit']
        b_dense_3_exp = weight_config['layer_data']['dense_3_b'][kind_of_analysis]['min_exp_bit']

        b_dense_6_man = weight_config['layer_data']['dense_6_b'][kind_of_analysis]['min_man_bit'][percentile_idx]
        b_lstm_1_man = weight_config['layer_data']['lstm_1_b'][kind_of_analysis]['min_man_bit'][percentile_idx]
        b_dense_5_man = weight_config['layer_data']['dense_5_b'][kind_of_analysis]['min_man_bit'][percentile_idx]
        b_dense_3_man = weight_config['layer_data']['dense_3_b'][kind_of_analysis]['min_man_bit'][percentile_idx]

        quantizers_config = \
            {
                'quantized_input':
                    {
                        'activation_quantizer': quantized_float(8, 8)
                    },
                'lstm_1':
                    {
                        'activation_quantizer': quantized_float_tanh(act_lstm_1_exp, act_lstm_1_man,
                                                                     act_lstm_1_exp_offset,
                                                                     use_exp_offset=1),
                        'recurrent_activation_quantizer': quantized_float_sigmoid(state_lstm_1_exp, state_lstm_1_man,
                                                                                  state_lstm_1_exp_offset,
                                                                                  use_exp_offset=1),
                        'kernel_quantizer': quantized_float(wei_lstm_1_exp, wei_lstm_1_man, wei_lstm_1_exp_offset,
                                                            use_exp_offset=1),
                        'recurrent_quantizer': quantized_float(rw_lstm_1_exp, rw_lstm_1_man, rw_lstm_1_exp_offset,
                                                               use_exp_offset=1),
                        'bias_quantizer': quantized_float(b_lstm_1_exp, b_lstm_1_man, b_lstm_1_exp_offset,
                                                          use_exp_offset=1),
                        'state_quantizer': quantized_float(state_lstm_1_exp, state_lstm_1_man, state_lstm_1_exp_offset,
                                                           use_exp_offset=1)
                    },
                'dense_3':
                    {
                        'activation_quantizer': quantized_float(act_dense_3_exp, act_dense_3_man,
                                                                act_dense_3_exp_offset,
                                                                use_exp_offset=1),
                        'kernel_quantizer': quantized_float(wei_dense_3_exp, wei_dense_3_man, wei_dense_3_exp_offset,
                                                            use_exp_offset=1),
                        'bias_quantizer': quantized_float(b_dense_3_exp, b_dense_3_man, b_dense_3_exp_offset,
                                                          use_exp_offset=1)
                    },
                'dense_5':
                    {
                        'activation_quantizer': quantized_float(act_dense_5_exp, act_dense_5_man,
                                                                act_dense_5_exp_offset,
                                                                use_exp_offset=1),
                        'kernel_quantizer': quantized_float(wei_dense_5_exp, wei_dense_5_man, wei_dense_5_exp_offset,
                                                            use_exp_offset=1),
                        'bias_quantizer': quantized_float(b_dense_5_exp, b_dense_5_man, b_dense_5_exp_offset,
                                                          use_exp_offset=1)
                    },
                'dense_6':
                    {
                        'activation_quantizer': quantized_float(act_dense_6_exp, act_dense_6_man,
                                                                act_dense_6_exp_offset,
                                                                use_exp_offset=1),
                        'kernel_quantizer': quantized_float(wei_dense_6_exp, wei_dense_6_man, wei_dense_6_exp_offset,
                                                            use_exp_offset=1),
                        'bias_quantizer': quantized_float(b_dense_6_exp, b_dense_6_man, b_dense_6_exp_offset,
                                                          use_exp_offset=1)
                    },
                'softmax':
                    {
                        'activation_quantizer': quantized_float_softmax(act_softmax_exp, act_softmax_man,
                                                                        act_softmax_exp_offset,
                                                                        use_exp_offset=1)
                    }
            }
        return quantizers_config
