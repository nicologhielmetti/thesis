import json
import pickle
from datetime import datetime
from json import JSONEncoder

import keras.models
import numpy as np
import qkeras.utils
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard

from shared_definitions import SharedDefinitons
from models_and_data import ModelsAndData


def _default(self, obj):
    return getattr(obj.__class__, "to_json", _default.default)(obj)


_default.default = JSONEncoder().default
JSONEncoder.default = _default

model_names = SharedDefinitons('quickdraw')
analyzed_names = SharedDefinitons('quickdraw_exact_after_flopo32')

X_train = np.load('models_and_data/quickdraw_dataset/X_train.npy', allow_pickle=True).astype(np.float32)
y_train = np.load('models_and_data/quickdraw_dataset/y_train.npy', allow_pickle=True).astype(np.float32)
X_test = np.load('models_and_data/quickdraw_dataset/X_test.npy', allow_pickle=True).astype(np.float32)
y_test = np.load('models_and_data/quickdraw_dataset/y_test.npy', allow_pickle=True).astype(np.float32)

analysis_file_path, analysis_name = analyzed_names.get_hetero_quantized_model_names('exact_after_flopo32')
model_file_path, model_name = model_names.get_flopo32_model_names()
act_config, wei_config = model_names.get_activation_and_weight_config()
config = model_names.get_config_from_act_and_weight_config(act_config, wei_config, 'exact_values')

model = keras.models.load_model(model_file_path)

qmodel = qkeras.utils.model_quantize(model, config, 0, model_names.get_custom_objects(), True)

time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
# mc = ModelCheckpoint(
#     analysis_name + '/checkpoint_' + analysis_name + '_' + time_str + '/' + analysis_name + '_{epoch:02d}_{val_loss:.2f}.h5'
#     , verbose=2, monitor='val_loss', mode='min', save_best_only=True)
# es = EarlyStopping(monitor='val_loss', mode='min', verbose=2, patience=5, restore_best_weights=True)

def get_activations(model, X_test):
    inp = model.input
    y_act = []
    for l in model.layers:
        i_model = keras.models.Model(inputs=[inp], outputs=[l.output])
        y_act.append(i_model.predict(X_test))
    return y_act


# qact = Common.get_activations_keras(qmodel, X_test[:1000])
# act  = Common.get_activations_keras(model, X_test[:1000])
qact = get_activations(model, X_test[:1000])
act = get_activations(qmodel, X_test[:1000])

# tb = TensorBoard(
#     log_dir=analysis_name + '/tensorboard_logs_' + analysis_name + '_' + time_str,
#     histogram_freq=1,
#     write_graph=False,
#     write_images=False,
#     update_freq=1,
#     profile_batch=0
# )
# qmodel.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# # history = qmodel.fit(X_train, y_train, batch_size=512, epochs=50,
# #                               validation_split=0.2, shuffle=True, callbacks=[mc, es, tb],
# #                               use_multiprocessing=True, workers=12)
#
# test_perf_q = qmodel.evaluate(x=X_test, y=y_test, verbose=1, workers=12, use_multiprocessing=True,
#                                      return_dict=True, callbacks=[tb]) # mc, es,])
#
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# test_perf = model.evaluate(x=X_test, y=y_test, verbose=1, workers=12, use_multiprocessing=True,
#                                      return_dict=True, callbacks=[tb])
#
# # with open(analysis_name + '/' + analysis_name + '_train_val_loss_history.pkl', 'wb') as file_pi:
# #     pickle.dump(history.history, file_pi)
#
# # rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
# # rc('text', usetex=True)
# # plt.figure(figsize=(16, 9))
# # plt.plot(history.history['loss'], linewidth=1)
# # plt.plot(history.history['val_loss'], linewidth=1)
# # plt.title('Model Loss over Epochs for qmodel: ' + analysis_name)
# # plt.ylabel('Loss')
# # plt.xlabel('Epoch')
# # plt.legend(['Training Sample Loss', 'Validation Sample Loss'])
# # plt.savefig(analysis_name + '_train_val_loss_plot.pdf', dpi=500)
# # plt.close()
#
# # with open(analysis_name + '/' + analysis_name + '_test_performance.json',
# #           'w') as json_file:
# #     json.dump(test_perf_q, json_file)
#
# def get_activations(model, X_test):
#     inp = model.input
#     y_act = []
#     for l in model.layers:
#         i_model = keras.models.Model(inputs=[inp], outputs=[l.output])
#         y_act.append(i_model.predict(X_test))
#     return y_act
#
#
# # qact = Common.get_activations_keras(qmodel, X_test[:1000])
# # act  = Common.get_activations_keras(model, X_test[:1000])
# qact = get_activations(model, X_test[:1000])
# act = get_activations(qmodel, X_test[:1000])
#
# # qmodel.save(analysis_file_path + '.best')