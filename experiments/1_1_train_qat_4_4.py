import json
import pickle
from datetime import datetime
from json import JSONEncoder

import numpy as np
import qkeras.utils
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard

from shared_definitions import SharedDefinitons
from models_and_data import ModelsAndData


def _default(self, obj):
    return getattr(obj.__class__, "to_json", _default.default)(obj)


_default.default = JSONEncoder().default
JSONEncoder.default = _default

names = SharedDefinitons('quickdraw')

X_train = np.load('models_and_data/quickdraw_dataset/X_train.npy', allow_pickle=True).astype(np.float32)
y_train = np.load('models_and_data/quickdraw_dataset/y_train.npy', allow_pickle=True).astype(np.float32)
X_test = np.load('models_and_data/quickdraw_dataset/X_test.npy', allow_pickle=True).astype(np.float32)
y_test = np.load('models_and_data/quickdraw_dataset/y_test.npy', allow_pickle=True).astype(np.float32)

model_file_path, model_name = names.get_homo_quantized_model_names(4, 4, 0)

model = ModelsAndData.get_quickdraw()

quantized_model = qkeras.utils.model_quantize(model, names.get_config_4_4(), 0, names.get_custom_objects(),
                                              False)

time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
mc = ModelCheckpoint(
    model_name + '/checkpoint_' + model_name + '_' + time_str + '/' + model_name + '_{epoch:02d}_{val_loss:.2f}.h5'
    , verbose=2, monitor='val_loss', mode='min', save_best_only=True)
es = EarlyStopping(monitor='val_loss', mode='min', verbose=2, patience=5, restore_best_weights=True)
tb = TensorBoard(
    log_dir=model_name + '/tensorboard_logs_' + model_name + '_' + time_str,
    histogram_freq=1,
    write_graph=False,
    write_images=False,
    update_freq=1,
    profile_batch=0
)
quantized_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = quantized_model.fit(X_train, y_train, batch_size=512, epochs=50,
                              validation_split=0.2, shuffle=True, callbacks=[mc, es, tb],
                              use_multiprocessing=True, workers=12)

test_perf = quantized_model.evaluate(x=X_test, y=y_test, verbose=1, workers=12, use_multiprocessing=True,
                                     return_dict=True, callbacks=[mc, es, tb])

with open(model_name + '/' + model_name + '_train_val_loss_history.pkl', 'wb') as file_pi:
    pickle.dump(history.history, file_pi)

# rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
# rc('text', usetex=True)
# plt.figure(figsize=(16, 9))
# plt.plot(history.history['loss'], linewidth=1)
# plt.plot(history.history['val_loss'], linewidth=1)
# plt.title('Model Loss over Epochs for qmodel: ' + model_name)
# plt.ylabel('Loss')
# plt.xlabel('Epoch')
# plt.legend(['Training Sample Loss', 'Validation Sample Loss'])
# plt.savefig(model_name + '_train_val_loss_plot.pdf', dpi=500)
# plt.close()

with open(model_name + '/' + model_name + '_test_performance.json',
          'w') as json_file:
    json.dump(test_perf, json_file)

quantized_model.save(model_file_path + '.best')
