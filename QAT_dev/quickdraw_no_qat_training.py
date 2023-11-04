import json
from datetime import datetime
import sys
sys.path.extend(['/data1/home/ghielmetti/thesis', '/data1/home/ghielmetti/thesis/PTQ_dev', '/data1/home/ghielmetti/thesis/QAT_dev', '/data1/home/ghielmetti/thesis/models_and_data'])

import numpy as np
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from models_and_data import ModelsAndData

X_train = np.load('../models_and_data/quickdraw_dataset/X_train.npy', allow_pickle=True)
y_train = np.load('../models_and_data/quickdraw_dataset/y_train.npy', allow_pickle=True)
X_test = np.load('../models_and_data/quickdraw_dataset/X_test.npy', allow_pickle=True)
y_test = np.load('../models_and_data/quickdraw_dataset/y_test.npy', allow_pickle=True)

quickdraw_quantized = ModelsAndData.get_quickdraw()
model_id = 'quickdraw_not_quantized'

time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
# ec = EpochCallbacks(epoch)
mc = ModelCheckpoint('ckpt_' + model_id + '_' + time_str + '/' + model_id + '_{epoch:02d}_{val_loss:.2f}.h5',
                     verbose=2, monitor='val_loss', mode='min', save_best_only=True)
es = EarlyStopping(monitor='val_loss', mode='min', verbose=2, patience=5)
tb = TensorBoard(
    log_dir='tensorboard_logs/' + model_id + '_' + time_str,
    histogram_freq=1,
    write_graph=False,
    write_images=False,
    update_freq=1,
    profile_batch=0
)
quickdraw_quantized.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = quickdraw_quantized.fit(X_train, y_train, batch_size=512, epochs=50,
                                  validation_split=0.2, shuffle=True, callbacks=[mc, es, tb],
                                  use_multiprocessing=True, workers=28)

test_perf = quickdraw_quantized.evaluate(x=X_test, y=y_test, verbose=1, workers=28, use_multiprocessing=True,
                                         return_dict=True, callbacks=[mc, es, tb])
with open('ckpt_' + model_id + '_' + time_str + '/' + model_id + '_test_performance.json', 'w') as json_file:
    json.dump(test_perf, json_file)

