import numpy as np
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from models_and_data import ModelsAndData

X_train = np.load('../models/quickdraw_dataset/X_train.npy', allow_pickle=True)
y_train = np.load('../models/quickdraw_dataset/y_train.npy', allow_pickle=True)
X_test = np.load('../models/quickdraw_dataset/X_test.npy', allow_pickle=True)
y_test = np.load('../models/quickdraw_dataset/y_test.npy', allow_pickle=True)

quickdraw_quantized = ModelsAndData.get_quickdraw()

mc = ModelCheckpoint('best_model_full_quantized_4_4.h5', monitor='val_loss', mode='min', save_best_only=True)
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
tb = TensorBoard(
    log_dir='tensorboard_logs_best_model_not_quantized',
    histogram_freq=1,
    write_graph=True,
    write_images=True,
    update_freq='batch',
)
quickdraw_quantized.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = quickdraw_quantized.fit(X_train, y_train, batch_size=512, epochs=50,
                                  validation_split=0.2, shuffle=True, callbacks=[mc, es, tb],
                                  use_multiprocessing=True)
