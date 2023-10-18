import tensorflow as tf
from keras.layers import Softmax
from tensorflow.keras.layers import Input, LSTM, Dropout, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import GlorotUniform, Orthogonal, Zeros, LecunUniform


def get_lstm_model_quickdraw():
    # Input layer
    input_layer = Input(shape=(100, 3), dtype='float32', name='input')

    # LSTM layer
    lstm_layer = LSTM(units=128, activation='tanh', recurrent_activation='sigmoid', use_bias=True,
                      kernel_initializer=GlorotUniform(seed=None),
                      recurrent_initializer=Orthogonal(gain=1.0, seed=None),
                      bias_initializer=Zeros(), name='lstm_1')(input_layer)

    # Dropout layer
    dropout_1 = Dropout(rate=0.6, name='dropout_2')(lstm_layer)

    # Dense layer
    dense_1 = Dense(units=256, activation='linear', use_bias=True,
                    kernel_initializer=GlorotUniform(seed=None), bias_initializer=Zeros(), name='dense_3')(dropout_1)

    # Dropout layer
    dropout_2 = Dropout(rate=0.5, name='dropout_4')(dense_1)

    # Dense layer
    dense_2 = Dense(units=128, activation='linear', use_bias=True,
                    kernel_initializer=GlorotUniform(seed=None), bias_initializer=Zeros(), name='dense_5')(dropout_2)

    # Dense output layer
    dense_3 = Dense(units=5, use_bias=True,
                    kernel_initializer=LecunUniform(seed=None), bias_initializer=Zeros(), name='dense_6')(dense_2)

    output_layer = Softmax()(dense_3)

    # Create the Keras model
    model = Model(inputs=input_layer, outputs=output_layer)
    return model


def train_categorical(model, X_train, y_train, batch_size=32, epochs=10, validation_split=0.2):
    loss_function = 'categorical_crossentropy'
    optimizer = 'adam'
    metrics = ['accuracy']

    # Compile the model with the specified loss function, optimizer, and metrics
    model.compile(optimizer=optimizer, loss=loss_function, metrics=metrics)

    # Assuming you have one-hot encoded labels for classification
    # If not, you may need to one-hot encode your labels using tf.keras.utils.to_categorical()
    # y_train = tf.keras.utils.to_categorical(y_train, num_classes=num_classes)

    # Train the model
    model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=validation_split)
