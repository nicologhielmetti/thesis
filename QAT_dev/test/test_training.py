import numpy as np
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import categorical_crossentropy
from sklearn.model_selection import train_test_split

from models_and_data import ModelsAndData


# Generate random data (replace this with your actual dataset)
X = np.random.rand(1000, 10)  # Example: 1000 samples, 1 feature
y = np.random.randint(0, 2, size=(1000, 5))  # Example: 1000 samples, 5 classes

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and compile the model
model = ModelsAndData.get_simple_dense_quantized(input_shape=(10,), units=5)
model.compile(loss=categorical_crossentropy, optimizer=Adam(), metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=200, batch_size=20, validation_data=(X_test, y_test))

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test loss: {loss:.4f}, Test accuracy: {accuracy:.4f}")
