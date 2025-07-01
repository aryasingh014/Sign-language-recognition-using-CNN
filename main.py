from preprocessing.data_preparation import load_data
from model.cnn_model import create_cnn_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
import numpy as np

# Load and preprocess
(X_train, X_test, y_train, y_test), label_map = load_data('dataset/asl_alphabet_train/asl_alphabet_train')


# Prepare labels
num_classes = len(label_map)
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

# Build model
input_shape = X_train.shape[1:]
model = create_cnn_model(input_shape, num_classes)
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Train
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), batch_size=32)

# Save model
model.save("sign_language_cnn.h5")
