import numpy as np
import os
from tensorflow.keras.utils import to_categorical


def load_data():
    X, y = [], []
    for file in os.listdir("dataset/processed_sequences"):
        X.append(np.load(f"dataset/processed_sequences/{file}"))
        y.append(ord(file[0]) - ord('A'))  # assume file names like A_seq1.npy
    X = np.array(X)
    y = to_categorical(y, num_classes=26)
    return X, y
