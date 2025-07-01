from .cnn_model import build_model
import numpy as np
from utils.data_loader import load_data

def train():
    X_train, y_train = load_data()
    model = build_model()
    model.fit(X_train, y_train, epochs=10, batch_size=8, validation_split=0.2)
    model.save("model/model_weights.h5")