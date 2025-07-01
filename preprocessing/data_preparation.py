import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

IMG_SIZE = 64

def load_data(dataset_path):
    X, y = [], []
    labels = sorted(os.listdir(dataset_path))
    label_map = {label: idx for idx, label in enumerate(labels)}

    for label in labels:
        folder = os.path.join(dataset_path, label)
        for img_name in os.listdir(folder):
            img_path = os.path.join(folder, img_name)
            try:
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                X.append(img)
                y.append(label_map[label])
            except:
                pass
    
    X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1) / 255.0
    y = np.array(y)
    return train_test_split(X, y, test_size=0.2), label_map
