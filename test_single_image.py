import cv2
import numpy as np
from tensorflow.keras.models import load_model

IMG_SIZE = 64  # Same as used in training

# Load model
model = load_model('sign_language_cnn.h5')

# Define your label map manually (as used in training)
label_map = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'del', 5: 'E', 6: 'F', 7: 'G', 8: 'H', 9: 'I',
    10: 'J', 11: 'K', 12: 'L', 13: 'M', 14: 'N', 15: 'nothing', 16: 'O', 17: 'P',
    18: 'Q', 19: 'R', 20: 'S', 21: 'space', 22: 'T', 23: 'U', 24: 'V', 25: 'W',
    26: 'X', 27: 'Y', 28: 'Z'
}

# Load and preprocess a test image
image_path = 'dataset/asl_alphabet_train/B/B1.jpg'
 # ✅ Update with your actual path
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
img = img.reshape(1, IMG_SIZE, IMG_SIZE, 1) / 255.0

# Predict
pred = model.predict(img)
predicted_label = np.argmax(pred)

print(f"Predicted Sign: {label_map[predicted_label]}")
