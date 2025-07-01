import cv2
import numpy as np
from tensorflow.keras.models import load_model

IMG_SIZE = 64

# Load model
model = load_model('sign_language_cnn.h5')

# Label map
label_map = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'del', 5: 'E', 6: 'F', 7: 'G', 8: 'H', 9: 'I',
    10: 'J', 11: 'K', 12: 'L', 13: 'M', 14: 'N', 15: 'nothing', 16: 'O', 17: 'P',
    18: 'Q', 19: 'R', 20: 'S', 21: 'space', 22: 'T', 23: 'U', 24: 'V', 25: 'W',
    26: 'X', 27: 'Y', 28: 'Z'
}

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Draw a square box for gesture area
    x1, y1, x2, y2 = 100, 100, 300, 300
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
    
    # Extract ROI and preprocess
    roi = frame[y1:y2, x1:x2]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
    reshaped = resized.reshape(1, IMG_SIZE, IMG_SIZE, 1) / 255.0

    # Predict
    prediction = model.predict(reshaped)
    label_index = np.argmax(prediction)
    label = label_map[label_index]

    # Show prediction
    cv2.putText(frame, f"Prediction: {label}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow("Real-Time Sign Language Recognition", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
