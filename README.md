# Sign-language-recognition-using-CNN

A deep learning-based project to recognize American Sign Language (ASL) hand gestures using a Convolutional Neural Network (CNN). This project supports both static image prediction and real-time sign detection via webcam.

---

## 📂 Project Structure

Sign language/
├── dataset/ # ASL Alphabet dataset (A-Z)
├── model/ # CNN model definition
├── preprocessing/ # Image loading & data prep
├── utils/ # Helper functions (optional)
├── real_time_recognition.py # Real-time webcam-based detection
├── test_single_image.py # Test with a static image
├── main.py # Model training script
├── requirements.txt # Required Python packages
└── README.md # You're here

---

## 🛠️ Tools & Technologies

| Tool/Library        | Purpose                            |
|---------------------|------------------------------------|
| Python 3.10         | Programming language               |
| TensorFlow / Keras  | CNN model building & training      |
| OpenCV              | Image processing & webcam access   |
| NumPy               | Data manipulation                  |
| scikit-learn        | Dataset splitting                  |
| matplotlib          | (Optional) Training visualization  |

---

## 📊 Model Architecture

- CNN with Conv2D, MaxPooling, Flatten, and Dense layers
- Input: Grayscale images (64×64 pixels)
- Output: 29 classes (A–Z, space, delete, nothing)
- Loss: `categorical_crossentropy`
- Optimizer: `Adam`
- Accuracy: ~95% on test set

---

## 🚀 How to Run

### 1. Clone the repository:

```bash
git clone https://github.com/your-username/sign-language-cnn.git
cd sign-language-cnn
