# American Sign Language (ASL) Recognition System 🤟

A real-time, high-accuracy computer vision system capable of recognizing the American Sign Language alphabet (A-Z) using hand landmark detection.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10.9-orange)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Random%20Forest-green)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-red)
![Accuracy](https://img.shields.io/badge/Accuracy-99.7%25-brightgreen)

## 📖 Overview

This project implements a robust ASL alphabet recognizer that works with standard webcams. Unlike traditional CNN-based approaches that operate on raw pixels, this system uses **Google MediaPipe** to extract 3D hand landmarks (skeletal data) and classifies them using a **Random Forest** model.

This approach offers several advantages:
*   **High Performance:** Runs smoothly on CPU in real-time.
*   **Robustness:** Works well with different backgrounds, lighting conditions, and skin tones.
*   **Dual Hand Support:** Recognizes signs performed with either the left or right hand.
*   **No ROI Constraint:** Users can perform signs anywhere in the camera frame.

## ✨ Features

*   **Real-time Recognition:** Instant classification of 26 alphabet letters + 'del', 'space', and 'nothing'.
*   **Skeletal Tracking:** Visualizes the hand skeleton on the screen.
*   **High Accuracy:** Achieved **99.73% accuracy** on the augmented test set (174k samples).
*   **Dynamic Normalization:** Hand position and scale invariance (works regardless of hand distance).

## 📂 Project Structure

```
.
├── data/
│   ├── landmarks_dataset.pkl    # Processed landmark features
│   └── archive/                 # (Optional) Raw image dataset path
├── models/
│   └── asl_landmark_model.pkl   # Trained Random Forest model
├── src/
│   ├── hand_tracker.py          # MediaPipe detection wrapper
│   ├── preprocessing.py         # Image processing utilities
│   ├── dataset.py               # Data loading utilities
│   ├── features.py              # Feature extraction (HOG/Hu - Legacy)
│   └── classifiers.py           # Model definitions
├── webcam_demo.py               # Main real-time application
├── process_dataset.py           # Script to convert images to landmarks
├── train_mirrored.py            # Training script with data augmentation
├── main.py                      # Basic training script
└── requirements.txt             # Python dependencies
```

## 🚀 Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/TuganBasaran/American-Sign-Language-Interpreter.git
    cd American-Sign-Language-Interpreter
    ```

2.  **Create a virtual environment (Recommended):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *Note: This project strictly requires `mediapipe==0.10.9` for macOS compatibility.*

## 💻 Usage

### 1. Real-time Demo
To start the live recognition system using your webcam:

```bash
python3 webcam_demo.py
```
*   **Controls:** Press `q` to exit.
*   **Note for macOS:** Ensure you have granted camera permissions to your terminal (System Settings > Privacy & Security > Camera).

### 2. Training the Model (Optional)
If you want to retrain the model with your own dataset or the original Kaggle dataset:

**Step 1: Process Images to Landmarks**
Extracts skeletal points from the raw image dataset (~87,000 images).
```bash
python3 process_dataset.py
```

**Step 2: Train with Augmentation**
Trains the Random Forest model using both original and mirrored (left-hand simulation) data.
```bash
python3 train_mirrored.py
```

## 🧠 How It Works

1.  **Input:** Webcam frame or static image.
2.  **Hand Detection:** MediaPipe extracts 21 3D landmarks (joints) of the hand.
3.  **Normalization:**
    *   **Translation Invariance:** All points are shifted relative to the wrist (landmark 0), making the wrist (0,0,0).
    *   **Scale Invariance:** All coordinates are divided by the absolute maximum value, ensuring the hand size doesn't affect the input features.
4.  **Classification:** The normalized 63-dimensional vector (21 points × {x,y,z}) is fed into a Random Forest Classifier.
5.  **Output:** The predicted character is displayed on the screen.

## 📊 Performance

*   **Model:** Random Forest (100 estimators)
*   **Dataset:** Kaggle ASL Alphabet Train (87,000 images) + Mirror Augmentation
*   **Total Training Samples:** ~174,000
*   **Test Accuracy:** **99.73%**

---
*Project developed for CS 423.*
