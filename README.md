# Real-Time Video Smile Detection (CNN & LeNet)

A robust Computer Vision and Deep Learning project designed to detect and classify smiles in real-time video streams. It uses a combination of classical Haar Cascade classifiers for face/mouth detection and a custom Convolutional Neural Network (LeNet architecture) to accurately predict whether a detected face is smiling or not.

## 📌 Overview

This project implements a complete pipeline for emotion recognition:
1. **Face & Mouth Localization:** Uses OpenCV's pre-trained Cascade Models to quickly isolate the face and mouth areas within a video frame.
2. **Preprocessing:** Crops, standardizes, and normalizes the detected regions to prepare them for the deep learning model.
3. **Classification:** Feeds the processed image array into a Keras/TensorFlow model built on the classic **LeNet** architecture to classify the emotion as "Smiling" or "Not Smiling".

## ✨ Key Features

- **Real-Time Video Inference:** Processes `.mp4` video files frame-by-frame and overlays the prediction results (bounding boxes and labels) directly on the video.
- **Custom LeNet Architecture:** Implements a multi-layered CNN (Conv2D -> MaxPooling2D -> Dense) using Keras `Sequential` API for high-accuracy binary classification.
- **Object-Oriented Design:** The code is cleanly structured into separate modules (`Model_Trainer`, `Smile_Detector`, `Face_detector`) for maintainability.
- **Aspect Ratio Maintenance:** Includes a smart helper function (`get_size_ratio`) to resize cropped parts safely without distorting the image proportions.

## 🏗️ Architecture & Pipeline

### 1. `Extract_Parts.py` (Detection & Cropping)
Utilizes `cv2.CascadeClassifier` to detect faces (`haarcascade_frontalface_default.xml`) and smiles (`haarcascade_smile.xml`). The `Face_detector` class handles the heavy lifting of cropping, aspect-ratio checking, resizing, and saving datasets.

### 2. `LeNet.py` (The CNN Model)
A custom implementation of the LeNet architecture optimized for grayscale smile detection. It features two sets of Convolutional/Activation/Pooling layers followed by Flatten, Dense layers, and a final Softmax classifier.

### 3. `Detector.py` (Inference)
Loads the trained `.h5` / `saved_model` and executes a `while` loop over the video frames. It predicts the probability array (`Smiling` vs `Not Smiling`) and dynamically draws rectangles and text using OpenCV.

## 🚀 How to Use

### Prerequisites
Make sure you have the following libraries installed:
```bash
pip install opencv-python tensorflow keras numpy
```
# Running the Detector
To run the pre-trained model on a video file, simply execute:

```bash
from Detector import Smile_Detector
```

# Detect smiles in a video using the trained model directory
```bash
Smile_Detector.Detect_Video("video.mp4", "trained-model")
```

# Training a New Model
If you want to train the model from scratch on your own dataset:

```bash
from Trainer import Model_Trainer
```


# Train and save the model
```bash
Model_Trainer.Train("trained-model")
```

## 🛠️ Tech Stack
Language: Python

Deep Learning Framework: TensorFlow, Keras

Computer Vision: OpenCV (cv2)

Data Manipulation: NumPy
