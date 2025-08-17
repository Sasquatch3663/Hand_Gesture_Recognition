# Hand_Gesture_Recognition
This project uses Python, OpenCV, and MediaPipe to detect real-time hand gestures via webcam. It can recognize individual and combined gestures (e.g., Rock, Peace, Thumbs Up) and map finger distance (thumb–index) to system volume using pycaw. The system provides real-time feedback with visual overlays and can detect gestures from one or two hands.

# Hand Gesture Recognition System with Volume Control

## 📌 Overview
This project implements a **real-time hand gesture recognition system** using **Python, OpenCV, and MediaPipe**.  
It can recognize a variety of gestures and also map the distance between thumb and index finger to control the **system volume** (Windows only, via pycaw).

## 🎯 Features
- Real-time hand detection using **MediaPipe Hands**.
- Recognition of multiple gestures:
  - Fist, Open Hand, Thumbs Up 👍, Rock 🤘, Peace ✌, Call Me 🤙, OK 👌, ILY ❤️, etc.
- **Two-hand gesture support** (e.g., Double Rock 🤘, Double Peace ✌).
- **Volume control** via thumb–index finger distance.
- Visual feedback with gesture labels and overlay lines.

## 🛠️ Tech Stack
- **Python 3.8+**
- **OpenCV** for image processing and display.
- **MediaPipe** for hand landmark detection.
- **NumPy** for distance and mapping calculations.
- **Pycaw** for controlling Windows system volume.

## 🚀 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/hand-gesture-recognition.git
   cd hand-gesture-recognition
