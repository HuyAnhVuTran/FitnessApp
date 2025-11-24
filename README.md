# AI Fitness Trainer & Form Analysis

This project is an intelligent fitness application that uses Computer Vision and Deep Learning to classify exercises and analyze user form in real-time. It helps users perform **Pushups** and **Squats** correctly by providing instant feedback on their technique.

## 🚀 Features

### 1. Exercise Classification (Deep Learning)
- **Technology**: Convolutional Neural Network (CNN) built with TensorFlow/Keras.
- **Function**: Automatically detects whether you are doing a Pushup or a Squat.
- **Accuracy**: ~87% on validation set.

### 2. Form Analysis (Pose Estimation)
- **Technology**: YOLOv8-Pose (Ultralytics).
- **Function**: Tracks key body joints (Shoulders, Hips, Knees, Ankles, Elbows) to calculate geometric angles.
- **Feedback**:
    - **Squats**: Measures knee angle to ensure you hit "depth" (< 90 degrees).
    - **Pushups**: Measures elbow angle to ensure full range of motion.

### 3. Real-time Webcam Feedback
- **Function**: Opens your webcam and overlays form analysis on the live video feed.
- **Visuals**:
    - **Green**: Good Form (Deep enough).
    - **Red**: Bad Form (Go lower).
    - **Yellow**: Standing / Up position.

## 🛠️ Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/HuyAnhVuTran/FitnessApp.git
    cd FitnessApp
    ```

2.  **Install Dependencies**:
    You need Python 3.8+. Install the required libraries:
    ```bash
    pip install tensorflow opencv-python ultralytics requests numpy
    ```

## 💻 Usage

Navigate to the computer vision directory:
```bash
cd back-end/computervision-deep
```

### 1. Real-time Analysis (Webcam)
This is the main feature. Run it to analyze your form live.
```bash
# For Squats
python realtime_analysis.py squat

# For Pushups
python realtime_analysis.py pushup
```
*Press 'q' to quit the window.*

### 2. Static Image Analysis
Analyze a specific image file for form.
```bash
python pose_estimation.py path/to/image.jpg <squat|pushup>
```

### 3. Train the Classifier
If you want to retrain the classification model with new data:
1.  Add images to `dataset/pushup` and `dataset/squat`.
2.  Run the training script:
    ```bash
    python deeplearning_exercise.py
    ```
    This saves the model to `exercise_model.keras`.

### 4. Populate Dataset
To download more training images from the web:
```bash
python populate_dataset.py
```

## 📂 Project Structure

- `realtime_analysis.py`: Main script for live webcam form analysis.
- `pose_estimation.py`: Logic for calculating angles and analyzing static images.
- `deeplearning_exercise.py`: Script to train the CNN classifier.
- `populate_dataset.py`: Helper to download images from the web.
- `dataset/`: Directory containing training images.
- `exercise_model.keras`: The trained deep learning model.

## 🤖 Tech Stack
- **Language**: Python
- **Computer Vision**: OpenCV, YOLOv8 (Ultralytics)
- **Deep Learning**: TensorFlow, Keras