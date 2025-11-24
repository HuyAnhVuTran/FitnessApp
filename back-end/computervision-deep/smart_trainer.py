import cv2
import numpy as np
from ultralytics import YOLO
import tensorflow as tf
from tensorflow import keras
import os
import time

# Import training logic
# Ensure we can import from the current directory
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from deeplearning_exercise import train_model

def calculate_angle(a, b, c):
    """
    Calculates the angle at point b given points a, b, c.
    Points are (x, y) coordinates.
    """
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360-angle
        
    return angle

def analyze_squat(keypoints):
    """
    Analyzes squat form based on YOLOv8 keypoints.
    Returns a status string, the knee angle, and a color (BGR).
    """
    hip = keypoints[11][:2].cpu().numpy()
    knee = keypoints[13][:2].cpu().numpy()
    ankle = keypoints[15][:2].cpu().numpy()
    
    angle = calculate_angle(hip, knee, ankle)
    
    if angle > 160:
        return "Standing", angle, (0, 255, 255) # Yellow
    elif angle < 90:
        return "Deep Squat (Good)", angle, (0, 255, 0) # Green
    elif angle < 140:
        return "Go Lower", angle, (0, 0, 255) # Red
    else:
        return "Squatting", angle, (255, 255, 0) # Cyan

def analyze_pushup(keypoints):
    """
    Analyzes pushup form based on YOLOv8 keypoints.
    Returns a status string, the elbow angle, and a color (BGR).
    """
    shoulder = keypoints[5][:2].cpu().numpy()
    elbow = keypoints[7][:2].cpu().numpy()
    wrist = keypoints[9][:2].cpu().numpy()
    
    angle = calculate_angle(shoulder, elbow, wrist)
    
    if angle > 160:
        return "Up Position", angle, (0, 255, 255) # Yellow
    elif angle < 90:
        return "Deep Pushup (Good)", angle, (0, 255, 0) # Green
    else:
        return "Go Lower", angle, (0, 0, 255) # Red

import uuid

def save_frame(frame, exercise_type, base_dir):
    """Saves the current frame to the dataset directory."""
    directory = os.path.join(base_dir, "dataset", exercise_type)
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    filename = f"{uuid.uuid4().hex}.jpg"
    filepath = os.path.join(directory, filename)
    cv2.imwrite(filepath, frame)
    print(f"Saved {exercise_type} image to {filepath}")
    return filepath

def run_smart_trainer():
    # 1. Load or Train Classification Model
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, 'exercise_model.keras')
    dataset_dir = os.path.join(base_dir, 'dataset')
    
    class_names = ['pushup', 'squat'] # Default
    
    # Helper to load/train model
    def load_or_train():
        nonlocal class_names, classifier_model
        if not os.path.exists(model_path):
            print("Model not found. Training new model...")
            classifier_model, class_names = train_model()
        else:
            print("Loading existing classification model...")
            classifier_model = keras.models.load_model(model_path)
            if os.path.exists(dataset_dir):
                 train_ds = tf.keras.utils.image_dataset_from_directory(
                    dataset_dir, validation_split=0.2, subset="training", seed=123, image_size=(128, 128), batch_size=8, verbose=0
                )
                 class_names = train_ds.class_names
        return classifier_model, class_names

    classifier_model = None
    classifier_model, class_names = load_or_train()
    print(f"Classifier ready. Classes: {class_names}")

    # 2. Load YOLO Pose Model
    print("Loading YOLOv8-pose model...")
    pose_model = YOLO('yolov8n-pose.pt')

    # 3. Start Webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Starting Smart Trainer...")
    print("Press 'c' to toggle Data Collection Mode.")
    print("Press 'q' to quit.")

    current_exercise = "Unknown"
    confidence_score = 0.0
    frame_count = 0
    classification_interval = 30 # Classify every 30 frames
    
    capture_mode = False
    last_action_msg = ""
    last_action_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        frame_count += 1
        
        # --- UI Overlay ---
        annotated_frame = frame.copy()

        if capture_mode:
            # --- Capture Mode UI ---
            cv2.putText(annotated_frame, "DATA COLLECTION MODE", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(annotated_frame, "s: Save Squat | p: Save Pushup", (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(annotated_frame, "t: Retrain Model | c: Exit Mode", (10, 85), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Show last action message
            if time.time() - last_action_time < 2.0:
                 cv2.putText(annotated_frame, last_action_msg, (10, 120), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        else:
            # --- Normal Analysis Mode ---
            
            # Classification
            if frame_count % classification_interval == 0:
                img_resized = cv2.resize(frame, (128, 128))
                img_array = np.expand_dims(img_resized, axis=0)
                
                prediction = classifier_model.predict(img_array, verbose=0)
                predicted_index = np.argmax(prediction)
                current_exercise = class_names[predicted_index]
                confidence_score = 100 * np.max(prediction)

            # Pose Estimation & Analysis
            results = pose_model(frame, verbose=False)
            annotated_frame = results[0].plot()

            for result in results:
                if result.keypoints is not None and len(result.keypoints) > 0:
                    keypoints = result.keypoints.data[0]
                    
                    status = "Analyzing..."
                    angle = 0
                    color = (255, 255, 255)

                    if current_exercise.lower() == 'squat':
                        status, angle, color = analyze_squat(keypoints)
                    elif current_exercise.lower() == 'pushup':
                        status, angle, color = analyze_pushup(keypoints)
                    
                    # Display Analysis UI
                    cv2.putText(annotated_frame, f"Exercise: {current_exercise} ({confidence_score:.0f}%)", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(annotated_frame, f"Status: {status}", (10, 60), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                    cv2.putText(annotated_frame, f"Angle: {int(angle)}", (10, 90), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            cv2.putText(annotated_frame, "Press 'c' for Data Collection", (10, frame.shape[0] - 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Show frame
        cv2.imshow('Smart AI Trainer', annotated_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            capture_mode = not capture_mode
            print(f"Capture Mode: {capture_mode}")
        
        if capture_mode:
            if key == ord('s'):
                save_frame(frame, 'squat', base_dir)
                last_action_msg = "Saved Squat Image"
                last_action_time = time.time()
            elif key == ord('p'):
                save_frame(frame, 'pushup', base_dir)
                last_action_msg = "Saved Pushup Image"
                last_action_time = time.time()
            elif key == ord('t'):
                print("Retraining model...")
                last_action_msg = "Retraining... Please Wait"
                last_action_time = time.time()
                # Force UI update before blocking
                cv2.putText(annotated_frame, last_action_msg, (10, 120), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.imshow('Smart AI Trainer', annotated_frame)
                cv2.waitKey(1)
                
                classifier_model, class_names = train_model()
                last_action_msg = "Model Retrained!"
                last_action_time = time.time()

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_smart_trainer()
