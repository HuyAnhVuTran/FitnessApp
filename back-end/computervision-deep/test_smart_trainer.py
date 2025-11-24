import cv2
import numpy as np
from ultralytics import YOLO
import tensorflow as tf
from tensorflow import keras
import os
import sys

# Add current directory to path to allow imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from smart_trainer import analyze_pushup, analyze_squat

def test_static_image():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = os.path.join(base_dir, "test_pushup.jpg")

    model_path = os.path.join(base_dir, "exercise_model.keras")
    dataset_dir = os.path.join(base_dir, "dataset")

    if not os.path.exists(image_path):
        print(f"Error: {image_path} not found.")
        return

    print(f"Testing on image: {image_path}")

    # 1. Load Classification Model
    if not os.path.exists(model_path):
        print("Error: Model not found. Please run smart_trainer.py first to train.")
        return
    
    print("Loading classification model...")
    classifier_model = keras.models.load_model(model_path)
    
    # Get class names
    class_names = ['pushup', 'squat']
    if os.path.exists(dataset_dir):
         train_ds = tf.keras.utils.image_dataset_from_directory(
            dataset_dir, validation_split=0.2, subset="training", seed=123, image_size=(128, 128), batch_size=8, verbose=0
        )
         class_names = train_ds.class_names
    
    # 2. Classify Image
    img = cv2.imread(image_path)
    img_resized = cv2.resize(img, (128, 128))
    img_array = np.expand_dims(img_resized, axis=0)
    
    prediction = classifier_model.predict(img_array, verbose=0)
    predicted_index = np.argmax(prediction)
    current_exercise = class_names[predicted_index]
    confidence_score = 100 * np.max(prediction)
    
    print(f"Classified Exercise: {current_exercise} ({confidence_score:.2f}%)")

    # 3. Pose Analysis
    print("Running pose estimation...")
    pose_model = YOLO('yolov8n-pose.pt')
    results = pose_model(img, verbose=False)
    
    for result in results:
        if result.keypoints is not None and len(result.keypoints) > 0:
            keypoints = result.keypoints.data[0]
            
            status = "Unknown"
            angle = 0
            
            if current_exercise.lower() == 'squat':
                status, angle, color = analyze_squat(keypoints)
            elif current_exercise.lower() == 'pushup':
                status, angle, color = analyze_pushup(keypoints)
            
            print(f"Analysis Result: Status='{status}', Angle={angle:.2f}")
            
            # Annotate and save
            annotated_frame = results[0].plot()
            cv2.putText(annotated_frame, f"Exercise: {current_exercise}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(annotated_frame, f"Status: {status}", (10, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            output_path = os.path.join(base_dir, "test_result_smart.jpg")
            cv2.imwrite(output_path, annotated_frame)
            print(f"Saved annotated result to {output_path}")

if __name__ == "__main__":
    test_static_image()
