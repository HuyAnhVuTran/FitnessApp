import cv2
import numpy as np
from ultralytics import YOLO
import sys
import os

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
    Returns a status string and the knee angle.
    """
    # YOLOv8 COCO Keypoints:
    # 11: Left Hip, 13: Left Knee, 15: Left Ankle
    
    # Check confidence (optional, but good practice)
    # keypoints is tensor of shape (17, 3) -> x, y, conf
    
    hip = keypoints[11][:2].cpu().numpy()
    knee = keypoints[13][:2].cpu().numpy()
    ankle = keypoints[15][:2].cpu().numpy()
    
    # Calculate knee angle
    angle = calculate_angle(hip, knee, ankle)
    
    status = "Unknown"
    if angle > 160:
        status = "Standing"
    elif angle < 90:
        status = "Deep Squat (Good Form)"
    elif angle < 140:
        status = "Squatting (Go Lower)"
    else:
        status = "Squatting (Start)"
        
    return status, angle

def analyze_pushup(keypoints):
    """
    Analyzes pushup form based on YOLOv8 keypoints.
    Returns a status string and the elbow angle.
    """
    # YOLOv8 COCO Keypoints:
    # 5: Left Shoulder, 7: Left Elbow, 9: Left Wrist
    
    shoulder = keypoints[5][:2].cpu().numpy()
    elbow = keypoints[7][:2].cpu().numpy()
    wrist = keypoints[9][:2].cpu().numpy()
    
    # Calculate elbow angle
    angle = calculate_angle(shoulder, elbow, wrist)
    
    status = "Unknown"
    if angle > 160:
        status = "Up Position"
    elif angle < 90:
        status = "Deep Pushup (Good Form)"
    else:
        status = "Pushup (Go Lower)"
        
    return status, angle

def process_image(image_path, exercise_type):
    # Load model (will download yolov8n-pose.pt automatically)
    model = YOLO('yolov8n-pose.pt')
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image {image_path}")
        return

    # Run inference
    results = model(image)
    
    for result in results:
        # Get keypoints for the first person detected
        if result.keypoints is not None and len(result.keypoints) > 0:
            keypoints = result.keypoints.data[0] # Get first person
            
            if exercise_type.lower() == 'squat':
                status, angle = analyze_squat(keypoints)
                print(f"Analysis for {image_path}: {status} (Knee Angle: {int(angle)})")
            elif exercise_type.lower() == 'pushup':
                status, angle = analyze_pushup(keypoints)
                print(f"Analysis for {image_path}: {status} (Elbow Angle: {int(angle)})")
            
            # Save annotated image
            output_path = "annotated_" + os.path.basename(image_path)
            result.save(filename=output_path)
            print(f"Saved annotated image to {output_path}")
        else:
            print(f"No person detected in {image_path}")

if __name__ == "__main__":
    if len(sys.argv) > 2:
        img_path = sys.argv[1]
        ex_type = sys.argv[2]
        process_image(img_path, ex_type)
    else:
        print("Usage: python pose_estimation.py <image_path> <exercise_type>")
        print("Example: python pose_estimation.py test_pushup.jpg pushup")

