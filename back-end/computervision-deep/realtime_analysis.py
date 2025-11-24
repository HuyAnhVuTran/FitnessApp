import cv2
import numpy as np
from ultralytics import YOLO
import sys

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

def run_realtime(exercise_type):
    # Load model
    print("Loading YOLOv8-pose model...")
    model = YOLO('yolov8n-pose.pt')
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print(f"Starting analysis for: {exercise_type}")
    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Run inference
        results = model(frame, verbose=False)
        
        # Annotate frame
        annotated_frame = results[0].plot()

        # Analyze form
        for result in results:
            if result.keypoints is not None and len(result.keypoints) > 0:
                keypoints = result.keypoints.data[0]
                
                status = "Unknown"
                angle = 0
                color = (255, 255, 255)
                
                if exercise_type.lower() == 'squat':
                    status, angle, color = analyze_squat(keypoints)
                elif exercise_type.lower() == 'pushup':
                    status, angle, color = analyze_pushup(keypoints)
                
                # Display status and angle
                cv2.putText(annotated_frame, f"Status: {status}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
                cv2.putText(annotated_frame, f"Angle: {int(angle)}", (10, 70), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

        # Show frame
        cv2.imshow('Real-time Form Analysis', annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        ex_type = sys.argv[1]
        run_realtime(ex_type)
    else:
        print("Usage: python realtime_analysis.py <exercise_type>")
        print("Example: python realtime_analysis.py squat")
