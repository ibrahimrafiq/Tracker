import cv2
import mediapipe as mp
import math
import numpy as np

# Initialize MediaPipe pose solution
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

def calculate_angle(a, b, c):
    """
    Calculate the angle between three points.
    
    Parameters:
    a (list): Point a [x, y]
    b (list): Point b [x, y]
    c (list): Point c [x, y]
    
    Returns:
    float: Angle in degrees
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return angle

# Open the video file
cap = cv2.VideoCapture('Enter Video Path')
paused = False
frame_number = 0

while cap.isOpened():
    if not paused:
        ret, frame = cap.read()
        if not ret:
            break
        frame_number += 1
    
    # Convert the image to RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the image and detect the pose
    results = pose.process(image)
    
    # Convert the image back to BGR for OpenCV
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    angles = []
    
    # Draw the pose annotation on the image
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        landmarks = results.pose_landmarks.landmark
        
        # Define the joints and their labels
        joints = [
            (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST, "Left Elbow"),
            (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_WRIST, "Right Elbow"),
            (mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.LEFT_ANKLE, "Left Knee"),
            (mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_KNEE, mp_pose.PoseLandmark.RIGHT_ANKLE, "Right Knee"),
            (mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_HIP, "Left Shoulder"),
            (mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_HIP, "Right Shoulder")
        ]
        
        for joint in joints:
            a = [landmarks[joint[0].value].x, landmarks[joint[0].value].y]
            b = [landmarks[joint[1].value].x, landmarks[joint[1].value].y]
            c = [landmarks[joint[2].value].x, landmarks[joint[2].value].y]
            
            angle = calculate_angle(a, b, c)
            angles.append((joint[3], angle))
    
    # Draw the box in the corner
    cv2.rectangle(image, (0, 0), (250, 200), (0, 0, 0), -1)
    
    # Display the angles in the box
    for idx, (joint, angle) in enumerate(angles):
        cv2.putText(image, f"{joint}: {int(angle)}", 
                    (10, 30 + idx * 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    
    # Display the image
    cv2.imshow('Pose Estimation', image)
    
    key = cv2.waitKey(10) & 0xFF
    if key == ord('q'):
        break
    elif key == 32:  # Space bar to pause/play
        paused = not paused
    elif key == 81:  # Left arrow to go back
        frame_number = max(0, frame_number - 100)  # Go back 10 frames
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    elif key == 83:  # Right arrow to go forward
        frame_number += 10  # Go forward 10 frames
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

cap.release()
cv2.destroyAllWindows()
