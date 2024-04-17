from glob import glob
all_video_paths = glob("/data/agent_h/datasets_chunyu/clip_videos_v3/**/*.mp4", recursive=True)
print(len(all_video_paths))

import cv2
from retinaface import RetinaFace
import json

video_path = all_video_paths[0]
print(video_path)
# Load the sample video
cap = cv2.VideoCapture(video_path)

# Initialize RetinaFace

# Initialize a list to store the face landmarks
face_landmarks_list = []

# Loop through each frame of the video
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform face detection and landmark detection using RetinaFace
    landmarks = RetinaFace.detect_faces(frame)
    face_landmarks_list.append(landmarks)

# Release the video capture
cap.release()

# Save the face landmarks as a list of JSON
output_path = "/data/agent_h/sample_result.json"
with open(output_path, "w") as f:
    json.dump(face_landmarks_list, f)