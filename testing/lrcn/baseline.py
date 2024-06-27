import os
import cv2
import numpy as np
import time
from glob import glob
from tensorflow.keras.models import load_model

# Load the saved LRCN model
lrcn_model = load_model('../../weights_and_results/baseline/LRCN_model.h5')

# Define constants
SEQUENCE_LENGTH = 30
IMAGE_HEIGHT, IMAGE_WIDTH = 64, 64
CLASSES_LIST = ["assault", "normal", "shooting", "theft"]

def predict_single_action_lrcn(roi_frames):
    roi_frames_resized = [cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH)) / 255.0 for frame in roi_frames]
    predicted_labels_probabilities = lrcn_model.predict(np.array([roi_frames_resized]))[0]
    predicted_label_index = np.argmax(predicted_labels_probabilities)
    predicted_class_name = CLASSES_LIST[predicted_label_index]
    return predicted_class_name, predicted_labels_probabilities[predicted_label_index]

# Define the path to the input video
video_path = '../../test_videos/input/dark11.mp4'

# Create a VideoCapture object to read from the video file
cap = cv2.VideoCapture(video_path)

# Get input video details
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

output_video_path = '../../test_videos/output/baseline/output2.mp4'

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

# Initialize variables
overall_action = "normal"
roi_frames_buffer = []

# Start measuring time
start_time = time.time()

# Process each frame of the input video
frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1
    print(f"Processing frame {frame_count}")

    roi_frames_buffer.append(frame)
    if len(roi_frames_buffer) == SEQUENCE_LENGTH:
        predicted_action, _ = predict_single_action_lrcn(roi_frames_buffer)
        overall_action = predicted_action
        roi_frames_buffer.pop(0)

        if overall_action != "normal":
            cv2.putText(frame, "Anomaly Detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(frame, f'Overall Action: {overall_action}', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Write the frame to the output video
    out.write(frame)

# Release VideoCapture and VideoWriter objects
cap.release()
out.release()

# End measuring time and print it
end_time = time.time()
processing_time = end_time - start_time
print(f"Total processing time: {processing_time:.2f} seconds")

# Compute the processing time for 1 frame
processing_time_per_frame = processing_time / frame_count
print(f"Processing time per frame: {processing_time_per_frame:.5f} seconds")

