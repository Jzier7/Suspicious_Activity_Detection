from glob import glob
from PIL import Image
from matplotlib import pyplot as plt
import tensorflow as tf 
import os

import cv2
import numpy as np
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision import transforms
from tensorflow.keras.models import load_model

# Load pre-trained Faster R-CNN model
faster_rcnn_model = fasterrcnn_resnet50_fpn(pretrained=True)
faster_rcnn_model.eval()

# Load the saved LRCN model
lrcn_model = load_model('../weights_and_results/improved1/LRCN_model.h5')

MODEL_DICT = {
    "dr": "./MIRNet/lite/mirnet_dr.tflite",
    "fp16": "./MIRNet/lite/mirnet_fp16.tflite",
    "int8": "./MIRNet/lite/mirnet_int8.tflite"
}

# Define constants
SEQUENCE_LENGTH = 30
IMAGE_HEIGHT, IMAGE_WIDTH = 64, 64
INPUT_HEIGHT, INPUT_WIDTH = 360, 640
CLASSES_LIST = ["assault", "normal", "shooting", "theft"]
# Define transforms
transform = transforms.Compose([transforms.ToTensor()])

def infer_tflite(model_type, frame):
    # Convert frame to PIL Image
    image = Image.fromarray(frame)
    
    image = np.array(image)
    image = image.astype('float32') / 255.0
    image = np.expand_dims(image, axis=0)

    # Run inference
    interpreter = tf.lite.Interpreter(model_path=MODEL_DICT[model_type])
    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]
    
    interpreter.resize_tensor_input(0, [1, image.shape[1], image.shape[2], 3])
    interpreter.allocate_tensors()
    interpreter.set_tensor(input_index, image)
    interpreter.invoke()
    raw_prediction = interpreter.tensor(output_index)()
    
    # Initialize output variable
    output = None
    
    # Post-process output if inference was successful
    if raw_prediction is not None:
        output = raw_prediction.squeeze() * 255.0
        output = np.clip(output, 0, 255).astype(np.uint8)
 
    output_image = Image.fromarray(output)
    output_image = np.array(output_image)
    
    return output_image

# Define function to predict single action using LRCN model
def predict_single_action_lrcn(roi_frames):
    roi_frames_resized = [cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH)) / 255.0 for frame in roi_frames]
    predicted_labels_probabilities = lrcn_model.predict(np.array([roi_frames_resized]))[0]
    predicted_label_index = np.argmax(predicted_labels_probabilities)
    predicted_class_name = CLASSES_LIST[predicted_label_index]
    return predicted_class_name, predicted_labels_probabilities[predicted_label_index]

# Define function to predict single action using Faster R-CNN
def predict_single_action_faster_rcnn(frame):
    torch_image = transform(frame).unsqueeze(0)
    with torch.no_grad():
        predictions = faster_rcnn_model(torch_image)
    max_confidence = 0
    predicted_box = None
    for box, label, score in zip(predictions[0]['boxes'], predictions[0]['labels'], predictions[0]['scores']):
        if score > 0.5 and label.item() == 1 and score > max_confidence:
            max_confidence = score
            predicted_box = box.cpu().numpy().astype(int)
    return predicted_box, max_confidence 

# Define the path to the input video
video_path = '../test_videos/input/long_videos/dark/input.mp4'

# Create a VideoCapture object to read from the video file
cap = cv2.VideoCapture(video_path)

# Get input video details
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

output_video_path = '../test_videos/output/improved/ouput1.mp4'

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (2 * frame_width, frame_height))

# Initialize variables
overall_action = "normal"
anomaly_detected = False

# Process each frame of the input video with MIRNet
frame_count = 0
mirnet_processed_frames = []
original_frames = []  # List to store original frames
total_processing_times_per_frame = []  # List to store total processing time per frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1
    print(f"Processing frame {frame_count}")

    start_time = time.time()  # Record start time for MIRNet processing

    # Resize frame to 640x360
    frame_resized = cv2.resize(frame, (INPUT_WIDTH, INPUT_HEIGHT))
    
    # Enhance frame using MIRNet
    enhanced_frame = infer_tflite("int8", frame_resized)
    
    # Resize enhanced frame back to original size
    enhanced_frame_original_size = cv2.resize(enhanced_frame, (frame_width, frame_height))
    
    mirnet_processed_frames.append(enhanced_frame_original_size)
    original_frames.append(frame)

    end_time = time.time()  # Record end time for MIRNet processing
    mirnet_processing_time = end_time - start_time  # Calculate MIRNet processing time for current frame

    # Process each enhanced frame with Faster R-CNN and LRCN
    start_time = time.time()  # Record start time for Faster R-CNN processing
    faster_rcnn_processing_times = []  # List to store Faster R-CNN processing time for each region of interest
    lrcn_processing_times = []  # List to store LRCN processing time for each region of interest
    for enhanced_frame, original_frame in zip(mirnet_processed_frames, original_frames):    
        predicted_box, confidence = predict_single_action_faster_rcnn(enhanced_frame)
        
        if confidence > 0.5:
            x1, y1, x2, y2 = predicted_box
            roi_frames = [enhanced_frame[y1:y2, x1:x2] for _ in range(SEQUENCE_LENGTH)]

            start_time_rcnn = time.time()  # Record start time for LRCN processing
            predicted_action, _ = predict_single_action_lrcn(roi_frames)
            end_time_rcnn = time.time()  # Record end time for LRCN processing
            lrcn_processing_time = end_time_rcnn - start_time_rcnn  # Calculate LRCN processing time for current frame
            lrcn_processing_times.append(lrcn_processing_time)  # Store LRCN processing time for current frame

            overall_action = predicted_action
            cv2.rectangle(enhanced_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(enhanced_frame, f'{predicted_action}, Confidence: {confidence:.2f}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            anomaly_detected = True
        else:
            anomaly_detected = False

        if overall_action != "normal" and anomaly_detected:
            cv2.putText(enhanced_frame, "Anomaly Detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(enhanced_frame, f'Overall Action: {overall_action}', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        end_time_rcnn = time.time()  # Record end time for Faster R-CNN processing
        faster_rcnn_processing_time = end_time_rcnn - start_time  # Calculate Faster R-CNN processing time for current frame
        faster_rcnn_processing_times.append(faster_rcnn_processing_time)  # Store Faster R-CNN processing time for current frame

    end_time = time.time()  # Record end time for Faster R-CNN processing
    total_processing_time = (end_time - start_time) + mirnet_processing_time  # Calculate total processing time for current frame
    total_processing_times_per_frame.append(total_processing_time)  # Store total processing time for current frame

    print(f"Total processing time for frame {frame_count}: {total_processing_time} seconds")

# Calculate average processing time per frame for each component
average_total_processing_time_per_frame = sum(total_processing_times_per_frame) / len(total_processing_times_per_frame)
print(f"Average total processing time per frame: {average_total_processing_time_per_frame} seconds")

