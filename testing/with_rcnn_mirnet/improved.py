import os
import cv2
from glob import glob
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from PIL import Image
import tensorflow as tf
import numpy as np
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision import transforms
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import seaborn as sns

# Load pre-trained Faster R-CNN model
faster_rcnn_model = fasterrcnn_resnet50_fpn(pretrained=True)
faster_rcnn_model.eval()

# Load the saved LRCN model
lrcn_model = load_model('../../weights_and_results/test10/LRCN_model.h5')

MODEL_DICT = {
    "dr": "../../simulator/MIRNet/lite/mirnet_dr.tflite",
    "fp16": "../../simulator/MIRNet/lite/mirnet_fp16.tflite",
    "int8": "../../simulator/MIRNet/lite/mirnet_int8.tflite"
}

# Define constants
SEQUENCE_LENGTH = 30
IMAGE_HEIGHT, IMAGE_WIDTH = 64, 64
INPUT_HEIGHT, INPUT_WIDTH = 360, 640
CLASSES_LIST = ["assault", "normal", "shooting", "theft"]

# Define transforms
transform = transforms.Compose([transforms.ToTensor()])

def infer_tflite(model_type, frame):
    image = Image.fromarray(frame)
    image = np.array(image)
    image = image.astype('float32') / 255.0
    image = np.expand_dims(image, axis=0)

    interpreter = tf.lite.Interpreter(model_path=MODEL_DICT[model_type])
    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]
    
    interpreter.resize_tensor_input(0, [1, image.shape[1], image.shape[2], 3])
    interpreter.allocate_tensors()
    interpreter.set_tensor(input_index, image)
    interpreter.invoke()
    raw_prediction = interpreter.tensor(output_index)()
    
    output = None
    if raw_prediction is not None:
        output = raw_prediction.squeeze() * 255.0
        output = np.clip(output, 0, 255).astype(np.uint8)
 
    output_image = Image.fromarray(output)
    output_image = np.array(output_image)
    
    return output_image

def predict_single_action_lrcn(roi_frames):
    roi_frames_resized = [cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH)) / 255.0 for frame in roi_frames]
    predicted_labels_probabilities = lrcn_model.predict(np.array([roi_frames_resized]))[0]
    predicted_label_index = np.argmax(predicted_labels_probabilities)
    predicted_class_name = CLASSES_LIST[predicted_label_index]
    return predicted_class_name, predicted_labels_probabilities[predicted_label_index]

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

# Paths to the directory containing class-wise folders
videos_dir = '../../../../../../../media/jzier/Elements/DATASETS/final_dataset/'

# Output directories
output_video_dir = '../../test_videos/output/improved_with_rcnn_mirnet/'
metrics_output_dir = './output'
os.makedirs(output_video_dir, exist_ok=True)
os.makedirs(metrics_output_dir, exist_ok=True)

ground_truths = []
predictions = []

# Process each class folder
for class_name in CLASSES_LIST:
    video_folder = os.path.join(videos_dir, class_name)
    video_files = glob(os.path.join(video_folder, '*.mp4'))
    
    for video_path in video_files:
        cap = cv2.VideoCapture(video_path)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        # Define the codec and create VideoWriter object for each video
        video_name = os.path.basename(video_path)
        output_video_path = os.path.join(output_video_dir, video_name)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (2 * frame_width, frame_height))

        frame_count = 0
        mirnet_processed_frames = []
        original_frames = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            print(f"Processing frame {frame_count} of class {class_name}")

            frame_resized = cv2.resize(frame, (INPUT_WIDTH, INPUT_HEIGHT))
            enhanced_frame = infer_tflite("int8", frame_resized)
            enhanced_frame_original_size = cv2.resize(enhanced_frame, (frame_width, frame_height))
            
            mirnet_processed_frames.append(enhanced_frame_original_size)
            original_frames.append(frame)

        cap.release()

        for enhanced_frame, original_frame in zip(mirnet_processed_frames, original_frames):
            predicted_box, confidence = predict_single_action_faster_rcnn(enhanced_frame)
            if confidence > 0.5:
                x1, y1, x2, y2 = predicted_box
                roi_frames = [enhanced_frame[y1:y2, x1:x2] for _ in range(SEQUENCE_LENGTH)]
                predicted_action, _ = predict_single_action_lrcn(roi_frames)
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
            
            concatenated_frame = np.hstack((original_frame, enhanced_frame))
            out.write(concatenated_frame)

            # Append ground truth and prediction
            ground_truths.append(class_name)
            predictions.append(overall_action)

        out.release()

# Calculate accuracy
accuracy = accuracy_score(ground_truths, predictions)
print(f"Accuracy: {accuracy}")

# Save accuracy to file
with open(os.path.join(metrics_output_dir, 'accuracy.txt'), 'w') as f:
    f.write(f"Accuracy: {accuracy}\n")

# Detailed classification report
report = classification_report(ground_truths, predictions, target_names=CLASSES_LIST)
print(report)

# Save classification report to file
with open(os.path.join(metrics_output_dir, 'classification_report.txt'), 'w') as f:
    f.write(report)

# Confusion matrix
conf_matrix = confusion_matrix(ground_truths, predictions)
print("Confusion Matrix:")
print(conf_matrix)

# Save confusion matrix as an image
plt.figure(figsize=(8, 6))
heatmap = sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='g')
heatmap.set_xticklabels(CLASSES_LIST, rotation=45)
heatmap.set_yticklabels(CLASSES_LIST, rotation=0)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.savefig(os.path.join(metrics_output_dir, 'confusion_matrix.png'))
plt.close()

