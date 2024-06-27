import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.utils import plot_model

# Specify constants
SEQUENCE_LENGTH = 30
IMAGE_HEIGHT, IMAGE_WIDTH = 64, 64
DATASET_DIR = "../../../../../../media/jzier/Elements/DATASETS/final_dataset_1/"
CLASSES_LIST = ["assault", "normal", "shooting", "theft"]

# Create a directory to save the plots and confusion matrix if it doesn't exist
output_directory = '../weights_and_results/baseline/'
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Define function to extract frames from video
def frames_extraction(video_path):
    print("Extracting frames from:", video_path)
    frames_list = []
    video_reader = cv2.VideoCapture(video_path)
    for _ in range(SEQUENCE_LENGTH):
        success, frame = video_reader.read()
        if not success:
            break
        resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))
        normalized_frame = resized_frame / 255
        frames_list.append(normalized_frame)
    video_reader.release()
    return frames_list

def create_dataset():
    features = []
    labels = []
    for class_index, class_name in enumerate(CLASSES_LIST):
        files_list = os.listdir(os.path.join(DATASET_DIR, class_name))
        for file_name in files_list:
            video_file_path = os.path.join(DATASET_DIR, class_name, file_name)
            frames = frames_extraction(video_file_path)
            if len(frames) == SEQUENCE_LENGTH:
                features.append(frames)
                labels.append(class_index)
    features = np.asarray(features)
    labels = np.array(labels)
    return features, labels

# Define LRCN model
def create_LRCN_model():
    model = Sequential()

    model.add(TimeDistributed(Conv2D(32, (3, 3), padding='same', activation='relu'),
                              input_shape=(SEQUENCE_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH, 3)))
    model.add(TimeDistributed(MaxPooling2D((4, 4))))

    model.add(TimeDistributed(Conv2D(64, (3, 3), padding='same', activation='relu')))
    model.add(TimeDistributed(MaxPooling2D((4, 4))))

    model.add(TimeDistributed(Conv2D(128, (3, 3), padding='same', activation='relu')))
    model.add(TimeDistributed(MaxPooling2D((2, 2))))

    model.add(TimeDistributed(Conv2D(256, (2, 2), padding='same', activation='relu')))
    model.add(TimeDistributed(MaxPooling2D((2, 2))))

    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(32))
    model.add(Dense(len(CLASSES_LIST), activation='softmax'))
    
    return model

# Create dataset
features, labels = create_dataset()

# One-hot encode the target labels for each sequence
one_hot_encoded_labels = to_categorical(labels, num_classes=len(CLASSES_LIST))

# Split the dataset
features_train, features_test, labels_train, labels_test = train_test_split(features, one_hot_encoded_labels,
                                                                            test_size=0.25, shuffle=True)

# Construct the LRCN model
lrcn_model = create_LRCN_model()

# Compile the LRCN model with Adam optimizer
lrcn_model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=["accuracy"])

# Define callbacks
early_stopping_callback = EarlyStopping(monitor='accuracy', patience=10, mode='max', restore_best_weights=True)

# Train the LRCN model
lrcn_training_history = lrcn_model.fit(x=features_train, y=labels_train, epochs=70, batch_size=4,
                                       shuffle=True, validation_split=0.25, callbacks=early_stopping_callback)

# Evaluate the LRCN model
lrcn_evaluation_metrics = lrcn_model.evaluate(features_test, labels_test)
lrcn_evaluation_loss, lrcn_evaluation_accuracy = lrcn_evaluation_metrics
print("LRCN Test Loss:", lrcn_evaluation_loss)
print("LRCN Test Accuracy:", lrcn_evaluation_accuracy)

# Save the trained LRCN model in the native Keras format
lrcn_model.save(os.path.join(output_directory, 'LRCN_model.h5'))

# Save the accuracy, loss, hyperparameters, and training details
with open(os.path.join(output_directory, 'model_metrics.txt'), 'w') as metrics_file:
    metrics_file.write("Techniques Used:\n")
    metrics_file.write("- LRCN (Long Short-Term Memory Convolutional Neural Network)\n")
    metrics_file.write("- Data Preprocessing: Frame Extraction, Normalization\n")
    metrics_file.write("- Training with Adam Optimizer, Early Stopping, Reduce LR on Plateau\n")
    metrics_file.write("Model Structure:\n")
    lrcn_model.summary(print_fn=lambda x: metrics_file.write(x + '\n'))
    metrics_file.write("\nLRCN Model Metrics:\n")
    metrics_file.write(f"Test Loss: {lrcn_evaluation_loss}\n")
    metrics_file.write(f"Test Accuracy: {lrcn_evaluation_accuracy}\n")
    metrics_file.write("Hyperparameters:\n")
    metrics_file.write(f"Sequence Length: {SEQUENCE_LENGTH}\n")
    metrics_file.write(f"Image Height: {IMAGE_HEIGHT}\n")
    metrics_file.write(f"Image Width: {IMAGE_WIDTH}\n")
    metrics_file.write(f"Classes List: {', '.join(CLASSES_LIST)}\n")
    metrics_file.write("\nTraining Details:\n")
    metrics_file.write(f"Epochs: 70\n")
    metrics_file.write(f"Batch Size: 4\n")
    metrics_file.write(f"Learning Rate: Adam default\n")
    metrics_file.write(f"Optimization Tool: Adam\n")

# Save the training history for later analysis if needed
with open(os.path.join(output_directory, 'lrcn_training_history.txt'), 'w') as history_file:
    history_file.write(str(lrcn_training_history.history))

# Plot training and validation loss
plt.plot(lrcn_training_history.history['loss'], label='Train Loss')
plt.plot(lrcn_training_history.history['val_loss'], label='Validation Loss')
plt.title('LRCN Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig(os.path.join(output_directory, 'lrcn_training_loss.png'))
plt.clf()

# Plot training and validation accuracy
plt.plot(lrcn_training_history.history['accuracy'], label='Train Accuracy')
plt.plot(lrcn_training_history.history['val_accuracy'], label='Validation Accuracy')
plt.title('LRCN Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig(os.path.join(output_directory, 'lrcn_training_accuracy.png'))

# Plot the structure of the constructed LRCN model
plot_model(lrcn_model, to_file=os.path.join(output_directory, 'LRCN_model_structure_plot.png'), 
           show_shapes=True, show_layer_names=True)

# Predict classes for test set
predictions = lrcn_model.predict(features_test)
predicted_labels = np.argmax(predictions, axis=1)
true_labels = np.argmax(labels_test, axis=1)

# Confusion matrix
conf_matrix = confusion_matrix(true_labels, predicted_labels)
plt.figure(figsize=(8, 6))
heatmap = sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='g')
heatmap.set_xticklabels(CLASSES_LIST, rotation=45)
heatmap.set_yticklabels(CLASSES_LIST, rotation=0)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.savefig(os.path.join(output_directory, 'confusion_matrix.png'))
plt.show()

# Classification report
classification_report_str = classification_report(true_labels, predicted_labels, target_names=CLASSES_LIST)
with open(os.path.join(output_directory, 'classification_report.txt'), 'w') as report_file:
    report_file.write("Classification Report:\n")
    report_file.write(classification_report_str)

print("Plots, Confusion Matrix, Metrics, Hyperparameters, Model Structure, and Training Details saved to directory:", output_directory)

