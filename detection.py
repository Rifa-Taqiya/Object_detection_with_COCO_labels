import tensorflow as tf
import tensorflow_hub as hub
import imageio
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# COCO label names
coco_labels = [
    "background", "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "street sign", "stop sign", "parking meter", "bench"
]

# Load the pre-trained model
model = tf.keras.models.load_model('path_to_saved_model')
print("Custom-trained model loaded successfully")

# Load video
reader = imageio.get_reader("C:\\Users\\RIFA\\Downloads\\12684958_2160_3840_30fps.mp4")

# Function to preprocess the frame
def preprocess_frame(frame):
    frame = tf.image.convert_image_dtype(frame, tf.uint8)[tf.newaxis, ...]
    return frame

# Function to detect objects
def detect_objects(model, frame):
    processed_frame = preprocess_frame(frame)
    result = model(processed_frame)
    return {key: value.numpy() for key, value in result.items()}

# Function to draw bounding boxes
def draw_bounding_boxes(frame, result):
    scores = np.array(result['detection_scores'])[0]
    for i in range(len(scores)):  # Process each detected object
        score = float(scores[i])  # Access each score directly
        if score > 0.6:  # Only process high-confidence detections
            box = result['detection_boxes'][0][i]
            label_index = int(result['detection_classes'][0][i])
            label = coco_labels[label_index] if label_index < len(coco_labels) else "Unknown"

            y1, x1, y2, x2 = box
            h, w, _ = frame.shape
            x1, x2, y1, y2 = int(x1 * w), int(x2 * w), int(y1 * h), int(y2 * h)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            text = f"{label} ({score:.2f})"
            cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


# Directory to save frames
output_dir = 'output_frames'
os.makedirs(output_dir, exist_ok=True)

frame_count = 0

# Process video frames
for frame in reader:
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    result = detect_objects(model, frame)
    draw_bounding_boxes(frame, result)

    # Save frame to disk
    output_path = os.path.join(output_dir, f'frame_{frame_count:04d}.jpg')
    cv2.imwrite(output_path, frame)
    frame_count += 1

# Parameters
frame_rate = 20  # Frames per second
output_file = 'output_video.mp4'
image_folder = 'C:\\Users\\RIFA\\PycharmProjects\\trial_numberplate\\.venv\\Scripts\\output_frames'

# Get all image filenames in the folder
images = sorted([img for img in os.listdir(image_folder) if img.endswith(('.png', '.jpg', '.jpeg'))])

# Read the first image to get the dimensions
first_frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = first_frame.shape

# Define the video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec
video = cv2.VideoWriter(output_file, fourcc, frame_rate, (width, height))

# Add images to the video
for image in images:
    frame = cv2.imread(os.path.join(image_folder, image))
    video.write(frame)

video.release()
print("Video saved as", output_file)

os.startfile(output_file)
