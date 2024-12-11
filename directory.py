import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import pandas as pd
import numpy as np
import os
import cv2
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Helper function to parse annotations
def parse_annotations(csv_file, img_dir):
    data = pd.read_csv(csv_file)
    images = []
    boxes = []
    labels = []
    for _, row in data.iterrows():
        img_path = os.path.join(img_dir, row['filename'])
        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            img = cv2.resize(img, (128, 128))  # Resize images to 128x128
            images.append(img)
            boxes.append([row['xmin'], row['ymin'], row['xmax'], row['ymax']])
            labels.append(row['class'])
    return np.array(images), np.array(boxes), np.array(labels)

# Load data
train_img_dir = "C:\\Users\\RIFA\\PycharmProjects\\trial_numberplate\\archive\\train"
train_csv = "C:\\Users\\RIFA\\PycharmProjects\\trial_numberplate\\archive\\train.csv"
val_img_dir = "C:\\Users\\RIFA\\PycharmProjects\\trial_numberplate\\archive\\valid"
val_csv = "C:\\Users\\RIFA\\PycharmProjects\\trial_numberplate\\archive\\valid.csv"

train_images, train_boxes, train_labels = parse_annotations(train_csv, train_img_dir)
val_images, val_boxes, val_labels = parse_annotations(val_csv, val_img_dir)

# Normalize images
train_images = train_images / 255.0
val_images = val_images / 255.0

# Convert labels to one-hot encoding
unique_classes = np.unique(train_labels)
class_to_idx = {cls: idx for idx, cls in enumerate(unique_classes)}
train_labels = np.array([class_to_idx[cls] for cls in train_labels])
val_labels = np.array([class_to_idx[cls] for cls in val_labels])
train_labels = tf.keras.utils.to_categorical(train_labels, len(unique_classes))
val_labels = tf.keras.utils.to_categorical(val_labels, len(unique_classes))

# Define the model
input_layer = Input(shape=(128, 128, 3))
x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2))(x)
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
output_layer = Dense(len(unique_classes), activation='softmax')(x)

model = Model(inputs=input_layer, outputs=output_layer)

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(train_images, train_labels,
                    validation_data=(val_images, val_labels),
                    epochs=10, batch_size=32)

# Plot accuracy and loss graphs
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()
