# Object_detection_with_COCO_labels
TensorFlow-based object detection model for traffic datasets with custom data preprocessing and training pipeline.

This repository contains a TensorFlow-based implementation of an object detection model designed to identify objects in traffic scenarios. The project processes images and corresponding label files, trains a custom Convolutional Neural Network (CNN) to predict object classes and their bounding boxes, and visualizes the performance using accuracy and loss graphs.

Key features of this repository include:

Preprocessing of images and label files in YOLO format, converting bounding box coordinates to absolute values.
A robust pipeline to handle variable-length bounding box annotations without using padding.
A custom data generator to dynamically load and process images and labels for training and validation.
Implementation of a CNN model for multi-task learning with separate outputs for object classes and bounding boxes.
Visualization of training progress, including classification accuracy and bounding box loss.
Files Included:

Python scripts for data preprocessing, model training, and evaluation.
Example dataset structure and annotations.
Detailed comments and documentation for ease of understanding and customization.
Usage:
Clone the repository, place your dataset in the specified format, and execute the main script to preprocess the data and train the model.

Requirements:

TensorFlow/Keras
Pandas
NumPy
OpenCV
Matplotlib
This repository serves as a starting point for developing custom object detection models, with a focus on handling real-world challenges like variable-length annotations and dataset preprocessing.
