# Object_detection_with_COCO_labels
TensorFlow-based object detection model for traffic datasets with custom data preprocessing and training pipeline./

This repository contains a deep learning model for object detection applied to traffic-related datasets. The model is built using TensorFlow and Keras, and it processes images and label data to detect objects such as vehicles, pedestrians, traffic signs, and more. The data is preprocessed to match the model's input requirements, and the network architecture consists of Convolutional Neural Networks (CNNs) with bounding box prediction.

**Features**/
*Custom Object Detection: Built for traffic-related datasets with object detection tasks.
*TensorFlow & Keras Integration: Utilizes TensorFlow for model training and Keras for ease of implementation.
*Data Preprocessing Pipeline: Loads images and corresponding label files in YOLO format and preprocesses them.
*Bounding Box Prediction: Predicts both class labels and bounding box coordinates for each detected object.
*Training on Custom Data: Compatible with custom traffic datasets in image and label file formats.

**Requirements**
Before running the code, ensure that you have the following libraries installed:

Python 3.7+
TensorFlow 2.x
OpenCV
Numpy
Pandas
Matplotlib

**Dataset**
The dataset should consist of:

Images: Located in the images/ folder, with subdirectories for train, val, and test data.
Labels: Located in the labels/ folder, with corresponding label files for the images. Each label file should be a text file containing class labels and bounding box coordinates in YOLO format.

**Model Architecture**
The model is a simple CNN-based architecture with two main outputs:

Class Output: Classifies objects in the image.
Bounding Box Output: Predicts the coordinates of the bounding box for the detected objects.

**Model Outputs**
After training, the model will output two things:

Class Prediction: A probability distribution over all object classes for each bounding box.
Bounding Box Coordinates: Predicted coordinates for the bounding boxes, which are normalized between 0 and 1 with respect to the image size.

**Acknowledgements**
TensorFlow: For the powerful deep learning framework.
OpenCV: For image manipulation.
Keras: For easy-to-use neural network layers and model management.
YOLO Format: For easy annotation and training of object detection models.
