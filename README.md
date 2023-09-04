# Image Classification and Object Detection

This repository contains applications of state-of-the-art models for both image classification and object detection.

## Table of Contents
- [Image Classification and Object Detection](#image-classification-and-object-detection)
  - [Table of Contents](#table-of-contents)
  - [Image Classification](#image-classification)
    - [ResNet-50](#resnet-50)
    - [VGG16](#vgg16)
  - [Object Detection](#object-detection)
    - [YOLOv8](#yolov8)
    - [Mask R-CNN](#mask-r-cnn)
  - [Usage](#usage)

## Image Classification
Image classification is the process of categorizing an image into predefined classes or labels, such as distinguishing between a cat and a dog based on visual content. It is one of the most common computer vision tasks and has many applications, including face recognition, content-based image retrieval, and medical imaging.

### ResNet-50

ResNet-50 is a powerful convolutional neural network (CNN) architecture that has shown remarkable performance in image classification tasks. It is a 50-layer deep neural network known for its residual connections, which help mitigate the vanishing gradient problem and enable training of very deep networks.

### VGG16

VGG16 is another popular CNN architecture known for its simplicity and effectiveness in image classification. It consists of 16 layers, including convolutional and fully connected layers, and has been widely used in various computer vision tasks.

<p align="center"><img src='Assets/image_classification1.png'></p>
<p align="center"><img src='Assets/image_classification2.png'></p>


## Object Detection
Object detection is the computer vision task of identifying and localizing multiple objects within an image by drawing bounding boxes around each recognized object, enabling precise object recognition and positioning. It is a more challenging task than image classification and has many applications, including self-driving cars, video surveillance, and medical imaging.

### YOLOv8

YOLOv8, short for "You Only Look Once version 8," is a state-of-the-art real-time object detection system. It is known for its speed and accuracy in detecting objects in images and videos. YOLOv8 combines the strengths of previous YOLO versions and leverages modern deep learning techniques.

<p align="center"><img src='Assets/yolov8_output.png'></p>

### Mask R-CNN

Mask R-CNN is a deep learning model for object detection and instance segmentation. It not only identifies objects in an image but also provides pixel-level segmentation masks for each object. This makes it suitable for tasks that require precise object localization.

<p align="center"><img src='Assets/maskrcnn_output.png'></p>

## Usage
To run the code, first clone the repository:
```bash
git clone https://github.com/mbar0075/Image-Classification-and-Object-Detection.git
```

Then, create the following environments:
```bash
conda env create -f yolov8.yml
conda env create -f maskrcnn.yml
```

Finally, activate the environments depending on the required application:
```bash
conda activate yolov8
conda activate maskrcnn
```