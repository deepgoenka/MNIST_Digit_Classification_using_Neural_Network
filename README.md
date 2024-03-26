# MNIST Digit Classification using Neural Network

This project aims to classify handwritten digits from the MNIST dataset using a neural network built with TensorFlow and Keras. The MNIST dataset is a commonly used benchmark dataset in machine learning for digit recognition tasks.

## Table of Contents

- [Introduction](#introduction)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Saving the Model](#saving-the-model)

## Introduction

The project utilizes a neural network architecture to classify handwritten digits from the MNIST dataset. It preprocesses the data, builds and trains the model, evaluates its performance, and saves the trained model for future use.

## Getting Started

### Prerequisites

Ensure you have the following dependencies installed:

- Python (>=3.6)
- TensorFlow
- Keras
- Matplotlib
- Seaborn
- NumPy
- scikit-learn
- OpenCV (cv2)
- Google Colab (if using)

### Installation

You can install the required packages using pip:

```bash
pip install tensorflow keras matplotlib seaborn numpy scikit-learn opencv-python-headless
```

## Model Architecture

The neural network model architecture consists of:

- Input layer: Flattened input images (28x28 pixels)
- Hidden layers: Fully connected layers with ReLU activation
- Output layer: Softmax activation for multiclass classification (10 output classes)

## Training

The model is trained using the MNIST dataset, with 10 epochs and a batch size of 32. Training loss and accuracy are monitored during the training process.

## Evaluation

After training, the model's performance is evaluated using the test dataset. Classification report and confusion matrix are generated to assess the model's accuracy and performance.

## Results

The project achieves an accuracy of 98% on the test dataset.

## Saving the Model

The trained model is saved using joblib for future use. You can load the model and make predictions on new data.
