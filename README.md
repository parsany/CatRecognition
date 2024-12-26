
# Cat Emotion Classifier

![plot image](./assets/plot.png)


Training a convolutional neural network using residual layers and Adam optimizer to classify images of cats into different emotion categories.


## Table of Contents
- [Cat Emotion Classifier](#cat-emotion-classifier)
  - [Table of Contents](#table-of-contents)
  - [Project Overview](#project-overview)
  - [Running](#running)
  - [Dataset](#dataset)
  - [Training](#training)
  - [Evaluation](#evaluation)

## Project Overview

This repository contains code for training a CNN model on a dataset of images of cats, classifying them into various emotional states. The model is trained using the `torchvision` and `torch` libraries, with data augmentation applied during training. After training, the model’s performance is evaluated using a confusion matrix and various performance metrics (loss, accuracy).

## Running

1. Clone the repository to your local machine:

    ```bash
    git clone https://github.com/vvsparsa/cat-emotion-classifier.git
    cd cat-emotion-classifier
    ```

2. Ensure you have PyTorch installed with CUDA support if you plan to train the model using a GPU. You can follow the official installation guide [here](https://pytorch.org/get-started/locally/).

## Dataset

https://universe.roboflow.com/cats-xofvm/cat-emotions


## Training

The model is trained using the following hyperparameters:

- **Epochs**: 37
- **Learning Rate**: 0.001
- **Batch Size**: 32
- **Dropout Rate**: 0.5

The optimizer used is Adam, and the loss function is Cross-Entropy Loss.

## Evaluation

![plot image](./assets/cmatrix.png)

The model's performance is evaluated on the validation dataset, and metrics such as loss and accuracy are reported. Additionally, confusion matrices are used to visually represent the classification performance.