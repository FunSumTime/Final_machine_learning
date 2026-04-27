# Pneumonia Image Classification CNN

## Overview

This project was my final project for my Machine Learning class. I built and trained my own convolutional neural network to classify medical images and predict whether an image showed pneumonia or not.

The project focused on working with image data, preprocessing, training a CNN, testing different input sizes, and thinking through how class imbalance affects a model.

## Project Purpose

The goal of this project was to better understand how machine learning can be used with medical images. Since I am interested in both computer science and medicine, this project gave me a chance to connect those two areas in a real way.

Instead of only using a prebuilt model, I created my own CNN so I could better understand how the layers worked together and how image classification models learn patterns.

## Features

- Custom convolutional neural network
- Pneumonia vs. non-pneumonia image classification
- Image preprocessing
- Testing with different image sizes
- Training and evaluation on medical image data
- Class imbalance awareness
- Exploration of ensemble-style prediction ideas

## Dataset Challenge

One of the main challenges with the dataset was class imbalance. There were more positive pneumonia images than normal images, which meant the model could become biased toward predicting pneumonia more often.

Because of this, I had to think more carefully about:

- How the data was split
- How preprocessing affected the model
- How accuracy could be misleading
- How the model handled positive and normal cases

## What I Tested

During the project, I experimented with different ways to improve or better understand the model.

Some of the things I looked at included:

- Changing image sizes to see how input size affected training
- Preprocessing the image data before training
- Watching how the model performed with an imbalanced dataset
- Exploring the idea of using multiple neural networks together

One idea I looked into was similar to an ensemble model. The idea was to have multiple neural networks make predictions and then use the most common prediction as the final output. This is similar to how models like random forests combine multiple outputs to make a stronger final decision.

## What I Learned

This project helped me understand that machine learning is not just about building a model and training it. The data matters a lot.

I learned more about:

- CNN architecture
- Image preprocessing
- Medical image classification
- Class imbalance
- Model evaluation
- How input size can affect performance
- Why testing different approaches is important

This project also helped me see how machine learning could be used in the medical field, while also showing me that medical AI systems need to be tested carefully because mistakes can matter a lot.

## Future Improvements

Some future improvements I would like to add include:

- Better handling of class imbalance
- More evaluation metrics like precision, recall, and F1-score
- A confusion matrix to better understand model mistakes
- Data augmentation
- Transfer learning with pretrained models
- Grad-CAM or heatmap explanations
- Comparing multiple CNN architectures
- Building an ensemble of multiple models

## Short Description

A machine learning final project where I built and trained a custom CNN to classify pneumonia in medical images. The project focused on image preprocessing, class imbalance, input size testing, and exploring ensemble-style prediction using multiple neural networks.
