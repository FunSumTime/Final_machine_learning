# Final_machine_learning


## Overview
This repository contains the source code for an image classification final project developed at Utah Tech University. The core of the project is a custom-built Convolutional Neural Network (CNN) designed from scratch to classify images. 

A significant focus of this project was placed on the data pipeline, preprocessing, and empirical testing to optimize model performance and address inherent dataset biases. The project also explores the viability of ensemble learning methods for this specific classification task.

## Key Experiments & Features
* **Custom CNN Architecture:** Designed and trained a proprietary convolutional neural network specifically for this classification task rather than relying solely on pre-trained models.
* **Mitigating Data Imbalance:** Identified a severe bias in the training data due to class imbalance. This was successfully resolved by computing and applying custom class weights to the loss function during training, penalizing the model for misclassifying underrepresented classes.
* **Data Augmentation (Rotation):** Implemented image tilting/rotation within the preprocessing pipeline to artificially expand the training dataset and improve the model's ability to generalize to new orientations.
* **Resolution Optimization:** Conducted systematic tests on various image input sizes. Counter-intuitively, the experiments revealed that downscaling (shrinking) the input images yielded a higher classification accuracy and better overall model performance.
* **Ensemble Model Exploration:** Investigated and evaluated ensemble modeling techniques to determine if combining multiple models would yield a higher predictive accuracy than the standalone custom CNN.

## Tech Stack
* **Language:** Python 3.x
* **Deep Learning Framework:** [TensorFlow / Keras]
* **Data Processing:** NumPy, Pandas, Scikit-learn


## Results & Findings
* **Image Sizing:** Shrinking the image dimensions effectively reduced noise and overfitting, allowing the CNN to focus on the primary features of the classes, leading to better validation scores.
* **Class Weights:** Applying class weights proved essential for this dataset, completely stabilizing the model's bias towards the majority class and improving recall for minority classes.

## Author
* **Austin Espinoza** - *Computer Science Senior, Utah Tech University*


Classification Model on Xray images
linke to dataset https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
