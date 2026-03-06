# Smart-Corn-Leaf-Disease-Detection-Using-Deep-learning---AgriScan
AI-based system for detecting corn leaf diseases using Deep Learning (CNN), Flask API, MongoDB, and Continuous Learning with user feedback.
Project Overview

Smart Corn Leaf Disease Detection is an AI-based system that automatically identifies diseases in corn leaves using Deep Learning. The system uses a Convolutional Neural Network (CNN) model to classify leaf images into different disease categories such as Common Rust, Northern Leaf Blight, Gray Leaf Spot, and Healthy leaves. The goal of the project is to help farmers detect plant diseases early and reduce crop loss.

How the System Works

The system works through the following steps:

User Authentication

Users register and log in using secure authentication with JWT tokens.

Image Upload

The user uploads a corn leaf image through the web interface.

Image Preprocessing

The image is resized and normalized to match the input format required by the deep learning model.

Disease Prediction

The processed image is passed to the trained CNN model.

The model analyzes patterns such as leaf spots, discoloration, and texture changes.

Prediction Result

The system returns the predicted disease class along with a confidence score.

History Storage

Each prediction is stored in a MongoDB database with timestamp and user information.

Continuous Learning

Users can provide feedback if a prediction is incorrect.

After collecting several incorrect predictions, the system automatically retrains the model to improve accuracy.

Key Features

Deep Learning-based disease classification

Real-time image prediction

User authentication with JWT

Prediction history tracking

Continuous learning through user feedback

Web interface for easy interaction

Technologies Used

Python

TensorFlow / Keras

Flask (REST API)

MongoDB
HTML / CSS / JavaScript

OpenCV & NumPy for image processing

Applications

Early detection of corn leaf diseases

Smart agriculture systems

Decision support tool for farmers
