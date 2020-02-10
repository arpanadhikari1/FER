# Facial Emotion Recognition using Deep Learning
## Introduction
Facial emotion recognition is the process of detecting human emotions from facial expressions. 
AI can detect emotions by learning what each facial expression means and applying that knowledge to the new information presented to it. 
### Project Objectives 
- Detecting human faces from an image
- Building an Artificial Intelligence model for automatically recognizing facial emotion 
## Files Description
### data/train.csv
This file contains the raw training data in tabular format consisting of the following fields 
- input: 48x48 pixel gray values (between 0 and 255)
- target: emotion category (beween 0 and 6: anger=0, disgust=1, fear=2, happy=3, sad=4, surprise=5, neutral=6) 
### Preparation.py 
This file contains the code to convert the raw training data(train csv) into images in "jpg" format and storing the data in an organized way that is sorting images by their emotion category, storing images of different categories in different folders according to emotion category inside a folder "Train data".
### Face Datasets 
This directory contains the images in "jpg" formats created by executing the file "Preparation.py".
### Train.py
This file contains the code to train the neural network model with the prepared data and saving the model weights generated for later used. 
### models
This directory contains two files:
- Facial_Emotion_Recognizer.pt: contains model weights generated from executing Train.py.
- haarscacade_frontalface_default.pt: contains the model for face detection. 
### Test.py
This file contains the code to detect all the faces from an image and predicting the emotion of each person using the newly build model.
## System Requirements
- **Language uaed:** Python 3.7
- **Tools used:** PyTorch, Torchvision, OpenCV, NumPy, Pandas, Matplotlib, Seaborn, Scikit Learn 
- **Platform used:** Spyder 3
## Acknowledgement 
The data and the model for face detection used in this project has been taken from https://www.kaggle.com/c/facial-keypoints-detector/data and https://www.kaggle.com/lalitharajesh/haarcascades/download/czIhRt0JFYMiIYwsaJ0y%2Fversions%2FzcaJOIihNcjthWl9XKtb%2Ffiles%2Fhaarcascade_eye.xml?datasetVersionNumber=1 respectively.
