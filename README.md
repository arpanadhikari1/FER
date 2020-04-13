# Facial Emotion Recognition using Deep Learning
## Introduction
Facial emotion recognition is the process of detecting human emotions from facial expressions. 
AI can detect emotions by learning what each facial expression means and applying that knowledge to the new information presented to it. 
### Project Objectives 
- Detecting all the human faces from an image
- Building an Artificial Intelligence model for automatically recognizing facial emotion of each person
## Files Description
### data/train.csv
This file contains the raw training data in tabular format consisting of the following fields 
- input: 48x48 pixel gray values (between 0 and 255)
- target: emotion category (beween 0 and 6: anger=0, disgust=1, fear=2, happy=3, sad=4, surprise=5, neutral=6) 
### Preparation.py 
This file contains the code to convert the raw training data(train csv) into images in "jpg" format and storing the data in an organized way that is sorting images by their emotion category, storing images of different categories in different directory according to emotion category inside a parent directory "Face Datasets".
### Face Datasets 
This directory contains the images in "jpg" formats created by executing the file "Preparation.py".
### Train.py
This file contains the code to train the neural network model using the prepared data stored in the directory "Face Datasets" and saving the model weights generated in the file "Facial_Emotion_Recognizer.pt" for later use. 
### models
This directory contains two files:
- **Facial_Emotion_Recognizer.pt**: model weights generated from executing Train.py.
- **haarcascade_frontalface_default.xml**: pre-built model for face detection. 
### Test.py
This file contains the code to detect all the faces from an image using the pre-built model "haarcascade_frontalface_default.xml" and predicting the emotion of each person using the newly build model "Facial_Emotion_Recognizer.pt".

## System Requirements
- **Language uaed:** Python 3.7
- **Tools used:** PyTorch, TorchVision, OpenCV, NumPy, Pandas, Matplotlib, Seaborn, Scikit Learn 
- **Platform used:** Spyder 3

## How to use
### Step 1 
Download all the files.
### Step 2
Save an image file("jpg" format) containing some human faces as "Test_image.jpg" in the same directory where the file "Test.py" is located.
### Step 3
Run the file "Test.py" in your compiler/IDE("Spyder 3" in my case), optionally you can run the file "Train.py" before executing "Test.py" if you want to retrain the model.

*Note: Before executing see that your system satisfies all the system requirements.*

## Acknowledgement 
The data and the model for face detection used in this project has been taken from https://www.kaggle.com/c/facial-keypoints-detector/download/4EFQ2wWv1JculvQOAyVD%2Fversions%2FXb8kwFAz90jTlAhRkFUo%2Ffiles%2Ftrain.csv and https://www.kaggle.com/lalitharajesh/haarcascades/download/czIhRt0JFYMiIYwsaJ0y%2Fversions%2FzcaJOIihNcjthWl9XKtb%2Ffiles%2Fhaarcascade_frontalface_default.xml?datasetVersionNumber=1 respectively.
