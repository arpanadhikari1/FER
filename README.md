# Facial-Emotion-Recognition
## Introduction
Facial emotion recognition is the process of detecting human emotions from facial expressions. 
AI can detect emotions by learning what each facial expression means and applying that knowledge to the new information presented to it. 
### Objective of this project 
- Detecting human faces from an image
- Building an Artificial Intelligence model for automatically recognizing facial emotion 

## Files Description
### train.csv
This file contains the raw training data in tabular format consisting of the following fields 
- input: 48x48 pixel gray values (between 0 and 255)
- target: emotion category (beween 0 and 6: anger=0, disgust=1, fear=2, happy=3, sad=4, surprise=5, neutral=6)
### Preparation.py 
This file contains the code to convert the raw training data(train csv) into images in "jpg" format and storing the data in an organized way that is sorting images by their emotion category, storing images of different categories in different folders according to emotion category inside a folder "Train data".
### Train.py
This file contains the code to build the
