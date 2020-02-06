import os 
import shutil

import pandas as pd
import numpy as np

import matplotlib .pyplot as plt

img_file=pd.read_csv("data/train.csv")
images=np.array(img_file["Pixels"].apply(lambda x:np.reshape(([int(i) for i in (x.split())]),([48,48]))))

classes={0:"anger",1:"distgust",2:"fear",3:"happy",4:"sad",5:"surprise",6:"neutral"}
labels=img_file["Emotion"]

def make_dir(path="New Folder"):
    if os.path.exists(path):
        shutil.rmtree(path)        
    os.makedirs(path)
    return path

def prepare(dest,images,labels,classes):
    for label in labels.unique():
        print("Directory ",classes[label]," created")
        path=make_dir(os.path.join(dest,classes[label]))
        for i,image in enumerate(images[labels.loc[labels==label].index]):
            plt.imsave(os.path.join(path,(str(i)+".jpg")),image)

dest_dir=make_dir("Face Datasets")
prepare(dest_dir,images,labels,classes)
