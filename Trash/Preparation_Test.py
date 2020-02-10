import os 
import shutil
import pandas as pd
import matplotlib .pyplot as plt
import numpy as np

img_file=pd.read_csv("data/test.csv")
images=np.array(img_file["Pixels"].apply(lambda x:np.reshape(([int(i) for i in (x.split())]),([48,48]))))

def make_dir(path="New Folder"):
    if os.path.exists(path):
        shutil.rmtree(path)        
    os.makedirs(path)
    return path

def prepare(dest,images):
    for i,image in enumerate(images):
        plt.imsave(os.path.join(dest,(str(i)+".jpg")),image)

dest_dir=make_dir("Face Datasets(Test)")
prepare(dest_dir,images)
