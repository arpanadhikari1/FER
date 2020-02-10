import numpy as np

import torch as t
import torch.nn as nn
import torch.nn.functional as f
import torchvision as tv

import matplotlib.pyplot as plt
import cv2

person=[]
classes=["Anger","Distgust","Fear","Happy","Neutral","Sad","Surprise"]

def predict(model,img_array):
    model.eval()
    transformation=tv.transforms.Compose([tv.transforms.ToPILImage(),
                                          tv.transforms.Grayscale(),
                                          tv.transforms.Resize((48,48)),
                                          tv.transforms.ToTensor(),
                                          tv.transforms.Normalize(mean=[0.5],std=[0.5])])

    img_tensor=t.stack([transformation(img) for img in img_array])
    input_feature=img_tensor
    
    predictions=[]
    for pred in model(input_feature).data.numpy().argmax(1):
        predictions.append(classes[pred])
        
    return np.array(predictions)

class Net(nn.Module):
    def __init__(self,num_class=7):
        super(Net,self).__init__()
        self.conv1=nn.Conv2d(in_channels=1,out_channels=8,kernel_size=3,padding=1)
        self.conv2=nn.Conv2d(in_channels=8,out_channels=40,kernel_size=3,padding=1)
#        self.conv3=nn.Conv2d(in_channels=40,out_channels=120,kernel_size=3,padding=1)
        self.pool=nn.MaxPool2d(kernel_size=2)
        self.drop=nn.Dropout2d(p=0.3)
        self.fc=nn.Linear(in_features=(12*12*40),out_features=num_class)

    def forward(self,x):
        x=f.relu(self.pool(self.conv1(x)))
        x=f.relu(self.pool(self.conv2(x)))
#        x=f.relu(self.pool(self.conv3(x)))
        x=f.dropout(self.drop(x),training=self.training)
        x=x.view(x.size()[0],-1)
        x=self.fc(x)
        return f.log_softmax(x,dim=1)

faceCascade=cv2.CascadeClassifier("models/haarcascade_frontalface_default.xml")
image=cv2.imread("Test_image.jpg")
#gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

faces=faceCascade.detectMultiScale(image,
                                   scaleFactor=1.2,
                                   minNeighbors=5,
#                                   minSize=(10,10),
                                   flags=cv2.CASCADE_SCALE_IMAGE)

for (x,y,w,h) in faces:
#    c=int((image[(y):y+(h),(x):x+(w)].shape[0])/10)
#    person.append(image[(y+c):y+(h-c),(x+c):x+(w-c)])
    person.append(image[y:y+h,x:x+w])
#    cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0)) 

cv2.imshow("face found",image)
cv2.waitKey(0)
cv2.destroyAllWindows()

model=Net(len(classes))
model.load_state_dict(t.load("models/Facial_Emotion_Recognizer.pt"))

predictions=predict(model,np.array(person))

population=len(person)
size=5*population
fig=plt.figure(figsize=(size,size))
for i in range(population):
    a=fig.add_subplot(population,1,(i+1))
    plt.imshow(person[i],cmap="gray")
    a.set_title(predictions[i])
