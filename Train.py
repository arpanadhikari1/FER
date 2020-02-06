import os

import torch as t
import torch.nn.functional as f
import torch.nn as nn
import torchvision as tv

import matplotlib.pyplot as plt
import sklearn.metrics as m
import seaborn as sns

train_path="Face Datasets"
classes=sorted(os.listdir(train_path))

def load_datasets(path):
    transformation=tv.transforms.Compose([tv.transforms.RandomHorizontalFlip(),
                                          tv.transforms.Grayscale(),
                                          tv.transforms.ToTensor(),
                                          tv.transforms.Normalize(mean=[0.5],std=[0.5])])

    datasets=tv.datasets.ImageFolder(root=train_path,
                                     transform=transformation)
    data_size=len(datasets)
    train_size=int(0.7*data_size)
    test_size=data_size-train_size
    
    train_data,test_data=t.utils.data.random_split(datasets,[train_size,test_size])
    
    train_loader=t.utils.data.DataLoader(train_data)
    test_loader=t.utils.data.DataLoader(test_data)
    return train_loader,test_loader

train_loader,test_loader=load_datasets(train_path)

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

def train(model,device,train_loader,optimizer,loss_fn,epoch):
    model.train()
    train_loss=0
    print("Epoch: ",epoch,"\n\nTraining...")
    for batch_count,(data,label) in enumerate(train_loader):
        data,label=data.to(device),label.to(device)
        optimizer.zero_grad()#reset the optimezer
        output=model(data)#push the data to neural network
        loss=loss_fn(output,label)
        train_loss+=loss.item()
        loss.backward()
        optimizer.step()
#        print("Batch {}  Loss: {:0.6f}".format(batch_count,loss.item()))
    #    print("\noutput = ",output,"\nloss=",loss,"\ntrain_loss=",train_loss)
    avg_loss=train_loss/(batch_count+1)
    print("Avarage loss: {0:0.6f}".format(avg_loss))
    return avg_loss

def test(model,device,test_loader,loss_fn):
    model.eval()
    test_loss=0
    correct=0
    print("Validation...")
    
    with t.no_grad():       
        for batch_count,(data,label) in enumerate(test_loader):
            data,label=data.to(device),label.to(device)
            output=model(data)
            test_loss+=loss_fn(output,label).item()
            _,predicted=t.max(output,1)
            correct+=t.sum(label==predicted).item()
    
        avg_loss=test_loss/(batch_count+1)
        accuracy=(correct/(len(test_loader.dataset)))*100
        print("Avarage loss: {0:0.6f}".format(avg_loss),"\nAccuracy: {0:0.6f}%\n".format(accuracy))
    return avg_loss

device="cpu"
if(t.cuda.is_available()):
    device="cuda"
model=Net().to(device)

optimizer=t.optim.Adam(model.parameters(),
                       lr=0.001,
                       weight_decay=0.001)
loss_fn1=nn.CrossEntropyLoss()

epoch_num=[]
training_loss=[]
validation_loss=[]

print("Training on: ",device)
epochs=20
for epoch in range(1,epochs+1):
    train_loss=train(model,device,train_loader,optimizer,loss_fn1,epoch)
    test_loss=test(model,device,test_loader,loss_fn1)
    epoch_num.append(epoch)
    training_loss.append(train_loss)
    validation_loss.append(test_loss)
    
plt.plot(epoch_num,training_loss)
plt.plot(epoch_num,validation_loss)
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend(["training","validation"])
plt.show()

labels=[]
predictions=[]

for data,label in test_loader:
    for label in label.data.numpy():
        labels.append(label)
    for pred in model(data).data.numpy().argmax(1):
        predictions.append(pred)

cm=m.confusion_matrix(labels,predictions)
sns.heatmap(cm,
            annot=True,
            xticklabels=classes,
            yticklabels=classes,
            cmap="Blues")
plt.xlabel("Predicted Emotion")
plt.ylabel("True Emotion")

t.save(model.state_dict(),"models/Facial_Emotion_Recognizer.pt")

del model
