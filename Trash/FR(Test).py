import os
import torch as t
import torch.nn.functional as f
import torch.nn as nn
import torchvision as tv
import matplotlib.pyplot as plt
import sklearn.metrics as m
import numpy as np
import seaborn as sns

train_path="Face Datasets"
classes=sorted(os.listdir(train_path))
length=len(classes)
#fig=plt.figure(figsize=(20,20))

#for i in range(len(person)):
#    plt.imshow(person[i],cmap="gray")
#    plt.show()
       
def load_datasets(path):
    transformation=tv.transforms.Compose([tv.transforms.Scale(48,48),
                                          tv.transforms.ToTensor(),
                                          tv.transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])])

    datasets=tv.datasets.ImageFolder(root=train_path,transform=transformation)
    data_size=len(datasets)
    train_size=int(0.7*data_size)
    test_size=data_size-train_size
    
    train_data,test_data=t.utils.data.random_split(datasets,[train_size,test_size])
    
    train_loader=t.utils.data.DataLoader(train_data)
    test_loader=t.utils.data.DataLoader(test_data)
    return train_loader,test_loader

train_loader,test_loader=load_datasets(train_path)

person=[]
l=t.tensor([-1])
for batch_count,(data,label) in enumerate(train_loader):
    flag=t.sum(label==l).item()
    if(flag==0):
        l=t.cat((l,label))
        person.append(data)
    if(len(l)>length):
        break
    
class Net(nn.Module):
    def __init__(self,num_class=7):
        super(Net,self).__init__()
        self.conv1=nn.Conv2d(in_channels=3,out_channels=12,kernel_size=3,padding=1)
        self.conv2=nn.Conv2d(in_channels=12,out_channels=36,kernel_size=3,padding=1)
        self.conv3=nn.Conv2d(in_channels=36,out_channels=72,kernel_size=3,padding=1)
        self.pool=nn.MaxPool2d(kernel_size=2)
    def forward_once(self,x):
        x=f.relu(self.pool(self.conv1(x)))
        x=f.relu(self.pool(self.conv2(x)))
        x=f.relu(self.pool(self.conv3(x)))
        x=x.view(x.size()[0],-1)
        return x
    def forward(self,x1,x2,x3):
        output1=self.forward_once(x1)
        output2=self.forward_once(x2)
        output3=self.forward_once(x3)
        return output1,output2,output3
    

device="cpu"
if(t.cuda.is_available()):
    device="cuda"
model=Net().to(device)

def distance(vect1,vect2):
    dist=(vect1-vect2)**2
    return dist

class triplet_loss(t.autograd.Function):
    @staticmethod
    def forward(ctx,a,p,n):
        ctx.save_for_backward(a,p,n)
        loss=t.max(t.tensor([(t.mean(distance(a,p)-distance(a,n)+0.5)),0]))
        return loss
    
    @staticmethod
    def backward(ctx,grad_output):
        a,p,n=ctx.saved_tensors
        grad_input=t.neg((distance(a,p)-distance(a,n)+0.5)*lr)
        return grad_input,None,None
    
def predict(image):
    dist=[]
    for x in person:
        img1,img2,_=model(image,x,x)    
        dist.append(t.mean(distance(img1,img2)))
    prediction=np.argmin(dist)
    return prediction

lr=0.001
def train(model,device,train_loader,optimizer,loss_fn,epoch):
    model.train()
    train_loss=0
    print("Epoch: ",epoch,"\n\nTraining...")
    for batch_count,(data,label) in enumerate(train_loader):
        data,label=data.to(device),label.to(device)
        optimizer.zero_grad()#reset the optimezer
        anchor=data#push the data to neural network
        positive=person[(label.item())]
        negetive=person[((label.item()+np.random.randint(1,length))%length)]
        a,p,n=model(anchor,positive,negetive)    
        loss=loss_fn(a,p,n)
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
            anchor=data#push the data to neural network
            predicted=predict(anchor)
            correct+=t.sum(label==predicted).item()
    
        avg_loss=test_loss/(batch_count+1)
        accuracy=(correct/(len(test_loader.dataset)))*100
        print("Avarage loss: {0:0.6f}".format(avg_loss),"\nAccuracy: {0:0.6f}%\n".format(accuracy))
    
    return avg_loss

optimizer=t.optim.Adam(model.parameters(),lr=0.001)
loss_fn1=nn.CrossEntropyLoss()
loss_fn2=triplet_loss.apply
epoch_num=[]
training_loss=[]
validation_loss=[]

print("Training on: ",device)
epochs=5
for epoch in range(1,epochs+1):
    train_loss=train(model,device,train_loader,optimizer,loss_fn2,epoch)
    test_loss=test(model,device,test_loader,loss_fn2)
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
    for l in label:
        labels.append(l.item())
    for x in data:
        predictions.append(predict(data))

cm=m.confusion_matrix(labels,predictions)
sns.heatmap(cm,annot=True,xticklabels=classes,yticklabels=classes,cmap="Blues")
plt.xlabel("Predicted Shape")
plt.ylabel("True Shape")

t.save(model.state_dict(),"FR.pt")

del model
