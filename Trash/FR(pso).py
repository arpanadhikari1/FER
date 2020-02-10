import os
import torch as t
import torch.nn.functional as f
import torch.nn as nn
import torchvision as tv
import matplotlib.pyplot as plt
import skimage as si
import sklearn.metrics as m
import numpy as np
import seaborn as sns
import pyswarms as ps

train_path="Face Datasets"
classes=sorted(os.listdir(train_path))

def load_datasets(path):
    transformation=tv.transforms.Compose([tv.transforms.RandomHorizontalFlip(),
                                          tv.transforms.ToTensor(),
                                          tv.transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])])

    #ImageFolder() load the images from each folder and tag the images with class name same as it's folder name
    datasets=tv.datasets.ImageFolder(root=train_path,transform=transformation)
    #print(datasets.classes)
    #plt.imshow(datasets[1][0][0],cmap="gray")
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
        self.conv1=nn.Conv2d(in_channels=3,out_channels=12,kernel_size=3,padding=1)
#        self.conv2=nn.Conv2d(in_channels=12,out_channels=24,kernel_size=3,padding=1)
        self.pool=nn.MaxPool2d(kernel_size=2)
        self.drop=nn.Dropout2d(p=0.3)
        self.fc=nn.Linear(in_features=(24*24*12),out_features=num_class)
    def forward(self,x):
        x=f.relu(self.pool(self.conv1(x)))
#        x=f.relu(self.pool(self.conv2(x)))
        x=f.dropout(self.drop(x),training=self.training)
        x=x.view(-1,(24*24*12))
        x=self.fc(x)
        return f.log_softmax(x,dim=1)
    
#model=nn.Linear(5,3)
#Transfer learning
#model=tv.models.resnet18(pretrained=True)
#for param in model.parameters():
##    print(param)
#    param.requires_grad=False
#    
#num_ftrs=model.fc.in_features
#model.fc=nn.Linear(num_ftrs,len(classes))
#print(model)
#for param in model.parameters():
#    print(param)

device="cpu"
if(t.cuda.is_available()):
    device="cuda"

model=Net().to(device)

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

optimizer=t.optim.Adam(model.parameters(),lr=0.001)
loss_fn=nn.CrossEntropyLoss()
#print([x for x in model.parameters()])
dimensions=0
for x in model.children():
    if (type(x)==nn.Conv2d or type(x)==nn.Linear):
        dimensions+=x.weight.data.nelement()

def init_weight(particles,model=model):
    i=0    
    for m in model.children():
        if type(m)==nn.Conv2d or type(m)==nn.Linear:
            weight=m.weight.data
            m.weight.data=t.tensor(np.reshape(particles[i:(i+weight.nelement())],list(weight.size()))).float()
            i=i+weight.nelement()
    return model

def forward_prop(model,device=device,train_loader=train_loader,optimizer=optimizer,loss_fn=loss_fn,epoch=10):
    loss=train(model,device,train_loader,optimizer,loss_fn,epoch)
    return loss
    
def func(x):
    n_particles=x.shape[0]
    losses=[forward_prop(init_weight(x[i])) for i in range(n_particles)]
    return losses
    
options={"c1":0.5,"c2":0.3,"w":0.5}
pso_optimizer=ps.single.GlobalBestPSO(n_particles=5,dimensions=dimensions,options=options)
cost,pos=pso_optimizer.optimize(func,iters=5)

model=init_weight(pos)

epoch_num=[]
training_loss=[]
validation_loss=[]

print("Training on: ",device)
epochs=20
for epoch in range(1,epochs+1):
    train_loss=train(model,device,train_loader,optimizer,loss_fn,epoch)
    test_loss=test(model,device,test_loader,loss_fn)
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
sns.heatmap(cm,annot=True,xticklabels=classes,yticklabels=classes,cmap="Blues")
plt.xlabel("Predicted Shape")
plt.ylabel("True Shape")

t.save(model.state_dict(),"FER.pt")

del model
