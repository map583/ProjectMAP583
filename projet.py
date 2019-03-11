#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 23:01:54 2019

@author: tete
"""

import torch
import numpy as np
%load_ext autoreload
%autoreload 2
from utils import load_data,default_loader,trainset,imshow,normal,layershow
from torch.utils.data import Dataset, DataLoader
import torchvision
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import time
import matplotlib.pyplot as plt
from torchsummary import summary
#%%
datapath = "/home/tete/data/Youtube"
data,paths,labels = load_data(datapath)
available_field = ['url','videoname','locs','frameids','label_names','crop','scale','origRes','isYouTubeSubset']

#shuffle the dataset
num_data=len(paths)
index=np.arange(num_data)
np.random.shuffle(index)
paths=[paths[i] for i in index]
labels=[labels[i] for i in index]

#how many data we will put into each set
train_num,val_num, test_num  = int(0.10*num_data),int(num_data*0.70),int(num_data*0.2)

assert(train_num+val_num+test_num<=num_data)


train_path,train_label=paths[0:train_num],labels[0:train_num]
val_path,  val_label  =paths[train_num:train_num+val_num],labels[train_num:train_num+val_num]
test_path, test_label =paths[train_num+val_num:],labels[train_num+val_num:]


train_data  = trainset(train_path,train_label)
trainloader = DataLoader(train_data, batch_size=4,shuffle=True)

val_data  = trainset(val_path,val_label)
valloader = DataLoader(val_data, batch_size=4,shuffle=True)

test_data  = trainset(test_path,test_label)
testloader = DataLoader(test_data, batch_size=4,shuffle=True)

print("number of data in the training set    :", len(train_data))
print("number of data in the validation set  :", len(val_data))
print("number of data in the test set        :", len(test_data))
#%%
count = 1
for data in testloader:
    if count == 1:
        inputs_try,labels_try = data
    count +=1
print(count,"batches")   
print(inputs_try.shape)
    
#%%
out = torchvision.utils.make_grid(inputs_try,padding=0)
imshow(out,labels_try)   
#%%
#t1=time.time()
#for i in range(100):
#    heat_map=p2heatmap(320,180,labels_try,sigma=0.1)    
#t2=time.time()
#print(t2-t1)
#%%
#layershow(heat_map, [1,2,3], title=None)
#%%
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3,stride=1,padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3,stride=1,padding=1)
        self.conv3 = nn.Conv2d(64, 7, kernel_size=3,stride=1,padding=1)
        
        self.softmax =nn.Softmax2d()
        if use_gpu:
            self.conv1=self.conv1.cuda()
            self.conv2=self.conv2.cuda()
            self.conv3=self.conv3.cuda()
            self.softmax=self.softmax.cuda()
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        #print(x.shape)
        return self.softmax(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # PyTorch v0.4.0
model = Net().to(device)

summary(model, (3, 180, 320))
#%%

def train(model,data_loader,loss_fn,optimizer,n_epochs=10):
    if use_gpu:
        model=model.to("cuda")
        
    model.train(True)
    
    loss_train = np.zeros(n_epochs)
    acc_train = np.zeros(n_epochs)
    optimizer = optimizer
    
    for epoch_num in range(n_epochs):
        running_corrects = 0.0
        running_loss = 0.0
        size = 0
        for data in data_loader:
            inputs, labels = data
            bs = labels.size(0)
            
            if use_gpu:
                inputs=inputs.cuda()
                labels=labels.cuda()
            outputs = model(inputs)
            #print(outputs.shape, labels.shape)
            loss = loss_fn(outputs,labels) 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            #running_loss+=loss.data.item()
            #_,preds = torch.max(outputs.data,1)
            preds=outputs.data
            # statistics
            running_loss += loss.data.item()
            #print(preds.shape,labels.shape)
            #running_corrects += torch.sum(preds == labels)
            
            if(size%int(len(data_loader)/3)==0 and size!=0):
                print(running_loss / size, size)
            size += bs
              
            
        epoch_loss = running_loss / size
        #epoch_acc = running_corrects.item() / size
        loss_train[epoch_num] = epoch_loss
        #acc_train[epoch_num] = epoch_acc
        print('Train - Loss: {:.4f}'.format(epoch_loss))
        
    return loss_train#, acc_train

#%%
use_gpu=True
pose_detection = Net()
# choose the appropriate loss
loss_fn = nn.MSELoss(reduction='sum')
#loss_fn = nn.MSELoss(reduction='sum')
learning_rate = 1e-6
# your SGD optimizer
#optimizer_cl = torch.optim.Adadelta(pose_detection.parameters(),lr = learning_rate)
optimizer_cl = torch.optim.SGD(pose_detection.parameters(),lr = learning_rate)
# and train for 10 epochs
l_t= train(pose_detection,trainloader,loss_fn,optimizer_cl,n_epochs = 10)


#%%
def test(model,loss_fn,optimizer,data_loader):
    model.train(False)
    if use_gpu:
        model=model.to("cuda")
    running_corrects = 0.0
    running_loss = 0.0
    size = 0

    for data in data_loader:
        inputs, labels = data    
        bs = labels.size(0)
        
        if use_gpu:
            inputs=inputs.cuda()
            labels=labels.cuda()
        
        outputs = model(inputs)
        loss = loss_fn(outputs,labels) 

        #running_loss+=loss.data.item()
        #_,preds = torch.max(outputs.data,1)
        # statistics
        running_loss += loss.data.item()
        #running_corrects += torch.sum(preds == labels)
        size += bs
    epoch_loss = running_loss / size
        #epoch_acc = running_corrects.item() / size
    loss_train[epoch_num] = epoch_loss
        #acc_train[epoch_num] = epoch_acc
    print('Test - Loss: {:.4f}'.format(epoch_loss))
    
#%%
pose_detection.train(False)    
outputs = model(inputs_try)



    