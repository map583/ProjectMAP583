#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 23:01:54 2019

@author: tete
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
%load_ext autoreload
%autoreload 2
from utils import load_data,default_loader,trainset
from torch.utils.data import Dataset, DataLoader
import torchvision
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
#%%
datapath = "/home/tete/data/Youtube"
data,paths,labels = load_data(datapath)
available_field = ['url','videoname','locs','frameids','label_names','crop','scale','origRes','isYouTubeSubset']
train_data  = trainset(paths,labels)
trainloader = DataLoader(train_data, batch_size=4,shuffle=True)
#%%
count = 1
for data in trainloader:
    if count == 1:
        inputs_try,labels_try = data
    count +=1
print(count,"batches")   
print(inputs_try.shape)
#%%
def normal(w,h,sigma,center=(160,90),mu=0):
    out=np.zeros(shape=[w,h])
    x,y=np.meshgrid(np.linspace(-2./w *center[0], 2./w*(w-center[0])),
                    np.linspace(-2./h *center[1], 2./h*(h-center[1])))
    d = np.sqrt(x*x+y*y)
    g = np.exp(-( (d-mu)**2 / ( 2.0 * sigma**2 ) ) )
    return g
def imshow(inp, label, title=None):
    
    inp = inp.numpy().transpose((1, 2, 0))
    print(inp.shape)
    
    annot=np.zeros(shape=(inp.shape[0],inp.shape[1]))
    print(annot.shape)
    for i in range(label.shape[0]):
        #print(label[0])
        for p in range(7):
            inp[int(label[i][p][1])][int(label[i][p][0])+320*i][0]+=10
    plt.figure(figsize=(10,10))
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated
#%%
out = torchvision.utils.make_grid(inputs_try,padding=0)
imshow(out,labels_try)
#%%
for i in trainloader:
    print(i)
#train_data.__getitem__(222)

