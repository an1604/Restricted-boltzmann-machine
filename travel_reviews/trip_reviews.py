# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 12:38:47 2023

@author: adina
"""

import pandas as pd 
import numpy as np
import torch

dataset = pd.read_csv('tripadvisor_review.csv')

# Iterate through the dataset and convert ratings to binary values
for i in range(dataset.shape[0]):
    for j in range(1, dataset.shape[1]):
        if float(dataset.iloc[i, j]) >= 3:
            dataset.iloc[i, j] = 1
        else:
            dataset.iloc[i, j] = 0
            
# encode the userID column 
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
dataset['User ID'] = le.fit_transform(dataset['User ID'])    

# feature scaling 
from sklearn.preprocessing import MinMaxScaler
sc =MinMaxScaler(feature_range=(0,1))
dataset['User ID'] = sc.fit_transform(dataset[['User ID']].values)  
# splitting the data into train set and test set
from sklearn.model_selection import train_test_split
training_set , test_set = train_test_split(dataset, test_size=0.2 , random_state=42)

# Convert DataFrames to NumPy arrays
training_set = training_set.to_numpy()
test_set = test_set.to_numpy()

# transfer them to torch tensors 
training_set = torch.Tensor(training_set)
test_set = torch.Tensor(test_set)

# build the RBM model 
class RBM(torch.nn.Module):
    def __init__(self, nv, nh):
        super(RBM, self).__init__()
        self.W = torch.nn.Parameter(torch.randn(nh, nv))
        self.a = torch.nn.Parameter(torch.randn(1, nh))
        self.b = torch.nn.Parameter(torch.randn(1, nv))
        
    def sample_h(self, x):
        activation = torch.mm(x, self.W.t()) + self.a
        p_h_given_v = torch.sigmoid(activation)
        return p_h_given_v, torch.bernoulli(p_h_given_v)
    
    def sample_v(self, x):
        activation = torch.mm(x, self.W) + self.b
        p_v_given_h = torch.sigmoid(activation)
        return p_v_given_h, torch.bernoulli(p_v_given_h)
    
    def free_energy(self, x):
        wx_b = torch.mm(x, self.W.t()) + self.b
        hidden_term = wx_b.exp().add(1).log().sum(1)
        return -hidden_term
    
    def contrastive_divergence(self, v0, k=1):
        vk = v0
        ph0, _ = self.sample_h(v0)
        
        for _ in range(k):
            _, hk = self.sample_h(vk)
            _, vk = self.sample_v(hk)
        
        phk, _ = self.sample_h(vk)
        
        return v0, vk, ph0, phk
    
    def train(self, v0, vk, ph0, phk, lr=0.01):
        self.W += lr * (torch.mm(v0.t(), ph0) - torch.mm(vk.t(), phk))
        self.b += lr * torch.sum((v0 - vk), 0)
        self.a += lr * torch.sum((ph0 - phk), 0)


nv = training_set.shape[1]
nh = 100
batch_size = 100

rbm = RBM(nv, nh)
nb_epochs = 200

for epoch in range(1, nb_epochs + 1):
    train_loss = 0
    s = 0

    for id_review in range(0, training_set.shape[0] - batch_size + 1, batch_size):
        vk = training_set[id_review: id_review + batch_size]
        v0 = training_set[id_review: id_review + batch_size]

        for k in range(10):
            ph0, _ = rbm.sample_h(v0)
            _, hk = rbm.sample_h(vk)
            v0, vk, _, _ = rbm.contrastive_divergence(v0, k=1)

        train_loss += torch.mean(torch.abs(v0- vk))
        s += 1

    print('epoch : ' + str(epoch) + ' loss : ' + str(train_loss.item() / s))


#test the RBM 
test_loss = 0
s = 0.
for id_review in range(test_set.shape[0]):
    v = test_set[id_review: id_review+ 1]
    """## we keeping this as trining_set cause we need to activate the neurons 
      of the RBM to get the predicted ratings of the tset set"""
    vt = test_set[id_review: id_review+1]
    if len(vt[vt >= 0 ]) >= 0 :
      _, h = rbm.sample_h(v)
      _, v = rbm.sample_v(h)
      test_loss += torch.mean(torch.abs(vt[vt >= 0] - v[vt >= 0]))
      s += 1.
print('test loss: '+str(test_loss.item()/s))