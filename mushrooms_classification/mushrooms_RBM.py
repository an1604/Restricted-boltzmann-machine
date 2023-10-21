# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 12:15:02 2023

@author: adina
"""

from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split

# Fetch dataset
from ucimlrepo import fetch_ucirepo
mushroom = fetch_ucirepo(id=73)

# Data (as pandas dataframes)
X = mushroom.data.features
y = mushroom.data.targets

# Handle the missing data
X = X.replace(np.nan, 'unknown')
y = y.replace('?', 'unknown')

# Encoding the features
le = LabelEncoder()
y = le.fit_transform(y)

le1 = LabelEncoder()

for column in X.columns:
   X[column] = le.fit_transform(X[column])

sc = MinMaxScaler(feature_range=(0, 1))
X_encoded = sc.fit_transform(X)

# dimention reduction 
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2 , f_classif

chi2_selector = SelectKBest(chi2 , k=5)
features = chi2_selector.fit_transform(X , y)

sc = MinMaxScaler(feature_range=(0, 1))
X_encoded = sc.fit_transform(features)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# replace the numpys arrays into torch Tensors
X_train = torch.Tensor(X_train)
X_test = torch.Tensor(X_test)
y_train = torch.Tensor(y_train)
y_test = torch.Tensor(y_test)


# Building the RBM class
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


# create an instance of the model
nv = X_train.shape[1]
nh = 100
batch_size = 100

rbm = RBM(nv, nh)
nb_epochs = 100

for epoch in range(1, nb_epochs + 1):
    train_loss = 0
    s = 0

    for id_mushroom in range(0, X_train.shape[0] - batch_size + 1, batch_size):
        vk = X_train[id_mushroom: id_mushroom + batch_size]
        v0 = X_train[id_mushroom: id_mushroom + batch_size]

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
for id_mushroom in range(X_test.shape[0]):
    v = X_test[id_mushroom: id_mushroom+ 1]
    """## we keeping this as trining_set cause we need to activate the neurons 
      of the RBM to get the predicted ratings of the tset set"""
    vt = X_test[id_mushroom: id_mushroom+1]
    if len(vt[vt >= 0 ]) >= 0 :
      _, h = rbm.sample_h(v)
      _, v = rbm.sample_v(h)
      test_loss += torch.mean(torch.abs(vt[vt >= 0] - v[vt >= 0]))
      s += 1.
print('test loss: '+str(test_loss.item()/s))






