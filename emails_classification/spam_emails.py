# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 10:21:03 2023

@author: adina
"""

from ucimlrepo import fetch_ucirepo 
import pandas as pd  
# fetch dataset 
spambase = fetch_ucirepo(id=94) 
  
# data (as pandas dataframes) 
X = spambase.data.features 
y = spambase.data.targets 

# feature selction 
from sklearn.feature_selection import VarianceThreshold
# Create thresholder 
thresholder = VarianceThreshold(threshold=.5)
# Create high variance feature matrix
features_high_variance = thresholder.fit_transform(X)

# returning the matrix to dataframe again to see the features names 
selected_feature_indices = thresholder.get_support()
selected_feature_names = X.columns[selected_feature_indices]

# Create a new DataFrame with the selected features
X = X[selected_feature_names]

#feature scaling 
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0,1))
columns = X.columns
X = sc.fit_transform(X)
X = pd.DataFrame(X ,columns= columns)

# Split the data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# convert them to torch tensors 
import torch
import numpy as np
X_train = torch.Tensor(np.array(X_train))
X_test = torch.Tensor(np.array(X_test))
y_train = torch.Tensor(np.array(y_train))
y_test = torch.Tensor(np.array(y_test))


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
        
        
nv = X_train.shape[1]
nh = 100
batch_size = 100

rbm = RBM(nv, nh)
nb_epochs = 100

for epoch in range(1, nb_epochs + 1):
    train_loss = 0
    s = 0

    for id_email in range(0, X_train.shape[0] - batch_size + 1, batch_size):
        vk = X_train[id_email: id_email + batch_size]
        v0 = X_train[id_email: id_email + batch_size]

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

threshold = 0.5  # Define a threshold for deciding binary predictions

predictions = []  # List to store predictions for each data point

for id_email in range(X_test.shape[0]):
    v = X_test[id_email: id_email + 1]
    
    # Activate the neurons of the RBM
    _, h = rbm.sample_h(v)
    _, v = rbm.sample_v(h)
    
    # Compute the loss
    vt = X_test[id_email: id_email + 1]
    loss = torch.mean(torch.abs(vt[vt >= 0] - v[vt >= 0]))
    
    test_loss += loss
    s += 1.
    
    prediction = 1 if loss < threshold else 0
    
    predictions.append(prediction)

# Calculate the test loss
test_loss = test_loss.item() / s
print('test loss: ' + str(test_loss))

# Print the predictions
print('Predictions:', predictions)

# confusion matrix and accuracy rate 
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Actual labels (ground truth)
actual_labels = y_test  # Replace with your actual test labels

# Create a confusion matrix
cm = confusion_matrix(actual_labels, predictions)

# Define class labels for better visualization (e.g., 'Spam' and 'Non-Spam')
class_labels = ['Non-Spam', 'Spam']

# Create a heatmap for visualization
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

true_positives = cm[1, 1]  
true_negatives = cm[0, 0]  
total = len(y_test)  

accuracy = (true_positives + true_negatives) / total
print(f'Accuracy: {accuracy:.2%}')


