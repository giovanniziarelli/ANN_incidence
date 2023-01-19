# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import os

folder = 'Test_ANN_incidence_NPI_prova/'#5000 neuroni hidden layer 

os.mkdir(folder)

class Net(nn.Module):
    def __init__(self, I0, training_size, dt, gamma):
      super(Net, self).__init__()
      self.neurons = 50
      self.dimTimes = 100
      self.init_shape = 100 # Timesteps X params_type, in this case 100 X 1 (only NPIs, no temperature nor other QOIs)
      self.fc0 = nn.Linear(self.init_shape, self.neurons)
      self.fc1 = nn.Linear(self.neurons,self.neurons)
      self.fc2 = nn.Linear(self.neurons,self.dimTimes-1)
      self.training_size = training_size
      self.dt = dt
      self.I0 = I0
      self.gamma = gamma
      self.I = torch.from_numpy(self.I0*np.ones((self.training_size,self.dimTimes)))
      self.tol = 0.0001#1e-5
      self.nmax = 100#1e2
      print('Number of neurons per layer', self.neurons)
   
    def forward(self, x):
      x1 = self.fc0(x).clone()
      x2 = F.relu(x1, inplace = False).clone()
      x3 = self.fc1(x2).clone()
      x4 = F.relu(x3, inplace = False).clone()
      x5 = self.fc2(x4).clone()
      output = x5.clone()
      self.I = torch.from_numpy(self.I0[0] * np.ones((self.training_size, self.dimTimes))).squeeze()
      
      for k in range(1,self.dimTimes):
          self.I[:,k] = self.I[:,k-1].clone() + self.dt * (output[:,k-1].clone() * self.I[:, k-1].clone() * (1 - self.I[:,k-1].clone()) - self.gamma * self.I[:,k-1].clone())
      return self.I.type(torch.double)

# Function modelling the fictious dependency of the transmission rate on the parameters
# For instance: temperature, variant prevelances, variant infctiousness, NPIs, percentage of vaccinations

def beta_fun_dep(t, params, training_size):
    NPIs = params
    aux = 0.5* NPIs
    return aux

#SOLVER SIS
def solution_SIS_vec(I0, fun, dt, gamma):
  
  I = I0 * np.ones(int(1/dt))
  for i in range(1, int(1/dt)):
    I[i] = I[i-1] + dt * (fun[i-1] * (1 - I[i-1]) * I[i-1] - gamma * I[i-1])
  
  return I

# Converter for NPIs
def convert(a):
    l = list()
    for i in range(len(a)):
        if a[i] == 1:
            l.append('White')
        elif a[i] == 4:
            l.append('Red')
        elif a[i] == 3:
            l.append('Orange')
        elif a[i] == 2:
            l.append('Yellow')
        elif a[i] == 5:
            l.append('Black')
        else:
            raise Exception("Error in NPI colors.")
    return l


# Solver parameters
I0 = 0.12
dt = 1/100
gamma = 1/20
t_points_len = 100
time_range = np.linspace(0,1,100)

# Hyperparameters
training_size = 30
learning_rate = 1e-3 
batch_size = training_size #complete batch
epochs =100#1000 
loss_list = []

torch.autograd.set_detect_anomaly(True)

# Net initialization
NN = Net(I0*np.ones(t_points_len), training_size, dt, gamma)

# Optimizer
optimizer = torch.optim.Adam(NN.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

# Training input (Non-Pharmaceutical Interventions)
np.random.seed(1)
NPIs_vec = np.zeros((t_points_len, training_size))

for i in range(training_size):
    NPIs_vec[:,i] = np.clip(np.random.rand() * np.sin(4 * np.pi * time_range) + np.random.rand(), 0, 1) 
    #if i % 5 ==0:
    #    plt.plot(NPIs_vec[:,i])
    #    plt.show()

# Converting NPIs into piecewise constant functions
NPIs_vec[NPIs_vec <= 0.1] = 0.1
NPIs_vec[(NPIs_vec>0.1)*(NPIs_vec<=0.3)] = 0.3
NPIs_vec[(NPIs_vec>0.3)*(NPIs_vec<=0.6)] = 0.6
NPIs_vec[(NPIs_vec>0.6)*(NPIs_vec<=0.9)] = 0.9
NPIs_vec[(NPIs_vec>0.9)*(NPIs_vec<=1)] = 1

# Loss function
loss_fn = nn.MSELoss()

# Training set
y_train = np.zeros((t_points_len, training_size))
params = NPIs_vec
betas = beta_fun_dep(time_range, params, training_size) 
X_train_list = list()

for i in range(training_size):
    y_train[:,i] = solution_SIS_vec(I0,betas[:,i],dt,gamma)
    X_train_list.append(NPIs_vec.T[i])

X_train = np.array(X_train_list).astype(np.float32)

from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

y_train = (y_train.T).astype(np.float32)
train_ds = TensorDataset(torch.Tensor(X_train), torch.Tensor(y_train).type(torch.double))
train_dataloader =DataLoader(train_ds, batch_size = batch_size)

from re import I

# Train loop definition
def train_loop(dataloader, model, loss_fn, optimizer,scheduler):
    size = len(dataloader.dataset)
    loss_list = []
    for batch, (X, y) in enumerate(dataloader):

        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:

            loss, current = loss.item(), batch * len(X)
            loss_list.append(loss)#torch.Tensor.detach(loss).numpy())
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    return loss_list[-1]

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    l = train_loop(train_dataloader, NN, loss_fn, optimizer,scheduler) 
    loss_list.append(l)
    scheduler.step(t)
print("Done!")


with open(folder + "loss.txt", "w") as output:
    output.write(str(loss_list))


for batch, (X, y) in enumerate(train_dataloader):
    
    pred = NN(X)
    #Plot
    for i in range(batch_size):
    # Compute prediction and loss
        plt.plot(pred[i].detach().numpy())
        plt.plot(y[i].detach().numpy(), '--')
        plt.savefig(folder + 'train_'+str(i)+'.png')
        plt.close()
        #plt.show()

torch.save(NN.state_dict(), folder+'NNstateTest.pth')

#model_scripted = torch.jit.script(NN) # Export to TorchScript
#model_scripted.save(folder+'model_scripted_Test'+str(k)+'.pt')

