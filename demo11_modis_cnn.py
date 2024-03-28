# -*- coding: utf-8 -*-
"""
Deep Feature Gaussian Processes for Single-Scene Aerosol Optical Depth Reconstruction

@author: skrisliu

skrisliu@gmail.com
"""

import numpy as np
import torch 
import torch.nn as nn
import rscls
import torch.utils.data as Data 
import time
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import argparse
import networks as nw
from datetime import datetime


parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--seed', type=int, default = 42)
args = parser.parse_args()


patch = 3
bsz = 64
PRINT = 'False'


yfile = 'data/modis/aod.npy'
imfile = 'data/modis/im.npy'


ytrainmask = np.load('data/modis/trainmask.npy')
ytestmask = np.load('data/modis/testmask.npy')


num_epochs = 50
    

torch.cuda.empty_cache()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


#%% load data
y = np.load(yfile)
y = (y-y.min()) / (y.max()-y.min() + 1e-6)
y = np.float32(y)


im = np.load(imfile)
im = np.float32(im)
for i in range(im.shape[-1]):
    im[:,:,i] = ( im[:,:,i]-im[:,:,i].min() ) / ( im[:,:,i].max()-im[:,:,i].min() + 1e-6 )

    
#%% load train-test samples
gt = y*ytrainmask

c1 = rscls.rscls(im,gt,cls=1)
c1.padding(patch)
x1_train = c1.all_sample()
x1_train = np.transpose(x1_train, (0,3,1,2))  

x1_train = x1_train[ytrainmask.reshape(-1),:,:,:]
y1_train = y[ytrainmask]
y1_train = y1_train.reshape(-1,1)



#% test data
gt = y*ytestmask

c2 = rscls.rscls(im,gt,cls=1)
c2.padding(patch)
x1_test = c2.all_sample()

x1_test = x1_test[ytestmask.reshape(-1),:,:,:]
y1_test = y[ytestmask].reshape(-1,1)
x1_test = np.transpose(x1_test, (0,3,1,2))






#%% train set
x1_train,y1_train = torch.from_numpy(x1_train),torch.from_numpy(y1_train)
train_set = Data.TensorDataset(x1_train,y1_train) 

train_loader = Data.DataLoader(
            dataset = train_set,
            batch_size = bsz,
            shuffle = True,
            num_workers = 0,
            )


#%% test set
x1_test,y1_test = torch.from_numpy(x1_test),torch.from_numpy(y1_test)
test_set = Data.TensorDataset(x1_test,y1_test)

test_loader = Data.DataLoader(
            dataset = test_set,
            batch_size = bsz,
            shuffle = False,
            num_workers = 0,
            )


#%% define network
train_N = len(train_set)

model = nw.WCRN3(bands=im.shape[-1])
model.to(device)
criterion = nn.L1Loss()



#%%  begin training
if True:
    time1 = int(time.time())
    
    # train the model using lr=0.1
    train_model = True
    if train_model:
        lr = 0.01
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
        model.train()
        
        total_step = len(train_loader)
        
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            running_loss = 0.0
            
            for i, (images, labels) in enumerate(train_loader):
                images = images.to(device)
                labels = labels.to(device)
                
                # Forward pass
                outputs,_ = model(images)
                loss = criterion(outputs, labels)
                
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += outputs.shape[0] * loss.item()
                running_loss += loss.item()
            
            if PRINT:
                print(epoch+1, epoch_loss / train_N)
            
    # train the model using lr=0.01
    train_model = True
    if train_model:
        lr = 0.001
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
        model.train()
        
        total_step = len(train_loader)
        
        for epoch in range(num_epochs//4):
            epoch_loss = 0.0
            running_loss = 0.0
            
            for i, (images, labels) in enumerate(train_loader):
                images = images.to(device)
                labels = labels.to(device)
                
                # Forward pass
                outputs,_ = model(images)
                loss = criterion(outputs, labels)
                
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += outputs.shape[0] * loss.item()
                running_loss += loss.item()
            
            if PRINT:
                print(epoch+1, epoch_loss / train_N)


#%% Test the model
if True:
    if True:
        pre = []
        obs = []
        model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                outputs,_ = model(images)
                
                pre.append(outputs.cpu().numpy().reshape(-1))
                obs.append(labels.numpy())
        
                
pre2 = np.concatenate(pre)
obs2 = np.concatenate(obs)

rr = r2_score(obs2,pre2)
mae = mean_absolute_error(obs2,pre2)
mse = mean_squared_error(obs2,pre2,squared=False)


#%% Get prediction map
gt = np.ones([im.shape[0],im.shape[1]])==1

c3 = rscls.rscls(im,gt,cls=1)
c3.padding(patch)
x1_predict = c3.all_sample()



x1_predict = np.transpose(x1_predict, (0,3,1,2))
y1_predict = np.zeros([x1_predict.shape[0],1])
x1_predict,y1_predict = torch.from_numpy(x1_predict),torch.from_numpy(y1_predict)
predict_set = Data.TensorDataset(x1_predict,y1_predict)

predict_loader = Data.DataLoader(
            dataset = predict_set,
            batch_size = bsz,
            shuffle = False,
            num_workers = 0,
            )


prex = []
obsx = []
model.eval()
with torch.no_grad():
    for images, labels in predict_loader:
        images = images.to(device)
        outputs,_ = model(images)
        
        prex.append(_.cpu().numpy())
    

prex2 = np.concatenate(prex) 
prex3 = prex2.reshape([im.shape[0],im.shape[1],prex2.shape[-1]])


#%% save model, features, and result summary
res = np.array([rr,mae,mse])

mPATH = 'save/modis_model_'+format(args.seed,'02d')+'_'+datetime.now().strftime("%Y%m%d%H%M%S")+'_rr'+format(rr,'.4f')+'.pt'
sPATH = 'save/modis_fea64_'+format(args.seed,'02d')+'_'+datetime.now().strftime("%Y%m%d%H%M%S")+'_rr'+format(rr,'.4f')+'.npy'
xPATH = 'save/modis_result_'+format(args.seed,'02d')+'_'+datetime.now().strftime("%Y%m%d%H%M%S")+'_rr'+format(rr,'.4f')+'.pt'

torch.save(model.state_dict(), mPATH)
np.save(sPATH, prex3)
np.save(xPATH, res)















