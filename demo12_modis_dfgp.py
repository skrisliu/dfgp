# -*- coding: utf-8 -*-
"""
Deep Feature Gaussian Processes for Single-Scene Aerosol Optical Depth Reconstruction

@author: skrisliu

skrisliu@gmail.com

"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import gpytorch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_squared_error
import time
from sklearn.linear_model import LinearRegression
import argparse
import pandas as pd


parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--bsz', type=int, default = 2048)
parser.add_argument('--seed', type=int, default = 42)
args = parser.parse_args()
batch_size = args.bsz
print(args.seed)


OPTIMIZER0 = 'Adam'
COV_MODE = 'DFGP'  # DFGP, DFGPS


imfile = 'data/modis/fea64.npy'
yfile = 'data/modis/aod.npy'
trainfile = 'data/modis/trainmask.npy'
testfile = 'data/modis/testmask.npy'


#%% load data
im = np.load(imfile)
im = im.reshape(240,300,64) # 64 dimensions
im = im.astype(np.float32)

#% add xy
im_x1 = np.zeros([im.shape[0],im.shape[1]])
im_x2 = np.zeros([im.shape[0],im.shape[1]])

for i in range(im.shape[0]):
    for j in range(im.shape[1]):
        im_x1[i,j] = i
        im_x2[i,j] = j

im_x1 = im_x1.reshape(im_x1.shape[0],im_x1.shape[1],1).astype(np.float32)
im_x2 = im_x2.reshape(im_x2.shape[0],im_x2.shape[1],1).astype(np.float32)

# scale to closer to 0-1
im_x1 = im_x1 / 240
im_x2 = im_x2 / 240

for i in range(im.shape[-1]):
    im[:,:,i] = ( im[:,:,i]-im[:,:,i].min() ) / ( im[:,:,i].max()-im[:,:,i].min() + 1e-6 )

im = np.concatenate([im,im_x1,im_x2],axis=-1)



#%% train test
trmask = np.load(trainfile)
temask = np.load(testfile)


x_tr = im[trmask] # training x
x_te = im[temask] # testing x

y = np.load(yfile)
y = (y - y.min()) / (y.max()-y.min()+1e-7)
y_tr = y[trmask] 
y_te = y[temask] 


#%% linear regression as baseline
reg = LinearRegression().fit(x_tr, y_tr)
pre1_lnearReg = reg.predict(x_te)

rr_raw = r2_score(y_te,pre1_lnearReg)

model0_params = reg.coef_
model0_params = model0_params.reshape(-1,1)
model0_params = np.float32(model0_params)


#%% numpy to pytorch format
x_tr = torch.Tensor(x_tr).contiguous()
x_te = torch.Tensor(x_te).contiguous()
y_tr = torch.Tensor(y_tr).contiguous()
y_te = torch.Tensor(y_te).contiguous()


# if gpu is available, load data to gpu
if torch.cuda.is_available():
    x_tr, y_tr, x_te, y_te = x_tr.cuda(), y_tr.cuda(), x_te.cuda(), y_te.cuda()


# data loader for training data -- pytorch specific function
train_dataset = TensorDataset(x_tr, y_tr)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) 

# data loader for testing data --pytorch specific function; shuffle must be False in testing
test_dataset = TensorDataset(x_te, y_te)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


#%% define network and GP
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy
from gpytorch.means import Mean
from gpytorch.constraints import Interval


# LinearMean function in GP (the XB in y=GP(XB,Σ)). This is a built-in function of GPytorch. Extracted directly here. 
class LinearMean(Mean):
    def __init__(self, input_size, batch_shape=torch.Size(), bias=True):
        super().__init__()
        self.register_parameter(name="weights", parameter=torch.nn.Parameter(torch.randn(*batch_shape, input_size, 1)))
        if bias:
            self.register_parameter(name="bias", parameter=torch.nn.Parameter(torch.randn(*batch_shape, 1)))
        else:
            self.bias = None
        self.weights = torch.nn.Parameter(torch.from_numpy(model0_params))

    def forward(self, x):
        res = x.matmul(self.weights).squeeze(-1)
        if self.bias is not None:
            res = res + self.bias
        return res
    

# DFGPS
class DFGPS(ApproximateGP):
    def __init__(self, inducing_points,likelihood):
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = VariationalStrategy(self, inducing_points, variational_distribution, learn_inducing_locations=True)
        super(DFGPS, self).__init__(variational_strategy)
        self.mean_module = LinearMean(input_size=66)
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(lengthscale_constraint=Interval(0.0125, 0.0350)))
        init_lengthscale = 0.02
        self.covar_module.base_kernel.initialize(lengthscale=init_lengthscale)
        self.likelihood = likelihood
        

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x[:,-2:]) 
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


# DFGP
class DFGP(ApproximateGP):
    def __init__(self, inducing_points,likelihood):
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = VariationalStrategy(self, inducing_points, variational_distribution, learn_inducing_locations=True)
        super(DFGP, self).__init__(variational_strategy)
        self.mean_module = LinearMean(input_size=66)
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(
            lengthscale_constraint=Interval(0.2, 0.8)))
        init_lengthscale = 0.4
        self.covar_module.base_kernel.initialize(lengthscale=init_lengthscale)
        self.likelihood = likelihood

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    


#%% build model

inducing_points = x_tr[:1000, :]  
likelihood = gpytorch.likelihoods.GaussianLikelihood()

if COV_MODE=='DFGPS':
    model = DFGPS(inducing_points=inducing_points,likelihood=likelihood)
elif COV_MODE=='DFGP':
    model = DFGP(inducing_points=inducing_points,likelihood=likelihood)
else:
    print('ERROR: Unknown Covariance Mode!!!')

if torch.cuda.is_available():
    model = model.cuda()
    likelihood = likelihood.cuda()


#%% training 1, learning rate=0.01
num_epochs = 300

model.train()
likelihood.train()


lr = 0.01
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=y_tr.size(0))

# training starts here
epochs_iter = np.arange(num_epochs)
for i in epochs_iter:
    printloss = 0
    count = 0
    for j, (x_batch, y_batch) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(x_batch)
        loss = -mll(output, y_batch)
        if True:
            print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
                i + 1, num_epochs, loss.item(), 
                model.covar_module.base_kernel.lengthscale, 
                model.likelihood.noise.item()
            ))
        loss.backward()
        optimizer.step()
        printloss += loss.item()
        count += 1
    print(i,printloss/count)
 
#%% training 2, smaller learning rate=0.001
num_epochs = 100

model.train()
likelihood.train()


lr = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

epochs_iter = np.arange(num_epochs)
for i in epochs_iter:
    printloss = 0
    count = 0
    for j, (x_batch, y_batch) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(x_batch)
        loss = -mll(output, y_batch)
        if True:
            print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
                i + 1, num_epochs, loss.item(), 
                model.covar_module.base_kernel.lengthscale, 
                model.likelihood.noise.item()
            ))
        loss.backward()
        optimizer.step()
        printloss += loss.item()
        count += 1
    print(i,printloss/count)

#%% testing
model.eval()
likelihood.eval()
means = torch.tensor([0.])
with torch.no_grad():
    for x_batch, y_batch in test_loader:
        preds = model(x_batch)
        means = torch.cat([means, preds.mean.cpu()]) # only get the mean of the prediction
means = means[1:]


# test summary
pp1 = y_te.cpu().numpy()
pp2 = means.numpy()

rr = r2_score(pp1,pp2)
mae = mean_absolute_error(pp1,pp2)
mse = mean_squared_error(pp1,pp2,squared=False)
print(mse,mae,rr)



mPATH = 'save/modis_model_'+COV_MODE +'_'+str(args.seed)+'.pt'
torch.save(model.state_dict(), mPATH)


sPATH = 'save/modis_summary_'+COV_MODE+'_'+str(args.seed)+'.csv'
a = {'mean-weight': model.mean_module.weights.detach().cpu().numpy().reshape(-1),
              'mean-bias': model.mean_module.bias.detach().cpu().numpy().reshape(-1), 
              'cov-lscale': model.covar_module.base_kernel.lengthscale.detach().cpu().numpy().reshape(-1),
              'metric': np.concatenate([rr.reshape(-1),mse.reshape(-1),mae.reshape(-1)])}
df = pd.DataFrame.from_dict(a, orient='index')
df = df.transpose()
df.to_csv(sPATH)





























