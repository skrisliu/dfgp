# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 21:53:52 2024

@author: sjliu
"""

import torch 
import torch.nn as nn

#%% network
class HResNet(nn.Module):
    def __init__(self, bands=13):
        super(HResNet, self).__init__()

        self.conv1a = nn.Conv2d(bands, 32, kernel_size=3, stride=1, padding='valid')
        self.conv1b = nn.Conv2d(bands, 32, kernel_size=3, stride=1, padding='valid')
        
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2a = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding='same')
        self.conv2b = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding='same')
        self.avg = nn.AvgPool2d(3, stride=1)

        self.fc = nn.Linear(64, 1)
        
    def forward(self, x):
        
        x1 = self.conv1a(x)
        x2 = self.conv1b(x)
        x = torch.cat((x1,x2),1)
        
        x1 = self.bn1(x)
        x1 = nn.ReLU()(x1)
        x1 = self.conv2a(x1)
        x1 = nn.ReLU()(x1)
        x1 = self.conv2b(x1)
        
        x = x+x1
        x = self.avg(x)
        x = x.view(x.size(0), -1)
        out = self.fc(x)
        
        return out, x

###
class WCRN(nn.Module):
    def __init__(self, bands=13):
        super(WCRN, self).__init__()

        self.conv1a = nn.Conv2d(bands, 32, kernel_size=3, stride=1, padding=0)
        self.conv1b = nn.Conv2d(bands, 32, kernel_size=1, stride=1, padding=0)
        self.maxp1 = nn.MaxPool2d(kernel_size = 3)
        self.maxp2 = nn.MaxPool2d(kernel_size = 5)
        
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2a = nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0)
        self.conv2b = nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0)

        self.fc = nn.Linear(64, 1)
        
    def forward(self, x):
        out = self.conv1a(x)
        out1 = self.conv1b(x)
        out = self.maxp1(out)
        out1 = self.maxp2(out1)
        
        out = torch.cat((out,out1),1)
        
        out1 = self.bn1(out)
        out1 = nn.ReLU()(out1)
        out1 = self.conv2a(out1)
        out1 = nn.ReLU()(out1)
        out1 = self.conv2b(out1)
        
        out = torch.add(out,out1)
        x = out.reshape(out.size(0), -1)
        out = self.fc(x)
        
        return out, x
    

###
class WCRN3(nn.Module):
    def __init__(self, bands=13):
        super(WCRN3, self).__init__()

        self.conv1a = nn.Conv2d(bands, 32, kernel_size=3, stride=1, padding=0)
        self.conv1b = nn.Conv2d(bands, 32, kernel_size=1, stride=1, padding=0)
        self.maxp1 = nn.MaxPool2d(kernel_size = 1)
        self.maxp2 = nn.MaxPool2d(kernel_size = 3)
        
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2a = nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0)
        self.conv2b = nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0)

        self.fc = nn.Linear(64, 1)
        
    def forward(self, x):
        out = self.conv1a(x)
        out1 = self.conv1b(x)
        # out = self.maxp1(out)
        out1 = self.maxp2(out1)
        
        out = torch.cat((out,out1),1)
        
        out1 = self.bn1(out)
        out1 = nn.ReLU()(out1)
        out1 = self.conv2a(out1)
        out1 = nn.ReLU()(out1)
        out1 = self.conv2b(out1)
        
        out = torch.add(out,out1)
        x = out.reshape(out.size(0), -1)
        out = self.fc(x)
        
        return out, x
    




















