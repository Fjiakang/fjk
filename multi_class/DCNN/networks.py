import torch.nn as nn
import torch
import torch.nn.functional as F


class D_CNN(nn.Module):
    def __init__(self,nums,size,channels):
        super(D_CNN,self).__init__()
        self.size = size
        self.channels = channels

        self.conv1 = nn.Sequential(
        nn.Conv2d(self.channels, 64, 3,padding=1),
        nn.MaxPool2d(2, 2),
        nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(128, 256, 3,padding=1),
        nn.MaxPool2d(2, 2),
        nn.ReLU()
        )
        self.Residual = nn.Sequential(
        nn.Conv2d(64,128,1),
        nn.MaxPool2d(2, 2),
        nn.ReLU()
        )
        self.branch1 = nn.Sequential(
            nn.Conv2d(64,32,1),#64
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32,32,5,padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(True)
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(64,64,1),#32
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64,64,3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )

        self.branch3 = nn.Sequential(
            nn.Conv2d(64,16,5,padding=2),#16
            nn.BatchNorm2d(16),
            nn.ReLU(True)
        )

        self.branch4 = nn.Sequential(
            nn.Conv2d(64,16,3,padding=1),#16
            nn.BatchNorm2d(16),
            nn.ReLU(True)
        )
        self.conv3 = nn.Sequential(
        nn.MaxPool2d(2,2),
        nn.Conv2d(128,128, 3,padding=1),
        nn.ReLU()
        )
        self.fc1 = nn.Linear(12*12*256,64)
        self.fc2 = nn.Linear(64,nums)
        self.f7 = nn.Softmax()
        self.f8 = nn.Dropout2d(0.5)
    def forward(self,x):
        y = self.conv1(x)
        j = self.Residual(y)
        q = torch.cat([self.branch1(y),self.branch2(y),self.branch3(y),self.branch4(y),],dim=1)
        y = self.conv3(q)
        y = torch.add(y, j)
        y = self.conv2(y)
        y = torch.flatten(y,1)
        y = F.dropout(y,0.5,training=self.training)
        y = self.fc1(y)
        y = F.relu(y, inplace=True)
        y = F.dropout(y, 0.5, training=self.training)
        y = self.fc2(y)
        y = self.f8(y)
        return y

        #return y

