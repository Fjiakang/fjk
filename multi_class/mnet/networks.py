# 引用文章：Human Motion Recognition With Limited Radar Micro-Doppler Signatures
import torch
import torch.nn as nn
class m_net(nn.Module):
    def __init__(self,nums,size,channels):
        super(m_net,self).__init__()
        self.size = size
        self.channels = channels
        
        self.conv1 = nn.Sequential(
        nn.Conv2d(self.channels,16,3),
        nn.MaxPool2d(2,2),
        nn.ReLU()
        )
        self.conv2 = nn.Sequential(
        nn.Conv2d(16,32,3),
        nn.BatchNorm2d(32),
        nn.MaxPool2d(2,2),
        nn.ReLU()
        )
        self.conv3 = nn.Sequential(
        nn.Conv2d(32,32,3,padding = 1),
        nn.BatchNorm2d(32),
        nn.ReLU()
        )
        self.conv4 = nn.Sequential(
        nn.Conv2d(32,64,3),
        nn.BatchNorm2d(64),
        nn.MaxPool2d(2,2),
        nn.ReLU()
        )
        self.conv5 = nn.Sequential(
        nn.Conv2d(64,64,3,padding = 1),
        nn.BatchNorm2d(64),
        nn.ReLU()
        )
        self.conv6 = nn.Sequential(
        nn.Conv2d(64,128,3),
        nn.BatchNorm2d(128),
        nn.MaxPool2d(2,2),
        nn.ReLU()
        )

        self.dilconv1 = nn.Sequential(
        nn.Conv2d(32,32,3,padding = 2,dilation = 2),
        nn.BatchNorm2d(32),
        nn.Tanh()
        )
        self.dilconv2 = nn.Sequential(
        nn.Conv2d(64,64,3,padding = 2,dilation = 2),
        nn.BatchNorm2d(64),
        nn.Tanh()
        )
        

        self.f1 = nn.Sequential(nn.Conv2d(32,64,1,1),nn.ReLU(),nn.Conv2d(64,32,1,1),nn.Sigmoid())                   
        self.f2 = nn.Sequential(nn.Conv2d(64,128,1,1),nn.ReLU(),nn.Conv2d(128,64,1,1),nn.Sigmoid())                   
   
        self.fc1 = nn.Linear(128*self.calsize(self.size),128)
        self.fc2 = nn.Linear(128,nums)
        self.f7 = nn.Softmax()
        self.f8 = nn.Dropout2d(0.5)
    
    def forward(self,x):
        y = self.conv1(x)
        y = self.conv2(y)
        y1 = y*self.f1(self.gap(y).view(y.size(0),y.size(1),1,1))
        y = torch.add(y1,self.conv3(self.dilconv1(y)))        
        y = self.conv4(y)
        y2 = y*self.f2(self.gap(y).view(y.size(0),y.size(1),1,1))    
        y = torch.add(y2,self.conv5(self.dilconv2(y)))
        y = self.conv6(y)
        y = self.fc1(y.view(y.size(0),-1))
        y = self.f8(y)
        return(self.f7(self.fc2(y)))
    
    def calsize(self,size):
        return pow(((size-30)//16),2)

    def gap(self,input):
        s = torch.mean(input, dim=-1)
        s = torch.mean(s, dim=-1)
        return s    
    
	
