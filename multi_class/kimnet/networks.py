#example
import torch.nn as nn

class kimnet_network(nn.Module):
    def __init__(self,nums,nc):
        super().__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(nc,4,kernel_size=5),
                                 nn.ReLU(),
                                 nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(nn.Conv2d(4,4,kernel_size=5),
                                 nn.ReLU(),
                                 nn.MaxPool2d(4))
        self.fc = nn.Linear(11*11*4,nums)

    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0),-1)
        out = self.fc(out)
        return out