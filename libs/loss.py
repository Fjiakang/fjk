import torch

class FocalLoss(torch.nn.Module):
 
    def __init__(self, gamma=2):
        super().__init__()
        self.gamma = gamma
 
    def forward(self, input, target):
        pt = input
        loss = - 0.25*(1 - pt) ** self.gamma * target * torch.log(pt) - \
            0.75*pt ** self.gamma * (1 - target) * torch.log(1 - pt)
        loss = torch.mean(loss)
        
        return loss