import torch.nn as nn
import torch
def gap(x):
    # input x [n, c, h, w]
    # output l [n, c]
    s = torch.max(x, dim=-1)
    s = torch.max(s, dim=-1)
    return s


class CAE_tr(nn.Module):
    def __init__(self,nums, nc):
        super().__init__()
        self.layer1_a = nn.Sequential(nn.Conv2d(nc,32,kernel_size=3),
                                 nn.ReLU())
        self.layer1_b = nn.Sequential(nn.Conv2d(nc,32,kernel_size=9,padding=3),
                                 nn.ReLU())
        self.layer1_p = nn.MaxPool2d(2,stride=2,return_indices=True) 


        self.layer2_a = nn.Sequential(nn.Conv2d(64,32,kernel_size=3),
                                 nn.ReLU())
        self.layer2_b = nn.Sequential(nn.Conv2d(64,32,kernel_size=9,padding=3),
                                 nn.ReLU())
        self.layer2_p = nn.MaxPool2d(2,stride=2,return_indices=True) 


        self.layer3_a = nn.Sequential(nn.Conv2d(64,32,kernel_size=3),
                                 nn.ReLU())
        self.layer3_b = nn.Sequential(nn.Conv2d(64,32,kernel_size=9,padding=3),
                                 nn.ReLU())
        self.layer3_p = nn.MaxPool2d(2,stride=2,return_indices=True) 


 
        self.layer4_p = nn.MaxUnpool2d(2,stride=2)
        self.layer4_a = nn.Sequential(nn.ConvTranspose2d(64,32,kernel_size=3),
                                 nn.ReLU())
        self.layer4_b = nn.Sequential(nn.ConvTranspose2d(64,32,kernel_size=9,padding=3),
                                 nn.ReLU())


        self.layer5_p = nn.MaxUnpool2d(2,stride=2)
        self.layer5_a = nn.Sequential(nn.ConvTranspose2d(64,32,kernel_size=3),
                                 nn.ReLU())
        self.layer5_b = nn.Sequential(nn.ConvTranspose2d(64,32,kernel_size=9,padding=3),
                                 nn.ReLU())


        self.layer6_p = nn.MaxUnpool2d(2,stride=2)
        self.layer6_a = nn.Sequential(nn.ConvTranspose2d(64,2,kernel_size=3),
                                 nn.ReLU())
        self.layer6_b = nn.Sequential(nn.ConvTranspose2d(64,1,kernel_size=9,padding=3),
                                 nn.ReLU())


        self.fc1 = nn.Sequential(nn.Linear(12*12*64,150),nn.Dropout2d())
        self.fc2 = nn.Sequential(nn.Linear(150,nums),nn.Dropout2d())

    def forward(self,x):
        out = self.layer1_a(x)
        out2 = self.layer1_b(x)
        out4= out2
        out = torch.cat([out,out2],dim=1)
        out1,i1 = self.layer1_p(out)

        out = self.layer2_a(out1)
        out2 = self.layer2_b(out1)
        out3 = out2
        out = torch.cat([out,out2],dim=1)
        out1,i2 = self.layer2_p(out)

        out = self.layer3_a(out1)
        out2 = self.layer3_b(out1)
        out = torch.cat([out,out2],dim=1)
        out,i3 = self.layer3_p(out)

        out1 = self.layer4_p(out,i3, output_size=out2.size())
        out = self.layer4_a(out1)
        out2 = self.layer4_b(out1)
        out = torch.cat([out,out2],dim=1)
        
        out1 = self.layer5_p(out,i2, output_size=out3.size())
        out = self.layer5_a(out1)
        out2 = self.layer5_b(out1)
        out = torch.cat([out,out2],dim=1)

        out1 = self.layer6_p(out,i1, output_size=out4.size())
        out = self.layer6_a(out1)
        out2 = self.layer6_b(out1)
        out = torch.cat([out,out2],dim=1)

        return out


class CAE_te(nn.Module):
    def __init__(self,nums, nc):
        super().__init__()
        self.layer1_a = nn.Sequential(nn.Conv2d(nc,32,kernel_size=3),
                                 nn.ReLU())
        self.layer1_b = nn.Sequential(nn.Conv2d(nc,32,kernel_size=9,padding=3),
                                 nn.ReLU())
        self.layer1_p = nn.MaxPool2d(2,stride=2) 


        self.layer2_a = nn.Sequential(nn.Conv2d(64,32,kernel_size=3),
                                 nn.ReLU())
        self.layer2_b = nn.Sequential(nn.Conv2d(64,32,kernel_size=9,padding=3),
                                 nn.ReLU())
        self.layer2_p = nn.MaxPool2d(2,stride=2) 


        self.layer3_a = nn.Sequential(nn.Conv2d(64,32,kernel_size=3),
                                 nn.ReLU())
        self.layer3_b = nn.Sequential(nn.Conv2d(64,32,kernel_size=9,padding=3),
                                 nn.ReLU())
        self.layer3_p = nn.MaxPool2d(2,stride=2) 


 
        self.layer4_p = nn.MaxUnpool2d(2,stride=2)
        self.layer4_a = nn.Sequential(nn.ConvTranspose2d(64,32,kernel_size=3),
                                 nn.ReLU())
        self.layer4_b = nn.Sequential(nn.ConvTranspose2d(64,32,kernel_size=9,padding=3),
                                 nn.ReLU())


        self.layer5_p = nn.MaxUnpool2d(2,stride=2)
        self.layer5_a = nn.Sequential(nn.ConvTranspose2d(64,32,kernel_size=3),
                                 nn.ReLU())
        self.layer5_b = nn.Sequential(nn.ConvTranspose2d(64,32,kernel_size=9,padding=3),
                                 nn.ReLU())


        self.layer6_p = nn.MaxUnpool2d(2,stride=2)
        self.layer6_a = nn.Sequential(nn.ConvTranspose2d(64,2,kernel_size=3),
                                 nn.ReLU())
        self.layer6_b = nn.Sequential(nn.ConvTranspose2d(64,1,kernel_size=9,padding=3),
                                 nn.ReLU())


        self.fc1 = nn.Sequential(nn.Linear(12*12*64,150),nn.Dropout2d())
        self.fc2 = nn.Sequential(nn.Linear(150,nums),nn.Dropout2d())

    def forward(self,x):
        out = self.layer1_a(x)
        out2 = self.layer1_b(x)
        out = torch.cat([out,out2],dim=1)
        out1 = self.layer1_p(out)

        out = self.layer2_a(out1)
        out2 = self.layer2_b(out1)
        out = torch.cat([out,out2],dim=1)
        out1 = self.layer2_p(out)

        out = self.layer3_a(out1)
        out2 = self.layer3_b(out1)
        out = torch.cat([out,out2],dim=1)
        out = self.layer3_p(out)

        out = out.view(out.size(0),-1)
        out = self.fc1(out)
        out = self.fc2(out)

        return out