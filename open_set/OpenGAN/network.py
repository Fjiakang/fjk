#引用文章：Open_set human activity recognition based on micro_Doppler signatures
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self,nc,tensorimg):
        super(Generator, self).__init__()
        self.growbase = 64
        self.nc = nc
        self.tensorimg = tensorimg
        self.input_layer = nn.Sequential(nn.ConvTranspose2d(self.tensorimg,self.growbase,kernel_size=4,stride=2,padding=0,bias=False))
        self.output_layer = nn.Sequential(nn.Conv2d(512,self.nc,kernel_size=1,bias=False))
        for i in range(1,4):
                exec('self.DB{}=self.dbblock(self.growbase*pow(2,{}))'.format(i,i-1))
                exec('self.TL{}=self.tlblock(self.growbase*pow(2,{}))'.format(i,i))

    def dbblock(self,inc):
        return nn.Sequential(
            nn.Conv2d(inc,inc*2,kernel_size=1,bias=False),
            nn.BatchNorm2d(inc*2),
            nn.ReLU(),
            nn.Conv2d(inc*2,inc*2,kernel_size=3,padding=1,bias=False),
            nn.BatchNorm2d(inc*2),
            nn.ReLU()
            )
    def tlblock(self,inc):
        return nn.Sequential(
            nn.Conv2d(inc,inc,kernel_size=1,bias=False),
            nn.ConvTranspose2d(inc,inc,2,2,0,bias=False)
            )

    def forward(self,x):
        out=self.input_layer(x)
        out1=out
        out=self.DB1(out)
        # out=out+out1
        out=self.TL1(out)
        out2=out
        out=self.DB2(out)
        # out=out+out2
        out=self.TL2(out)
        out3=out
        out=self.DB3(out)
        # out=out+out3
        out=self.TL3(out)
        out=self.output_layer(out)
        return out

class Discriminator(nn.Module):
    def __init__(self, nc,num_class):
        super(Discriminator, self).__init__()
        self.growbase = 64
        self.input_layer = nn.Sequential(nn.Conv2d(nc,self.growbase,kernel_size = 4,stride = 4,bias=False))
        self.output_layer = nn.Sequential(
                            nn.Linear(512,num_class+1),
                            nn.Softmax(dim=1)
                            )
        for i in range(1,4):
                exec('self.DB{}=self.dbblock(self.growbase*pow(2,{}))'.format(i,i-1))
                exec('self.TL{}=self.tlblock(self.growbase*pow(2,{}))'.format(i,i))

    def dbblock(self,inc):
        return nn.Sequential(
            nn.Conv2d(inc,inc*2,kernel_size=1,bias=False),
            nn.BatchNorm2d(inc*2),
            nn.ReLU(),
            nn.Conv2d(inc*2,inc*2,kernel_size=3,padding=1,bias=False),
            nn.BatchNorm2d(inc*2),
            nn.ReLU()
            )
        
    def tlblock(self,inc):
        return nn.Sequential(
            nn.Conv2d(inc,inc,kernel_size=1,bias=False),
            nn.AvgPool2d(2,stride=2)
            )


    def forward(self,x):
        out=self.input_layer(x)
        out1=out
        out=self.DB1(out)
        # out=out+out1
        out=self.TL1(out)
        out2=out
        out=self.DB2(out)
        # out=out+out2
        out=self.TL2(out)
        out3=out
        out=self.DB3(out)
        # out=out+out3
        out=self.TL3(out)
        out=out.view(out.size(0),-1)
        out=self.output_layer(out)

        return out