
import torch.nn as nn

###
def weights_init(mod):

    classname = mod.__class__.__name__
    if classname.find("Conv") != -1:
        mod.weight.data.normal_(0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        mod.weight.data.normal_(1.0, 0.02)
        mod.bias.data.fill_(0)


###
class Decoder(nn.Module):
    def __init__(self, nc):
        super(Decoder, self).__init__()
        super().__init__()
        self.layer1 = nn.Sequential(nn.ConvTranspose2d(512, 256, 5, 2, 0, bias=False), nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        self.layer2 = nn.Sequential(nn.ConvTranspose2d(256, 128, 5, 2, 0, bias=False), nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        self.layer3 = nn.Sequential(nn.ConvTranspose2d(128, 64, 5, 2, 0, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.layer4 = nn.Sequential(nn.ConvTranspose2d(64, nc, 5, 2, 0, bias=False), nn.BatchNorm2d(nc), nn.ReLU(inplace=True))

    def forward(self, x):
        output1 = self.layer1(x)
        output2 = self.layer2(output1)
        output3 = self.layer3(output2)
        output = self.layer4(output3)
        return output


class Encoder(nn.Module):
    def __init__(self, nc):
        super().__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(nc, 64, 5, 2, 0, bias=False), nn.BatchNorm2d(64), nn.LeakyReLU(0.2, inplace=True))
        self.layer2 = nn.Sequential(nn.Conv2d(64, 128, 5, 2, 0, bias=False), nn.BatchNorm2d(128), nn.LeakyReLU(0.2, inplace=True))
        self.layer3 = nn.Sequential(nn.Conv2d(128, 256, 5, 2, 0, bias=False), nn.BatchNorm2d(256), nn.LeakyReLU(0.2, inplace=True))
        self.layer4 = nn.Sequential(nn.Conv2d(256, 512, 5, 2, 0, bias=False), nn.BatchNorm2d(512), nn.LeakyReLU(0.2, inplace=True))

    def forward(self, x):
        output1 = self.layer1(x)
        output2 = self.layer2(output1)
        output3 = self.layer3(output2)
        output = self.layer4(output3)
        return output


##
class NetR(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.encoder = Encoder(args.nc)
        self.decoder = Decoder(args.nc)

    def forward(self, x):
        x_rec = self.decoder(self.encoder(x))
        return x_rec


##
class NetD(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.classifier = Encoder(args.nc)
        self.linear = nn.Sequential(nn.Linear(512, 1), nn.Sigmoid())

    def forward(self, x):
        output1 = self.classifier(x)
        output2 = output1.view(output1.size(0), -1)
        output = self.linear(output2).squeeze()
        return output
