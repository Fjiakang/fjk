import torch
import torch.nn as nn


from .blocks import BasicRFB, BasicConv, BasicRFB_a


class vgg_backbone(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.num_classes = args.num_classes +1
        size = args.inp_imgsize
        self.vgg = nn.ModuleList(vgg_base(base[str(size)], 3))
        self.Norm = BasicRFB_a(512, 512, stride=1, scale=1.0)
        self.extras = nn.ModuleList(add_extras(extras[str(size)], 1024, size))
        self.indicator = [3, 5][size != 300]

    def forward(self, x):
        features = list()
        for k in range(23):
            x = self.vgg[k](x)
        s = self.Norm(x)
        features.append(s)

        for k in range(23, len(self.vgg)):
            x = self.vgg[k](x) 
        for k, v in enumerate(self.extras):
            x = v(x)
            if k < self.indicator or k % 2 == 0:
                features.append(x)

        return features


base = {
    "300": [64, 64, "M", 128, 128, "M", 256, 256, 256, "C", 512, 512, 512, "M", 512, 512, 512],
    "512": [64, 64, "M", 128, 128, "M", 256, 256, 256, "C", 512, 512, 512, "M", 512, 512, 512],
}
extras = {
    "300": [1024, "S", 512, "S", 256],
    "512": [1024, "S", 512, "S", 256, "S", 256, "S", 256],
}


def vgg_base(cfg, i, batch_norm=False):
    layers = []
    in_channels = i
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == "C":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [pool5, conv6, nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
    return layers


def add_extras(cfg, i, size, batch_norm=False):
    # Extra layers added to VGG for feature scaling
    layers = []
    in_channels = i
    flag = False
    for k, v in enumerate(cfg):
        if in_channels != "S":
            if v == "S":
                if in_channels == 256 and size == 512:
                    layers += [BasicRFB(in_channels, cfg[k + 1], stride=2, scale=1.0, visual=1)]
                else:
                    layers += [BasicRFB(in_channels, cfg[k + 1], stride=2, scale=1.0, visual=2)]
            else:
                layers += [BasicRFB(in_channels, v, scale=1.0, visual=2)]
        in_channels = v
    if size == 512:
        layers += [BasicConv(256, 128, kernel_size=1, stride=1)]
        layers += [BasicConv(128, 256, kernel_size=4, stride=1, padding=1)]
    elif size == 300:
        layers += [BasicConv(256, 128, kernel_size=1, stride=1)]
        layers += [BasicConv(128, 256, kernel_size=3, stride=1)]
        layers += [BasicConv(256, 128, kernel_size=1, stride=1)]
        layers += [BasicConv(128, 256, kernel_size=3, stride=1)]
    else:
        print("Error: Sorry only RFBNet300 and RFBNet512 are supported!")
        return
    return layers
