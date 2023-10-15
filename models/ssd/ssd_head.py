import torch
import torch.nn as nn


class ssd_head(nn.Module):
    def __init__(self, args):
        super().__init__()
        size = args.inp_imgsize
        num_classes = args.num_classes + 1
        head = multibox(mbox[str(size)], output_channels[str(size)], num_classes)
        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])

    def forward(self, features):
        loc = list()
        conf = list()

        for (x, l, c) in zip(features, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)

        return {"loc": loc, "conf": conf}


output_channels = {
    "300": [512, 1024, 512, 256, 256, 256],
    "512": [512, 1024, 512, 256, 256, 256, 256],
}
mbox = {
    "300": [4, 6, 6, 6, 4, 4],
    "512": [4, 6, 6, 6, 6, 4, 4],
}  # number of boxes per feature map location


def multibox(cfg, cfg_outc, num_classes):
    loc_layers = []
    conf_layers = []
    for k, v in enumerate(cfg_outc):
        loc_layers += [nn.Conv2d(v, cfg[k] * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(v, cfg[k] * num_classes, kernel_size=3, padding=1)]
    return (loc_layers, conf_layers)
