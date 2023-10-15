import torch
import torch.nn as nn
from math import sqrt as sqrt
from torch.autograd import Variable
from itertools import product as product

from .ssd_head import ssd_head
from .ssd_backbone import vgg_backbone


class ssd_net(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.backbone = vgg_backbone(args)
        self.head = ssd_head(args)
        Dsets = args.det_dataset
        size = args.inp_imgsize
        exec("from .config import {}_{} as config".format(Dsets, size))
        exec("self.priorbox = PriorBox(config)")
        with torch.no_grad():
            self.priors = Variable(self.priorbox.forward())

    def forward(self, x):
        x = self.backbone(x)
        outputs = self.head(x)
        return outputs, self.priors


class PriorBox:
    """Compute priorbox coordinates in center-offset form for each source
    feature map.
    """

    def __init__(self, config):
        super(PriorBox, self).__init__()
        self.image_size = config["min_dim"]
        # number of priors for feature map location (either 4 or 6)
        self.num_priors = len(config["aspect_ratios"])
        self.variance = config["variance"] or [0.1]
        self.feature_maps = config["feature_maps"]
        self.min_sizes = config["min_sizes"]
        self.max_sizes = config["max_sizes"]
        self.steps = config["steps"]
        self.aspect_ratios = config["aspect_ratios"]
        self.clip = config["clip"]
        for v in self.variance:
            if v <= 0:
                raise ValueError("Variances must be greater than 0")

    def forward(self):
        mean = []
        for k, f in enumerate(self.feature_maps):
            for i, j in product(range(f), repeat=2):
                f_k = self.image_size / self.steps[k]
                # unit center x,y
                cx = (j + 0.5) / f_k
                cy = (i + 0.5) / f_k

                # aspect_ratio: 1
                # rel size: min_size
                s_k = self.min_sizes[k] / self.image_size
                mean += [cx, cy, s_k, s_k]

                # aspect_ratio: 1
                # rel size: sqrt(s_k * s_(k+1))
                s_k_prime = sqrt(s_k * (self.max_sizes[k] / self.image_size))
                mean += [cx, cy, s_k_prime, s_k_prime]

                # rest of aspect ratios
                for ar in self.aspect_ratios[k]:
                    mean += [cx, cy, s_k * sqrt(ar), s_k / sqrt(ar)]
                    mean += [cx, cy, s_k / sqrt(ar), s_k * sqrt(ar)]
        # back to torch land
        output = torch.Tensor(mean).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output
