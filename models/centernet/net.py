import torch.nn as nn
from .net_blocks import *


class exkp(nn.Module):
    def __init__(
        self,
        n,
        nstack,
        dims,
        modules,
        heads,
        pre=None,
        cnv_dim=256,
        make_tl_layer=None,
        make_br_layer=None,
        make_cnv_layer=make_cnv_layer,
        make_heat_layer=make_kp_layer,
        make_tag_layer=make_kp_layer,
        make_regr_layer=make_kp_layer,
        make_up_layer=make_layer,
        make_low_layer=make_layer,
        make_hg_layer=make_layer,
        make_hg_layer_revr=make_layer_revr,
        make_pool_layer=make_pool_layer,
        make_unpool_layer=make_unpool_layer,
        make_merge_layer=make_merge_layer,
        make_inter_layer=make_inter_layer,
        kp_layer=residual,
    ):
        super(exkp, self).__init__()

        self.nstack = nstack
        self.heads = heads

        curr_dim = dims[0]

        self.pre = nn.Sequential(convolution(7, 3, 128, stride=2), residual(3, 128, 256, stride=2)) if pre is None else pre

        self.kps = nn.ModuleList(
            [
                kp_module(
                    n,
                    dims,
                    modules,
                    layer=kp_layer,
                    make_up_layer=make_up_layer,
                    make_low_layer=make_low_layer,
                    make_hg_layer=make_hg_layer,
                    make_hg_layer_revr=make_hg_layer_revr,
                    make_pool_layer=make_pool_layer,
                    make_unpool_layer=make_unpool_layer,
                    make_merge_layer=make_merge_layer,
                )
                for _ in range(nstack)
            ]
        )
        self.cnvs = nn.ModuleList([make_cnv_layer(curr_dim, cnv_dim) for _ in range(nstack)])

        self.inters = nn.ModuleList([make_inter_layer(curr_dim) for _ in range(nstack - 1)])

        self.inters_ = nn.ModuleList([nn.Sequential(nn.Conv2d(curr_dim, curr_dim, (1, 1), bias=False), nn.BatchNorm2d(curr_dim)) for _ in range(nstack - 1)])
        self.cnvs_ = nn.ModuleList([nn.Sequential(nn.Conv2d(cnv_dim, curr_dim, (1, 1), bias=False), nn.BatchNorm2d(curr_dim)) for _ in range(nstack - 1)])

        ## keypoint heatmaps
        for head in heads.keys():
            if "hm" in head:
                module = nn.ModuleList([make_heat_layer(cnv_dim, curr_dim, heads[head]) for _ in range(nstack)])
                self.__setattr__(head, module)
                for heat in self.__getattr__(head):
                    heat[-1].bias.data.fill_(-2.19)
            else:
                module = nn.ModuleList([make_regr_layer(cnv_dim, curr_dim, heads[head]) for _ in range(nstack)])
                self.__setattr__(head, module)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, image):
        # print('image shape', image.shape)
        inter = self.pre(image)
        outs = []

        for ind in range(self.nstack):
            kp_, cnv_ = self.kps[ind], self.cnvs[ind]
            kp = kp_(inter)
            cnv = cnv_(kp)

            out = {}
            for head in self.heads:
                layer = self.__getattr__(head)[ind]
                y = layer(cnv)
                out[head] = y

            outs.append(out)
            if ind < self.nstack - 1:
                inter = self.inters_[ind](inter) + self.cnvs_[ind](cnv)
                inter = self.relu(inter)
                inter = self.inters[ind](inter)
        return outs


def make_hg_layer(kernel, dim0, dim1, mod, layer=convolution, **kwargs):
    layers = [layer(kernel, dim0, dim1, stride=2)]
    layers += [layer(kernel, dim1, dim1) for _ in range(mod - 1)]
    return nn.Sequential(*layers)


class HourglassNet(exkp):
    def __init__(self, heads, num_stacks=2):
        n = 5
        dims = [256, 256, 384, 384, 384, 512]
        modules = [2, 2, 2, 2, 2, 4]

        super(HourglassNet, self).__init__(n, num_stacks, dims, modules, heads, make_tl_layer=None, make_br_layer=None, make_pool_layer=make_pool_layer, make_hg_layer=make_hg_layer, kp_layer=residual, cnv_dim=256)


def get_large_hourglass_net(num_layers, heads, head_conv):
    model = HourglassNet(heads, 2)
    return model
