# This file contains experimental modules

import numpy as np
import os
import torch
import torch.nn as nn
from pathlib import Path
import time
from .common import Conv, DWConv
# from FINet.utils.google_utils import attempt_download 修改

class CrossConv(nn.Module):
	# Cross Convolution Downsample
	def __init__(self, c1, c2, k=3, s=1, g=1, e=1.0, shortcut=False):
		# ch_in, ch_out, kernel, stride, groups, expansion, shortcut
		super(CrossConv, self).__init__()
		c_ = int(c2 * e)  # hidden channels
		self.cv1 = Conv(c1, c_, (1, k), (1, s))
		self.cv2 = Conv(c_, c2, (k, 1), (s, 1), g=g)
		self.add = shortcut and c1 == c2

	def forward(self, x):
		return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C3(nn.Module):
	# Cross Convolution CSP
	def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
		super(C3, self).__init__()
		c_ = int(c2 * e)  # hidden channels
		self.cv1 = Conv(c1, c_, 1, 1)
		self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
		self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
		self.cv4 = Conv(2 * c_, c2, 1, 1)
		self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
		self.act = nn.LeakyReLU(0.1, inplace=True)
		self.m = nn.Sequential(*[CrossConv(c_, c_, 3, 1, g, 1.0, shortcut) for _ in range(n)])

	def forward(self, x):
		y1 = self.cv3(self.m(self.cv1(x)))
		y2 = self.cv2(x)
		return self.cv4(self.act(self.bn(torch.cat((y1, y2), dim=1))))


class Sum(nn.Module):
	# Weighted sum of 2 or more layers https://arxiv.org/abs/1911.09070
	def __init__(self, n, weight=False):  # n: number of inputs
		super(Sum, self).__init__()
		self.weight = weight  # apply weights boolean
		self.iter = range(n - 1)  # iter object
		if weight:
			self.w = nn.Parameter(-torch.arange(1., n) / 2, requires_grad=True)  # layer weights

	def forward(self, x):
		y = x[0]  # no weight
		if self.weight:
			w = torch.sigmoid(self.w) * 2
			for i in self.iter:
				y = y + x[i + 1] * w[i]
		else:
			for i in self.iter:
				y = y + x[i + 1]
		return y


class GhostConv(nn.Module):
	# Ghost Convolution https://github.com/huawei-noah/ghostnet
	def __init__(self, c1, c2, k=1, s=1, g=1, act=True):  # ch_in, ch_out, kernel, stride, groups
		super(GhostConv, self).__init__()
		c_ = c2 // 2  # hidden channels
		self.cv1 = Conv(c1, c_, k, s, g, act)
		self.cv2 = Conv(c_, c_, 5, 1, c_, act)

	def forward(self, x):
		y = self.cv1(x)
		return torch.cat([y, self.cv2(y)], 1)


class GhostBottleneck(nn.Module):
	# Ghost Bottleneck https://github.com/huawei-noah/ghostnet
	def __init__(self, c1, c2, k, s):
		super(GhostBottleneck, self).__init__()
		c_ = c2 // 2
		self.conv = nn.Sequential(GhostConv(c1, c_, 1, 1),  # pw
								  DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),  # dw
								  GhostConv(c_, c2, 1, 1, act=False))  # pw-linear
		self.shortcut = nn.Sequential(DWConv(c1, c1, k, s, act=False),
									  Conv(c1, c2, 1, 1, act=False)) if s == 2 else nn.Identity()

	def forward(self, x):
		return self.conv(x) + self.shortcut(x)


class MixConv2d(nn.Module):
	# Mixed Depthwise Conv https://arxiv.org/abs/1907.09595
	def __init__(self, c1, c2, k=(1, 3), s=1, equal_ch=True):
		super(MixConv2d, self).__init__()
		groups = len(k)
		if equal_ch:  # equal c_ per group
			i = torch.linspace(0, groups - 1E-6, c2).floor()  # c2 indices
			c_ = [(i == g).sum() for g in range(groups)]  # intermediate channels
		else:  # equal weight.numel() per group
			b = [c2] + [0] * groups
			a = np.eye(groups + 1, groups, k=-1)
			a -= np.roll(a, 1, axis=1)
			a *= np.array(k) ** 2
			a[0] = 1
			c_ = np.linalg.lstsq(a, b, rcond=None)[0].round()  # solve for equal weight indices, ax = b

		self.m = nn.ModuleList([nn.Conv2d(c1, int(c_[g]), k[g], s, k[g] // 2, bias=False) for g in range(groups)])
		self.bn = nn.BatchNorm2d(c2)
		self.act = nn.LeakyReLU(0.1, inplace=True)

	def forward(self, x):
		return x + self.act(self.bn(torch.cat([m(x) for m in self.m], 1)))


class Ensemble(nn.ModuleList):
	# Ensemble of models
	def __init__(self):
		super(Ensemble, self).__init__()

	def forward(self, x, augment=False):
		y = []
		for module in self:
			y.append(module(x, augment)[0])
		# y = torch.stack(y).max(0)[0]  # max ensemble
		# y = torch.cat(y, 1)  # nms ensemble
		y = torch.stack(y).mean(0)  # mean ensemble
		return y, None  # inference, train output


def attempt_load(weights, map_location=None):
	# Loads an ensemble of models weights=[a,b,c] or a single model weights=[a] or weights=a
	model = Ensemble()
	for w in weights if isinstance(weights, list) else [weights]:
		attempt_download(w)
		model.append(torch.load(w, map_location=map_location)['model'].float().fuse().eval())  # load FP32 model

	if len(model) == 1:
		return model[-1]  # return model
	else:
		print('Ensemble created with %s\n' % weights)
		for k in ['names', 'stride']:
			setattr(model, k, getattr(model[-1], k))
		return model  # return ensemble
def attempt_download(weights):
    # Attempt to download pretrained weights if not found locally
    weights = weights.strip().replace("'", '')
    msg = weights + ' missing, please download trained logs and weights by "python download.py logs" or try downloading from https://github.com/zhangzhengde0225/FINet'

    r = 1  # return
    if len(weights) > 0 and not os.path.isfile(weights):
        d = {'yolov3-spp.pt': '1mM67oNw4fZoIOL1c8M3hHmj66d8e-ni_',  # yolov3-spp.yaml
             'yolov5s.pt': '1R5T6rIyy3lLwgFXNms8whc-387H0tMQO',  # yolov5s.yaml
             'yolov5m.pt': '1vobuEExpWQVpXExsJ2w-Mbf3HJjWkQJr',  # yolov5m.yaml
             'yolov5l.pt': '1hrlqD1Wdei7UT4OgT785BEk1JwnSvNEV',  # yolov5l.yaml
             'yolov5x.pt': '1mM8aZJlWTxOg7BZJvNUMrTnA2AbeCVzS',  # yolov5x.yaml
             }

        file = Path(weights).name
        if file in d:
            r = gdrive_download(id=d[file], name=weights)

        if not (r == 0 and os.path.exists(weights) and os.path.getsize(weights) > 1E6):  # weights exist and > 1MB
            os.remove(weights) if os.path.exists(weights) else None  # remove partial downloads
            s = "curl -L -o %s 'storage.googleapis.com/ultralytics/yolov5/ckpt/%s'" % (weights, file)
            r = os.system(s)  # execute, capture return values

            # Error check
            if not (r == 0 and os.path.exists(weights) and os.path.getsize(weights) > 1E6):  # weights exist and > 1MB
                os.remove(weights) if os.path.exists(weights) else None  # remove partial downloads
                raise Exception(msg)

def gdrive_download(id='1n_oKgR81BJtqk75b00eAjdv03qVCQn2f', name='coco128.zip'):
    # Downloads a file from Google Drive, accepting presented query
    # from utils.google_utils import *; gdrive_download()
    t = time.time()

    print('Downloading https://drive.google.com/uc?export=download&id=%s as %s... ' % (id, name), end='')
    os.remove(name) if os.path.exists(name) else None  # remove existing
    os.remove('cookie') if os.path.exists('cookie') else None

    # Attempt file download
    os.system("curl -c ./cookie -s -L \"drive.google.com/uc?export=download&id=%s\" > /dev/null" % id)
    if os.path.exists('cookie'):  # large file
        s = "curl -Lb ./cookie \"drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=%s\" -o %s" % (
            id, name)
    else:  # small file
        s = 'curl -s -L -o %s "drive.google.com/uc?export=download&id=%s"' % (name, id)
    r = os.system(s)  # execute, capture return values
    os.remove('cookie') if os.path.exists('cookie') else None

    # Error check
    if r != 0:
        os.remove(name) if os.path.exists(name) else None  # remove partial
        print('Download error ')  # raise Exception('Download error')
        return r

    # Unzip if archive
    if name.endswith('.zip'):
        print('unzipping... ', end='')
        os.system('unzip -q %s' % name)  # unzip
        os.remove(name)  # remove zip to free space

    print('Done (%.1fs)' % (time.time() - t))
    return r
