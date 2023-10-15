import libs.aug as transforms
import numpy as np
from datasets.VOC import VOC, VHR_means, VHR_std, SSDD_means, SSDD_std
import torch

class Add_train_transform:
    def __init__(self, args):
        self.args = args

    def __call__(self, img, boxes, labels):
        boxes = boxes.astype(np.float32)
        add_transform = eval("transforms.Compose([transforms.ToPercentCoords(), transforms.Resize({size}), transforms.SubtractMeans({detname}_means, {detname}_std)])".format(size=self.args.inp_imgsize, detname=self.args.detname))

        img, boxes, labels = add_transform(img, boxes, labels)
        img = img[:, :, (2, 1, 0)]
        img = torch.from_numpy(img).permute(2,0,1)
        return img, boxes, labels
