import glob
import math
import os
import random
import shutil
import time
from pathlib import Path
from threading import Thread

import cv2
import numpy as np
import torch
from PIL import Image, ExifTags
from torch.utils.data import Dataset
from tqdm import tqdm
from datasets.VOC import VOCAnnotationTransform
import xml.etree.ElementTree as ET
def preprocess_img_boxes(image,boxes,input_ksize):
        '''
        resize image and bboxes 
        Returns
        image_paded: input_ksize  
        bboxes: [None,4]
        '''
        min_side, max_side    = input_ksize
        h,  w, _  = image.shape

        smallest_side = min(w,h)
        largest_side=max(w,h)
        scale=min_side/smallest_side
        if largest_side*scale>max_side:
            scale=max_side/largest_side
        nw, nh  = int(scale * w), int(scale * h)
        image_resized = cv2.resize(image, (nw, nh))

        pad_w=32-nw%32
        pad_h=32-nh%32

        image_paded = np.zeros(shape=[nh+pad_h, nw+pad_w, 3],dtype=np.uint8)
        image_paded[:nh, :nw, :] = image_resized

        if boxes is None:
            return image_paded
        else:
            boxes[:, [0, 2]] = boxes[:, [0, 2]] * scale 
            boxes[:, [1, 3]] = boxes[:, [1, 3]] * scale 
            return scale
        
def finet_collate(batch):
        boxes_list = []
        #imgs_list = []
        boxes = []
        classes = []
        classes_list = []
        # img, label, path, shapes = zip(*batch)  # transposed
        # for i, l in enumerate(label):
        #     l[:, 0] = i  # add target image index for build_targets()
        imgs_list,targets_list=zip(*batch)
        for i in range(len(imgs_list)):  
            #imgs_list.append(img_list)
            boxes_list.append(torch.cat((torch.full([len(targets_list[i]),1],i),np.delete(targets_list[i],0,1)),1))
            # classes_list.append(targets_list[i][:,1])
            #boxes_list.append(targets_list[i])        torch.cat((torch.tensor([1,1]).reshape(-1,1),np.delete(targets_list[i],0,1)),1)
        return imgs_list,boxes_list
#torch.full([len(targets_list[i]),1],i)
def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y
class Model_train_transform:
    def __init__(self, args, means, std):
        self.args = args
        self.mixup = 0.0
        self.batch_shapes = [[640,640]]
        self.img_size = 640
        self.augment = True
        self.mosaic = True
        self.rect = True
        self.mosaic_border = [-self.img_size // 2, -self.img_size // 2]
        self.degrees = 0.0
        self.translate = 0.5
        self.scale = 0.5
        self.shear = 0.0
        self.perspective = 0.0
        self.root = "./data/VHR_voc"
        self.ids = list()
        for line in open(os.path.join(self.root, "ImageSets", "Main/" + "train" + ".txt")):
            self.ids.append((self.root, line.strip()))
    def __call__(self, img, bboxs, label,index):
        # if self.image_weights:
        #     index = self.indices[index]
        boxes = xyxy2xywh(bboxs)
        labels = np.hstack((label.reshape(-1,1),boxes))
        #hyp = self.hyp
        #if self.mosaic:
            # Load mosaic
        
        img, labels = load_mosaic(self,img,labels,index)  # (608, 608, 3)  (n, 5) n是1-4这样可以保证数据都是有目标的。
            #shapes = None
        labels = labels.astype(np.float)
            # MixUp https://arxiv.org/pdf/1710.09412.pdf
            # if random.random() < 0.0:
            #     print("fkl")
            #     img2, labels2 = load_mosaic(self, random.randint(0, len(self.labels) - 1))
            #     r = np.random.beta(8.0, 8.0)  # mixup ratio, alpha=beta=8.0
            #     img = (img * r + img2 * (1 - r)).astype(np.uint8)
            #     labels = np.concatenate((labels, labels2), 0)

        # else:
        #     # Load image
            
        #     img, (h0, w0), (h, w) = load_image(self, index)
            
        #     # Letterbox
        #     shape = self.batch_shapes[self.batch[index]] if self.rect else self.img_size  # final letterboxed shape
            
        #     img, ratio, pad, _ = letterbox(img, shape, auto=False, scaleup=self.augment)
        #     shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling

        #     # Load labels
        #     labels = []
        #     x = self.labels[index]
        #     if x.size > 0:
        #         # Normalized xywh to pixel xyxy format
        #         labels = x.copy()
        #         labels[:, 1] = ratio[0] * w * (x[:, 1] - x[:, 3] / 2) + pad[0]  # pad width
        #         labels[:, 2] = ratio[1] * h * (x[:, 2] - x[:, 4] / 2) + pad[1]  # pad height
        #         labels[:, 3] = ratio[0] * w * (x[:, 1] + x[:, 3] / 2) + pad[0]
        #         labels[:, 4] = ratio[1] * h * (x[:, 2] + x[:, 4] / 2) + pad[1]

        # # print('augment', self.augment, self.mosaic)  # True
        
        #if self.augment:
            # Augment imagespace
            # if not self.mosaic: 
            #     #print("fjk") # not True, 没进去
            #     img, labels = random_perspective(img, labels,
            #                                      degrees=hyp['degrees'],
            #                                      translate=hyp['translate'],
            #                                      scale=hyp['scale'],
            #                                      shear=hyp['shear'],
            #                                      perspective=hyp['perspective'])
        
        # Augment colorspace
        augment_hsv(img, hgain=0.015, sgain=0.7, vgain=0.4)  # 色域变换：色调、饱和度和亮度

            # Apply cutouts
            # if random.random() < 0.9:
            #     labels = cutout(img, labels)

        nL = len(labels)  # number of labels
        # print(f'number of lablels: {nL}')
        if nL:
        
            labels[:, 1:5] = xyxy2xywh(labels[:, 1:5])  # convert xyxy to xywh
            labels[:, [2, 4]] =labels[:, [2, 4]] / img.shape[0]  # normalized height 0-1
            labels[:, [1, 3]] =labels[:, [1, 3]] / img.shape[1]  # normalized width 0-1

        if self.augment:
            # flip up-down
            if random.random() < 0.0:
                img = np.flipud(img)
                if nL:
                    labels[:, 2] = 1 - labels[:, 2]

            # flip left-right
            if random.random() < 0.5:
                img = np.fliplr(img)
                if nL:
                    labels[:, 1] = 1 - labels[:, 1]

        labels_out = torch.zeros((nL, 6))
        if nL:
            labels_out[:, 1:] = torch.from_numpy(labels)

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img)
        #path = self.img_files[index]
        # print(labels_out.shape)  [n, 6]  n是n个目标，数据增强之后可能有多个，6的前两个值是0，后4个值是xywh


        #return img, labels_out, self.img_files[index], shapes
        return img/255.0, labels_out
    
def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, roi=None):
    # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
    shape = img.shape[:2]  # current shape [height, width]  720, 1280

    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    # print(dw, dh, ratio)
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 64), np.mod(dh, 64)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    # print(dw, dh)
    dw /= 2  # divide padding into 2 sides
    dh /= 2
    # print(new_unpad)

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

    # roi
    if roi is not None:
        x1, x2 = roi[0] / shape[1] * new_unpad[0], roi[2] / shape[1] * new_unpad[0]  # convert from pixel to percet
        y1, y2 = roi[1] / shape[0] * new_unpad[1], roi[3] / shape[0] * new_unpad[1]
        img = img[int(y1):int(y2), int(x1):int(x2)]
        rest_h = img.shape[0] % 32
        rest_w = img.shape[1] % 32
        dh = 0 if rest_h == 0 else (32-rest_h)/2
        dw = 0 if rest_w == 0 else (32-rest_w)/2
        recover = [new_shape[0], new_unpad[1], int(x1)-dw, int(y1)-dh]
    else:
        recover = None

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh), recover
def load_mosaic(self, img_,labelss,index):
    # loads images in a mosaic
    labels_len =10975
    labels4 = []
    self.labels = VOCAnnotationTransform(detname = "VHR")
    s = self.img_size  # 608
    yc, xc = s, s  # mosaic center x, y
    # print(len(self.labels)) 1480个，是数据集的数目
    indices = [index] + [random.randint(0, labels_len - 1) for _ in range(3)]  # 3 additional image indices
    self.ann_path = os.path.join("%s", "Annotations", "%s.xml")
    
    for i, index in enumerate(indices):  # 4张图片，1张是出传入的，3张是随机的
        # Load image
        img_id = self.ids[index]
        label = ET.parse(self.ann_path % img_id).getroot()
        img, _, (h, w) = load_image(self, index)
        # print(img.shape, _, h, w, 'xxx')
        """
        zzd手动把图变成608 608
        new_image = Image.new('RGB', (max(w, h), max(w, h)),
                              (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)))
        dy = (max(w, h) - min(w, h))//2
        ar = h/w
        img = Image.fromarray(np.uint8(img))
        new_image.paste(img, (0, dy))
        # img.save('xxxxvv0.png')
        img = new_image
        # img.save('xxxxvv1.png')
        img = np.asarray(img)
        h = max(w, h)
        w = max(w, h)
        """


        # place img in img4
        if i == 0:  # top left
            img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
            x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
        elif i == 1:  # top right
            x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
            x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
        elif i == 2:  # bottom left
            x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, max(xc, w), min(y2a - y1a, h)
        elif i == 3:  # bottom right
            x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

        img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
        padw = x1a - x1b
        padh = y1a - y1b
        x = self.labels(label,w,h)
        x = np.array(x)
        # Labels
        #x = labels
        """
        修改标签，匹配608 608
        class, xc, yc, w, h
        x[:, 2] = x[:, 2]*ar + dy/608
        x[:, 4] = x[:, 4]*ar
        """


        labels = np.hstack((x[:,4:],x[:,:4]))
        # if x.size > 0:  # Normalized xywh to pixel xyxy format
        #     labels[:, 1] = w * (x[:, 1] - x[:, 3] / 2) + padw
        #     labels[:, 2] = h * (x[:, 2] - x[:, 4] / 2) + padh
        #     labels[:, 3] = w * (x[:, 1] + x[:, 3] / 2) + padw
        #     labels[:, 4] = h * (x[:, 2] + x[:, 4] / 2) + padh
        labels4.append(labels)

    # Concat/clip labels
    if len(labels4):
        labels4 = np.concatenate(labels4, 0)
        # np.clip(labels4[:, 1:] - s / 2, 0, s, out=labels4[:, 1:])  # use with center crop
        np.clip(labels4[:, 1:], 0, 2 * s, out=labels4[:, 1:])  # use with random_affine

        # Replicate
        # img4, labels4 = replicate(img4, labels4)

    # Augment
    # img4 = img4[s // 2: int(s * 1.5), s // 2:int(s * 1.5)]  # center crop (WARNING, requires box pruning)
    img4, labels4 = random_perspective(img4, labels4,
                                       degrees=self.degrees,
                                       translate=self.translate,
                                       scale=self.scale,
                                       shear=self.shear,
                                       perspective=self.perspective,
                                       border=self.mosaic_border)  # border to remove

    return img4, labels4
def xywh2xyxy(x):
	# Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
	y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
	y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
	y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
	y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
	y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
	return y
def random_perspective(img, targets=(), degrees=10, translate=.1, scale=.1, shear=10, perspective=0.0, border=(0, 0)):
    # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10))
    # targets = [cls, xyxy]

    height = img.shape[0] + border[0] * 2  # shape(h,w,c)
    width = img.shape[1] + border[1] * 2

    # Center
    C = np.eye(3)
    C[0, 2] = -img.shape[1] / 2  # x translation (pixels)
    C[1, 2] = -img.shape[0] / 2  # y translation (pixels)

    # Perspective
    P = np.eye(3)
    P[2, 0] = random.uniform(-perspective, perspective)  # x perspective (about y)
    P[2, 1] = random.uniform(-perspective, perspective)  # y perspective (about x)

    # Rotation and Scale
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
    s = random.uniform(1 - scale, 1 + scale)
    # s = 2 ** random.uniform(-scale, scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

    # Translation
    T = np.eye(3)
    T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width  # x translation (pixels)
    T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height  # y translation (pixels)

    # Combined rotation matrix
    M = T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT
    if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # image changed
        if perspective:
            img = cv2.warpPerspective(img, M, dsize=(width, height), borderValue=(114, 114, 114))
        else:  # affine
            img = cv2.warpAffine(img, M[:2], dsize=(width, height), borderValue=(114, 114, 114))

    # Visualize
    # import matplotlib.pyplot as plt
    # ax = plt.subplots(1, 2, figsize=(12, 6))[1].ravel()
    # ax[0].imshow(img[:, :, ::-1])  # base
    # ax[1].imshow(img2[:, :, ::-1])  # warped

    # Transform label coordinates
    n = len(targets)
    if n:
        # warp points
        xy = np.ones((n * 4, 3))
        xy[:, :2] = targets[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
        xy = xy @ M.T  # transform
        if perspective:
            xy = (xy[:, :2] / xy[:, 2:3]).reshape(n, 8)  # rescale
        else:  # affine
            xy = xy[:, :2].reshape(n, 8)

        # create new boxes
        x = xy[:, [0, 2, 4, 6]]
        y = xy[:, [1, 3, 5, 7]]
        xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

        # # apply angle-based reduction of bounding boxes
        # radians = a * math.pi / 180
        # reduction = max(abs(math.sin(radians)), abs(math.cos(radians))) ** 0.5
        # x = (xy[:, 2] + xy[:, 0]) / 2
        # y = (xy[:, 3] + xy[:, 1]) / 2
        # w = (xy[:, 2] - xy[:, 0]) * reduction
        # h = (xy[:, 3] - xy[:, 1]) * reduction
        # xy = np.concatenate((x - w / 2, y - h / 2, x + w / 2, y + h / 2)).reshape(4, n).T

        # clip boxes
        xy[:, [0, 2]] = xy[:, [0, 2]].clip(0, width)
        xy[:, [1, 3]] = xy[:, [1, 3]].clip(0, height)

        # filter candidates
        i = box_candidates(box1=targets[:, 1:5].T * s, box2=xy.T)
        targets = targets[i]
        targets[:, 1:5] = xy[i]

    return img, targets
def box_candidates(box1, box2, wh_thr=2, ar_thr=20, area_thr=0.2):  # box1(4,n), box2(4,n)
    # Compute candidate boxes: box1 before augment, box2 after augment, wh_thr (pixels), aspect_ratio_thr, area_ratio
    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    ar = np.maximum(w2 / (h2 + 1e-16), h2 / (w2 + 1e-16))  # aspect ratio
    return (w2 > wh_thr) & (h2 > wh_thr) & (w2 * h2 / (w1 * h1 + 1e-16) > area_thr) & (ar < ar_thr)  # candidates
def load_image(self, index):
    self.img_path = os.path.join("%s", "JPEGImages", "%s.jpg")
    img_id = self.ids[index]
    # loads 1 image from dataset, returns img, original hw, resized hw
    img = cv2.imread(self.img_path % img_id)
    #if img is None:  # not cached
    #path = self.img_files[index]
    #img = cv2.imread(path)  # BGR
    #assert img is not None, 'Image Not Found ' + path
    h0, w0 = img.shape[:2]  # orig hw
    r = self.img_size / max(h0, w0)  # resize image to img_size
    if r != 1:  # always resize down, only resize up if training with augmentation
        interp = cv2.INTER_AREA if r < 1 and not self.augment else cv2.INTER_LINEAR
        img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=interp)
    return img, (h0, w0), img.shape[:2]  # img, hw_original, hw_resized
    # else:
    #     return self.imgs[index], self.img_hw0[index], self.img_hw[index]  # img, hw_original, hw_resized


def augment_hsv(img, hgain=0.5, sgain=0.5, vgain=0.5):

    r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
    hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    dtype = img.dtype  # uint8

    x = np.arange(0, 256, dtype=np.int16)
    lut_hue = ((x * r[0]) % 180).astype(dtype)
    lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
    lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

    img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))).astype(dtype)
    cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)  # no return needed

    # Histogram equalization
    # if random.random() < 0.2:
    #     for i in range(3):
    #         img[:, :, i] = cv2.equalizeHist(img[:, :, i])
def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y

class Test_transform():
    def __init__(self, args):
        self.args = args
        self.augment = False
        self.stride = 32
        self.means=[0.485, 0.456, 0.406]
        self.std=[0.229, 0.224, 0.225]
        self.nb = 195
        self.rect = True
        self.img_size = 640
        self.root = "./data/VHR_voc"
        self.ids = list()
        for line in open(os.path.join(self.root, "ImageSets", "Main/" + "test" + ".txt")):
            self.ids.append((self.root, line.strip()))
        #self.test_tran = Model_train_transform(self.args, self.means, self.std)
    def __call__(self, img_, bboxs, label,index):
        # self.input_ksize = [800, 1024]
        
        # s = self.shapes  # wh
        # ar = s[:, 1] / s[:, 0]  # aspect ratio
        # irect = ar.argsort()
        # self.img_files = [self.img_files[i] for i in irect]
        # self.label_files = [self.label_files[i] for i in irect]
        # self.labels = [self.labels[i] for i in irect]
        # self.shapes = s[irect]  # wh
        # ar = ar[irect]
        # shapes = [[1, 1]] * self.nb
        # for i in range(self.nb):
        #     #ari = ar[bi == i]
        #     mini, maxi = 0, 194
        #     if maxi < 1:
        #         shapes[i] = [maxi, 1]
        #     elif mini > 1:
        #         shapes[i] = [1, 1 / mini]
        #     print(np.array(shapes))
        #     self.batch_shapes = np.ceil(np.array(shapes) * 640 / self.stride + pad).astype(np.int) * self.stride
        labels = np.hstack((label.reshape(-1,1),bboxs))
        labels = labels.astype(np.float)
        # img, boxes, scale =preprocess_img_boxes(img,bboxs,self.input_ksize)
        # #target = np.hstack((boxes,label.reshape(-1,1)))
        # img=transforms.ToTensor()(img)
        # #targets=torch.from_numpy(target)
        # #classes=torch.LongTensor(classes)
        # data = [(img,torch.tensor(np.hstack((boxes,label.reshape(-1,1)))).cpu())]
        # batch_imgs,batch_boxes,batch_classes = fcos_collate_fn(data)
        # #targets = [np.hstack((batch_boxes[0],batch_classes.reshape(-1,1))),scale]
        # #targets = np.array([[ batch_boxes[0][0][0], batch_boxes[0][0][1], batch_boxes[0][0][2], batch_boxes[0][0][3], batch_classes[0][0]]])
        # return batch_imgs, scale
    


        img, (h0, w0), (h, w) = load_image(self, index)
       
            # Letterbox
        if index<751:
            shape = [480,640] # final letterboxed shape
        else:
            shape=[640,480]
        img, ratio, pad, _ = letterbox(img, shape, auto=False, scaleup=False)
        scale =preprocess_img_boxes(img_,labels[:,1:],shape)
        shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling

        # Load labels
        #labels = []
        #x = self.labels[index]
        # if x.size > 0:
        #     # Normalized xywh to pixel xyxy format
        #     labels = x.copy()
        #     labels[:, 1] = ratio[0] * w * (x[:, 1] - x[:, 3] / 2) + pad[0]  # pad width
        #     labels[:, 2] = ratio[1] * h * (x[:, 2] - x[:, 4] / 2) + pad[1]  # pad height
        #     labels[:, 3] = ratio[0] * w * (x[:, 1] + x[:, 3] / 2) + pad[0]
        #     labels[:, 4] = ratio[1] * h * (x[:, 2] + x[:, 4] / 2) + pad[1]
        nL = len(labels)  # number of labels
        # print(f'number of lablels: {nL}')
        if nL:
            
            labels[:, 1:5] = xyxy2xywh(labels[:, 1:5])  # convert xyxy to xywh
            labels[:, [2, 4]] =labels[:, [2, 4]] / img.shape[0]  # normalized height 0-1
            labels[:, [1, 3]] =labels[:, [1, 3]]/ img.shape[1]  # normalized width 0-1
            labels_out = torch.zeros((nL, 6))
        if nL:
            labels_out[:, 1:] = torch.from_numpy(labels)
            img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
            img = np.ascontiguousarray(img)
            img = torch.from_numpy(img)
#            path = self.img_files[index]
        return img/255.0, [labels_out,img_.shape,scale]
    