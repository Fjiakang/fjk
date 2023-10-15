import cv2
import torch
import numpy as np
from torchvision import transforms
#import libs.aug as transforms
from datasets.VOC import VOC, VHR_means, VHR_std, SSDD_means, SSDD_std
from datasets.COCO import COCO_
import skimage.io
import skimage.transform
import random
from torch.utils.data.sampler import Sampler
import skimage.color
import skimage
from PIL import Image
class Add_train_transform:
    def __init__(self, args):
        self.args = args
        self.scale = 0.038
        # self.min_side = 608
        # self.max_side = 1024

    def __call__(self, image, box, label):
        # height, width = image.shape[0:2]#279,322 (279,322,3)
        # scale = self.scale
        boxes = box.astype(np.float32)
        # #image = cv2.resize(image, (round(height * scale), round(width * scale)))
        # new_height, new_width, cn = image.shape 
        add_transform = eval("transforms.Compose([transforms.ToPercentCoords(), transforms.Resize({size}), transforms.SubtractMeans({detname}_means, {detname}_std)])".format(size=self.args.inp_imgsize, detname=self.args.detname))
        img, _, labels = add_transform(image, boxes, label)
        # pad_w = 32 - new_height % 32
        # pad_h = 32 - new_width % 32

        # new_image = np.zeros((new_height + pad_w, new_width + pad_h, cn)).astype(np.float32)
        #new_image[:height, :width, :] = image.astype(np.float32)


        return img, box.astype(np.float32), labels
# class Add_train_transform:
#     def __init__(self, args):
#         self.args = args

#     def __call__(self, img, boxes, labels):
#         boxes = boxes.astype(np.float32)
#         add_transform = eval("transforms.Compose([transforms.ToPercentCoords(), transforms.Resize({size}), transforms.SubtractMeans({detname}_means, {detname}_std)])".format(size=self.args.inp_imgsize, detname=self.args.detname))

#         img, boxes, labels = add_transform(img, boxes, labels)
#         return img, boxes, labels

class Test_transform:
    def __init__(self, args):
        self.args = args

    #def __call__(self, image, boxes, label):
        # height = image.shape[0]
        # width= image.shape[1]
        # cn = image.shape[2]
        # scale = 1

        # pad_w = 32 - height % 32
        # pad_h = 32 - width % 32

        # new_image = np.zeros((height + pad_w, width + pad_h, cn)).astype(np.float32)
        # new_image[:height, :width, :] = image.astype(np.float32)

        # boxes *= scale
        # boxes = boxes.astype(np.float32)
        # add_transform = eval("transforms.Compose([transforms.ToPercentCoords(), transforms.Resize({size}), transforms.SubtractMeans({detname}_means, {detname}_std)])".format(size=self.args.inp_imgsize, detname=self.args.detname))

        # new_img, boxes, labels = add_transform(image, boxes, label)
    def __call__(self, img, bboxs, label,index):
        img = img.astype(np.float32)/255.0
        target = np.hstack((bboxs,label.reshape(-1,1)))
        target = target.astype(float)
        sample = {"img":img,"annot":target}
        self.transform = transforms.Compose([Normalizer(),  Resizer()])
        sample = self.transform(sample)
        # img, target_info = sample['img'], sample['annot']
        # return img, np.array(target_info)
        img = sample['img']
        target_info =  [np.array(sample['annot']), sample['scale']]
        return img, target_info
        
class Normalizer(object):

    def __init__(self):
        self.mean = np.array([[[0.485, 0.456, 0.406]]])
        self.std = np.array([[[0.229, 0.224, 0.225]]])

    def __call__(self, sample):

        image, annots = sample['img'], sample['annot']

        return {'img':((image.astype(np.float32)-self.mean)/self.std), 'annot': annots}
class Augmenter(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample, flip_x=0.5):

        if np.random.rand() < flip_x:
            image, annots = sample['img'], sample['annot']
            image = image[:, ::-1, :]

            rows, cols, channels = image.shape

            x1 = annots[:, 0].copy()
            x2 = annots[:, 2].copy()
            
            x_tmp = x1.copy()

            annots[:, 0] = cols - x2
            annots[:, 2] = cols - x_tmp

            sample = {'img': image, 'annot': annots}

        return sample
class Resizer(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample, min_side=608, max_side=1024):
        image, annots = sample['img'], sample['annot']

        rows, cols, cns = image.shape

        smallest_side = min(rows, cols)

        # rescale the image so the smallest side is min_side
        scale = min_side / smallest_side

        # check if the largest side is now greater than max_side, which can happen
        # when images have a large aspect ratio
        largest_side = max(rows, cols)

        if largest_side * scale > max_side:
            scale = max_side / largest_side

        # resize the image with the computed scale
        image = skimage.transform.resize(image, (int(round(rows*scale)), int(round((cols*scale)))))
        rows, cols, cns = image.shape

        pad_w = 32 - rows%32
        pad_h = 32 - cols%32

        new_image = np.zeros((rows + pad_w, cols + pad_h, cns)).astype(np.float32)
        new_image[:rows, :cols, :] = image.astype(np.float32)

        annots[:, :4] *= scale

        return {'img': torch.from_numpy(new_image), 'annot': torch.from_numpy(annots), 'scale': scale}
class Model_train_transform:
    def __init__(self, args, means, std):
        self.args = args
        
    def __call__(self, img, bboxs, label,index):
        img = img.astype(np.float32)/255.0
        target =  np.hstack((bboxs,label.reshape(-1,1)))
        target = target.astype(float)
        sample = {"img":img,"annot":target}
        self.transform = transforms.Compose([Normalizer(), Augmenter(), Resizer()])
        sample = self.transform(sample)
        img = sample['img'].cuda()
        target_info =  [np.array(sample['annot']), sample['scale']]
        return img, target_info
class AspectRatioBasedSampler(Sampler):

    def __init__(self, data_source, batch_size, drop_last):
        self.data_source = data_source
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.groups = self.group_images()

    def __iter__(self):
        random.shuffle(self.groups)
        for group in self.groups:
            yield group

    def __len__(self):
        if self.drop_last:
            return len(self.data_source) // self.batch_size
        else:
            return (len(self.data_source) + self.batch_size - 1) // self.batch_size

    def group_images(self):
        # determine the order of the images
        order = list(range(len(self.data_source)))
        order.sort(key=lambda x: self.data_source.image_aspect_ratio(x))

        # divide into groups, one group = one batch
        return [[order[x % len(order)] for x in range(i, i + self.batch_size)] for i in range(0, len(order), self.batch_size)]
