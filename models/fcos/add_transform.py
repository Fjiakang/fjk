import cv2
import math
import torch
import numpy as np
from torchvision import transforms


class Model_train_transform():
    def __init__(self, args,means, std):
        self.args = args
        
        self.mean=[0.485, 0.456, 0.406]
        self.std=[0.229, 0.224, 0.225]
    

    def __call__(self, img, bboxs, label,index):
        self.input_ksize = [512, 800]
        img, boxes, _ =preprocess_img_boxes(img,bboxs,self.input_ksize)
        target = np.hstack((boxes,label.reshape(-1,1)+1))
        img=transforms.ToTensor()(img)
        targets=torch.from_numpy(target)
        #classes=torch.LongTensor(classes)

        return img, targets

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
            return image_paded, boxes,scale
        


def fcos_collate_fn(data):
        mean=[0.485, 0.456, 0.406]
        std=[0.229, 0.224, 0.225]
        boxes_list = []
        boxes = []
        classes = []
        classes_list = []
        imgs_list,targets_list=zip(*data)
        for i in range(len(imgs_list)):  
             boxes_list.append(np.delete(targets_list[i],4,1))
             classes_list.append(targets_list[i][:,4])
            #boxes.append(boxes_list[i])
        # for i in range(len(targets_list[0])):
        #     boxes_list.append(np.delete(targets_list[0][i],-1).view(-1,4))
        #     classes_list.append(targets_list[0][i][4])
        assert len(imgs_list)==len(boxes_list)==len(classes_list)
        batch_size=len(boxes_list)
        pad_imgs_list=[]
        pad_boxes_list=[]
        pad_classes_list=[]

        h_list = [int(s.shape[1]) for s in imgs_list]
        w_list = [int(s.shape[2]) for s in imgs_list]
        max_h = np.array(h_list).max()
        max_w = np.array(w_list).max()
        for i in range(batch_size):
            img=imgs_list[i]
            pad_imgs_list.append(transforms.Normalize(mean, std,inplace=True)(torch.nn.functional.pad(img,(0,int(max_w-img.shape[2]),0,int(max_h-img.shape[1])),value=0.)))


        max_num=0
        for i in range(batch_size):
            n=boxes_list[i].shape[0]
            if n>max_num:max_num=n   
        for i in range(batch_size):
            pad_boxes_list.append(torch.nn.functional.pad(boxes_list[i],(0,0,0,max_num-boxes_list[i].shape[0]),value=-1))
            pad_classes_list.append(torch.nn.functional.pad(classes_list[i],(0,max_num-classes_list[i].shape[0]),value=-1))
        

        batch_boxes=torch.stack(pad_boxes_list)
        batch_classes=torch.stack(pad_classes_list)
        batch_imgs=torch.stack(pad_imgs_list)

        return batch_imgs,batch_boxes,batch_classes



class Test_transform():
    def __init__(self, args):
        self.args = args
        
        self.means=[0.485, 0.456, 0.406]
        self.std=[0.229, 0.224, 0.225]
        self.test_tran = Model_train_transform(self.args, self.means, self.std)
    def __call__(self, img, bboxs, label,index):
        self.input_ksize = [800, 1024]
        
        img, boxes, scale =preprocess_img_boxes(img,bboxs,self.input_ksize)
        #target = np.hstack((boxes,label.reshape(-1,1)))
        img=transforms.ToTensor()(img)
        #targets=torch.from_numpy(target)
        #classes=torch.LongTensor(classes)
        data = [(img,torch.tensor(np.hstack((boxes,label.reshape(-1,1)))).cpu())]
        batch_imgs,batch_boxes,batch_classes = fcos_collate_fn(data)
        #targets = [np.hstack((batch_boxes[0],batch_classes.reshape(-1,1))),scale]
        #targets = np.array([[ batch_boxes[0][0][0], batch_boxes[0][0][1], batch_boxes[0][0][2], batch_boxes[0][0][3], batch_classes[0][0]]])
        return batch_imgs, scale