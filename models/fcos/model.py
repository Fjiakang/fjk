import os
from tkinter import N
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
from .fcos_net import fcos_net
import math,time
from torchvision.ops import nms
from datasets.voc_eval import evaluate_detections
from torch.autograd import Variable
from torch.autograd import Function
from libs.Visualizer import Visualizer
from models.base_model import BaseModel
from libs.utils.boxes_utils import batched_nms, box_nms
from .loss import LOSS, coords_fmap2orig
from tqdm import tqdm

#python  main.py --det_model fcos --task train --detname VHR --det_dataset VOC --num_classes 10   --batch_size 2 --lr 1e-4 --num_epoch 30


class fcos(BaseModel):
    def __init__(self, parser):
        super().__init__(parser)
       # parser.add_args(arguments())
        self.args = parser.get_args()
        torch.set_default_tensor_type("torch.cuda.FloatTensor") if not self.device == "cpu" else torch.set_default_tensor_type("torch.FloatTensor")

        self.save_folder = os.path.join(self.args.outf, self.args.det_dataset, self.args.detname, self.args.det_model, "basize_{}epoch_{}lr_{}".format(self.args.batch_size, self.args.num_epoch, self.args.lr))
        self.train_folder = os.path.join(self.save_folder, "train")
        self.test_folder = os.path.join(self.save_folder, "test")
        self.val_folder = os.path.join(self.args.load_weights_dir.split("train")[0], "val")

        self.vis = Visualizer(self.args, self.save_folder)

        self.epoch = self.args.num_epoch
        self.num_classes = self.args.num_classes+1

        self.model = fcos_net(self.args)
        self.model = self.model.cuda()
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr = self.args.lr) if self.args.task == "train" else None
        self.net_dict = {"name": ["fcos"], "network": [self.model]}
        self.best_trigger = False
        self.final_trigger = False
        self.wegiter_trigger = False
        self.global_steps = 1
        self.criterion = LOSS()

  
    def train(self, train_loader, testset):

#        self.load_weights(self.args.load_weights_dir) if self.args.load_weights_dir != "" else self.vgg_weights()
        self.model.train()
        print("Finished loading model!\nStart training")

        best = float(0)
        for epo in range(1, self.epoch + 1):
            loss_train = self.train_epoch(epo, train_loader)
            # if epo in self.lr_step:
            #     self.optimizer.param_groups[0]["lr"] *= 0.1
            print(self.global_steps)
            self.best_trigger = True if loss_train < best else False
            best = loss_train if loss_train < best else best
            self.final_trigger = True if epo == self.epoch else False
            self.save_weights(epo, self.train_folder)

           #if epo%20==0:
            self.inference(testset, from_train=epo)
        self.model.train()

    def train_epoch(self, epo, train_loader):
        self.model.train()
        epoch_size = len(train_loader) // self.args.batch_size
        
        loss_mean = float(0)
        num_iters = len(train_loader)
        for iteration,data in tqdm(enumerate(train_loader)):

            batch_imgs,batch_boxes,batch_classes=data
            batch_imgs=batch_imgs.cuda()
            batch_boxes=batch_boxes.cuda()
            batch_classes=batch_classes.cuda()

            lr = lr_func(self,epoch_size)
            for param in self.optimizer.param_groups:
                param['lr']=lr
            self.global_steps+=1
            
            #start_time=time.time()

            self.optimizer.zero_grad()
            out, targets = self.model([batch_imgs,batch_boxes,batch_classes])
            cls_loss,cnt_loss,reg_loss,total_loss = self.criterion(out, targets)
            #loss=losses[-1]
            #loss_mean += total_loss
            total_loss.backward()
            self.optimizer.step()

            #end_time=time.time()
           # cost_time=int((end_time-start_time)*1000)
            self.save_loggin_print(epo, { "cls_loss": cls_loss, "cnt_loss": cnt_loss, "reg_loss": reg_loss}, iteration, num_iters)

        return total_loss

    def test(self, testset):
        self.load_weights(self.args.load_weights_dir)

        print("Finished loading model!")
        
        self.inference(testset)

    def inference(self, testset,  fold=None, from_train=-1):
        self.model.eval()
        fold = self.val_folder if fold == None else fold
        all_boxes = [[[] for _ in range(len(testset))] for _ in range(self.num_classes)]
       
        for i in tqdm(range(len(testset))):
            with torch.no_grad():
                data = testset[i]
                scale = data[1]
                img = data[0].cuda()
            
                out=self.model(img)
                self.detection_head=DetectHead().cuda()
                self.clip_boxes=ClipBoxes().cuda()
                scores,classes,boxes_= self.detection_head(out)
                boxes=self.clip_boxes(img,boxes_)
                boxes = boxes/scale
                for j in range(self.num_classes):
                    inds = np.where(classes[0].cpu().numpy()==j)[0]
                    if len(inds) == 0:
                        all_boxes[j][i] = np.empty([0, 5], dtype=np.float32)
                        continue
                    c_dets = np.hstack((boxes[0][inds].cpu().detach().numpy(),scores[0][inds].cpu().detach().numpy().reshape(-1,1)))
                    all_boxes[j][i] = c_dets




        if self.args.test_save:
                img_id, annotation = testset.pull_anno(i)
                test_result_file = os.path.join(self.save_folder, "test_result.txt")
                self.vis.write_gt(test_result_file, img_id, annotation)
                eval("from datasets.VOC import {}_CLASSES as CLASSES".format(self.args.detname))
                self.vis.write_bb(test_result_file, detections, CLASSES)
        print("Evaluting detections...")
        if self.args.det_dataset == 'VOC':
            if from_train != -1:
                #print(all_boxes[1])            
                evaluate_detections(self.args, from_train, all_boxes, self.save_folder, testset)
            else:
                evaluate_detections(self.args, "test", all_boxes, fold, testset)
        else:
            
            testset.run_eval( testset, all_boxes, self.save_folder)


class DetectHead(nn.Module):     
    def __init__(self):
        super().__init__()
        self.score_threshold=0.3
        self.nms_iou_threshold=0.2
        self.max_detection_boxes_num=150
        self.strides=[8,16,32,64,128]
    def forward(self,inputs):
        '''
        inputs  list [cls_logits,cnt_logits,reg_preds]  
        cls_logits  list contains five [batch_size,class_num,h,w]  
        cnt_logits  list contains five [batch_size,1,h,w]  
        reg_preds   list contains five [batch_size,4,h,w] 
        '''
        cls_logits,coords=self._reshape_cat_out(inputs[0],self.strides)#[batch_size,sum(_h*_w),class_num]
        cnt_logits,_=self._reshape_cat_out(inputs[1],self.strides)#[batch_size,sum(_h*_w),1]
        reg_preds,_=self._reshape_cat_out(inputs[2],self.strides)#[batch_size,sum(_h*_w),4]

        cls_preds=cls_logits.sigmoid_()
        cnt_preds=cnt_logits.sigmoid_()

        cls_scores,cls_classes=torch.max(cls_preds,dim=-1)#[batch_size,sum(_h*_w)]
        
        cls_scores=cls_scores*(cnt_preds.squeeze(dim=-1))#[batch_size,sum(_h*_w)]
        cls_classes=cls_classes+1#[batch_size,sum(_h*_w)]

        boxes=self._coords2boxes(coords,reg_preds)#[batch_size,sum(_h*_w),4]

        #select topk
        max_num=min(self.max_detection_boxes_num,cls_scores.shape[-1])
        topk_ind=torch.topk(cls_scores,max_num,dim=-1,largest=True,sorted=True)[1]#[batch_size,max_num]
        _cls_scores=[]
        _cls_classes=[]
        _boxes=[]
        for batch in range(cls_scores.shape[0]):
            _cls_scores.append(cls_scores[batch][topk_ind[batch]])#[max_num]
            _cls_classes.append(cls_classes[batch][topk_ind[batch]])#[max_num]
            _boxes.append(boxes[batch][topk_ind[batch]])#[max_num,4]
        cls_scores_topk=torch.stack(_cls_scores,dim=0)#[batch_size,max_num]
        cls_classes_topk=torch.stack(_cls_classes,dim=0)#[batch_size,max_num]
        boxes_topk=torch.stack(_boxes,dim=0)#[batch_size,max_num,4]
        assert boxes_topk.shape[-1]==4
        return self._post_process([cls_scores_topk,cls_classes_topk,boxes_topk])

    def _post_process(self,preds_topk):
        '''
        cls_scores_topk [batch_size,max_num]
        cls_classes_topk [batch_size,max_num]
        boxes_topk [batch_size,max_num,4]
        '''
        _cls_scores_post=[]
        _cls_classes_post=[]
        _boxes_post=[]
        cls_scores_topk,cls_classes_topk,boxes_topk=preds_topk
        for batch in range(cls_classes_topk.shape[0]):
            mask=cls_scores_topk[batch]>=self.score_threshold
            _cls_scores_b=cls_scores_topk[batch][mask]#[?]
            _cls_classes_b=cls_classes_topk[batch][mask]#[?]
            _boxes_b=boxes_topk[batch][mask]#[?,4]
            nms_ind=self.batched_nms(_boxes_b,_cls_scores_b,_cls_classes_b,self.nms_iou_threshold)
            _cls_scores_post.append(_cls_scores_b[nms_ind])
            _cls_classes_post.append(_cls_classes_b[nms_ind])
            _boxes_post.append(_boxes_b[nms_ind])
        scores,classes,boxes= torch.stack(_cls_scores_post,dim=0),torch.stack(_cls_classes_post,dim=0),torch.stack(_boxes_post,dim=0)
        
        return scores,classes,boxes
    
    @staticmethod
    def box_nms(boxes,scores,thr):
        '''
        boxes: [?,4]
        scores: [?]
        '''
        if boxes.shape[0]==0:
            return torch.zeros(0,device=boxes.device).long()
        assert boxes.shape[-1]==4
        x1,y1,x2,y2=boxes[:,0],boxes[:,1],boxes[:,2],boxes[:,3]
        areas=(x2-x1+1)*(y2-y1+1)
        order=scores.sort(0,descending=True)[1]
        keep=[]
        while order.numel()>0:
            if order.numel()==1:
                i=order.item()
                keep.append(i)
                break
            else:
                i=order[0].item()
                keep.append(i)
            
            xmin=x1[order[1:]].clamp(min=float(x1[i]))
            ymin=y1[order[1:]].clamp(min=float(y1[i]))
            xmax=x2[order[1:]].clamp(max=float(x2[i]))
            ymax=y2[order[1:]].clamp(max=float(y2[i]))
            inter=(xmax-xmin).clamp(min=0)*(ymax-ymin).clamp(min=0)
            iou=inter/(areas[i]+areas[order[1:]]-inter)
            idx=(iou<=thr).nonzero().squeeze()
            if idx.numel()==0:
                break
            order=order[idx+1]
        return torch.LongTensor(keep)

    def batched_nms(self,boxes, scores, idxs, iou_threshold):
        
        if boxes.numel() == 0:
            return torch.empty((0,), dtype=torch.int64, device=boxes.device)
        # strategy: in order to perform NMS independently per class.
        # we add an offset to all the boxes. The offset is dependent
        # only on the class idx, and is large enough so that boxes
        # from different classes do not overlap
        max_coordinate = boxes.max()
        offsets = idxs.to(boxes) * (max_coordinate + 1)
        boxes_for_nms = boxes + offsets[:, None]
        keep = self.box_nms(boxes_for_nms, scores, iou_threshold)
        return keep

    def _coords2boxes(self,coords,offsets):
        '''
        Args
        coords [sum(_h*_w),2]
        offsets [batch_size,sum(_h*_w),4] ltrb
        '''
        x1y1=coords[None,:,:]-offsets[...,:2]
        x2y2=coords[None,:,:]+offsets[...,2:]#[batch_size,sum(_h*_w),2]
        boxes=torch.cat([x1y1,x2y2],dim=-1)#[batch_size,sum(_h*_w),4]
        return boxes


    def _reshape_cat_out(self,inputs,strides):
        '''
        Args
        inputs: list contains five [batch_size,c,_h,_w]
        Returns
        out [batch_size,sum(_h*_w),c]
        coords [sum(_h*_w),2]
        '''
        batch_size=inputs[0].shape[0]
        c=inputs[0].shape[1]
        out=[]
        coords=[]
        for pred,stride in zip(inputs,strides):
            pred=pred.permute(0,2,3,1)
            coord=coords_fmap2orig(pred,stride).to(device=pred.device)
            pred=torch.reshape(pred,[batch_size,-1,c])
            out.append(pred)
            coords.append(coord)
        return torch.cat(out,dim=1),torch.cat(coords,dim=0)
    
class ClipBoxes(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,batch_imgs,batch_boxes):
        batch_boxes=batch_boxes.clamp_(min=0)
        h,w=batch_imgs.shape[2:]
        batch_boxes[...,[0,2]]=batch_boxes[...,[0,2]].clamp_(max=w-1)
        batch_boxes[...,[1,3]]=batch_boxes[...,[1,3]].clamp_(max=h-1)
        return batch_boxes

def lr_func(self,epoch_size):
    lr_init=5e-5
    lr_end=1e-6
    
    warmpup_steps_RATIO=0.12
    total_steps=epoch_size*self.args.num_epoch
    warmpup_steps=total_steps*warmpup_steps_RATIO
    
    if self.global_steps<warmpup_steps:
        lr=self.global_steps/warmpup_steps*lr_init
    else:
        lr=lr_end+0.5*(lr_init-lr_end)*(
            (1+math.cos((self.global_steps-warmpup_steps)/(total_steps-warmpup_steps)*math.pi))
        )
    return float(lr)


