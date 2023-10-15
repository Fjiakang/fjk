import os
import torch
import numpy as np
import torch.nn as nn
from .yolo import ModelSE
from .loss import loss_f
#from datasets.VOC import VOC_CLASSES
from datasets.voc_eval import evaluate_detections
from torch.autograd import Variable
from libs.Visualizer import Visualizer
from models.base_model import BaseModel
from tqdm import tqdm
# from apex import amp
import torchvision
import time
import math
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.cuda import amp
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import torch.backends.cudnn as cudnn
# python  main.py --det_model FINet --task train --detname VHR --det_dataset VOC --num_classes 10   --batch_size 2 --lr 1e-2 --num_epoch 30 --weight_decay 5e-4  --momentum 0.937

def init_seeds(seed=0):
	torch.manual_seed(seed)

	# Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
	if seed == 0:  # slower, more reproducible
		cudnn.deterministic = True
		cudnn.benchmark = False
	else:  # faster, less reproducible
		cudnn.deterministic = False
		cudnn.benchmark = True
def arguments():
    args = {
        # "--strides": [8, 16, 32],
        # "--use_l1": True,
        # "--amp_training": False,
        # "--conf_thre": 0.7,
        # "--nms_thre": 0.5,
        # "--depth": 0.33,
        # "--width": 0.50,
        # "--n_anchors": 1,
        # "--in_channels": [256, 512, 1024],
        '--anchor_t': (1, 2.0, 8.0),  # anchor-multiple threshold
    }  # notsure  # not sure
    return args


class FINet(BaseModel):
    def __init__(self, parser):
        super().__init__(parser)
        parser.add_args(arguments())
        self.args = parser.get_args()

        self.save_folder = os.path.join(self.args.outf, self.args.det_dataset, self.args.det_model, "basize_{}epoch_{}lr_{}".format(self.args.batch_size, self.args.num_epoch, self.args.lr))
        self.train_folder = os.path.join(self.save_folder, "train")
        self.test_folder = os.path.join(self.save_folder, "test")
        self.val_folder = os.path.join(self.save_folder, "val")

        self.vis = Visualizer(self.args, self.save_folder)
        self.epoch = self.args.num_epoch
        self.drop_lr = self.args.lr_step
        init_seeds(1)
        self.model = ModelSE()
        self.model = self.model.cuda()
        pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
        for k, v in self.model.named_parameters():
            if v.requires_grad:
                if '.bias' in k:
                    pg2.append(v)  # biases
                elif '.weight' in k and '.bn' not in k:
                    pg1.append(v)  # apply weight decay
                else:
                    pg0.append(v)  # all else
        self.optimizer = optim.SGD(pg0, lr=self.args.lr, momentum=self.args.momentum,nesterov=True) if self.args.task == "train" else None
        self.optimizer.add_param_group({'params': pg1, 'weight_decay': self.args.weight_decay})  # add pg1 with weight_decay
        self.optimizer.add_param_group({'params': pg2})  # add pg2 (biases)
        print('Optimizer groups: %g .bias, %g conv.weight, %g other' % (len(pg2), len(pg1), len(pg0)))
        del pg0, pg1, pg2
        self.lf = lambda x: (((1 + math.cos(x * math.pi / self.epoch)) / 2) ** 1.0) * 0.8 + 0.2  # cosine
        self.scheduler = lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self.lf)
        self.ema = ModelEMA(self.model)
        self.net_dict = {"name": ["finet"], "network": [self.model]}
        self.num_classes = self.args.num_classes
        self.best_trigger = False
        self.final_trigger = False
        self.wegiter_trigger = False
        #self.criterion = get_loss(self.args)

    def train(self, train_loader, testset):

#        self.load_weights(self.args.load_weights_dir) if self.args.load_weights_dir != "" else self.vgg_weights()
        self.model.train()
        print("Finished loading model!\nStart training")
        scaler = amp.GradScaler(enabled=True)
        best = float(0)
        self.accumulate = max(round(64 / self.args.batch_size), 1)  # accumulate loss before optimizing
        self.args.weight_decay *= self.args.batch_size * self.accumulate / 64 
        for epo in range(1, self.epoch + 1):
            
            loss_train = self.train_epoch(epo,scaler, train_loader)
            # if epo in self.lr_step:
            #     self.optimizer.param_groups[0]["lr"] *= 0.1
#            print(self.global_steps)
            self.best_trigger = True if loss_train < best else False
            best = loss_train if loss_train < best else best
            self.final_trigger = True if epo == self.epoch else False
            self.save_weights(epo, self.train_folder)

           
            self.inference(testset, from_train=epo)
        self.model.train()

    def train_epoch(self, epo, scaler, train_loader):
        self.model.train()
        epoch_size = len(train_loader) // self.args.batch_size
        mloss = torch.zeros(4).cuda()
        loss_mean = float(0)
        num_iters = len(train_loader)
        
        self.optimizer.zero_grad()
        for iteration,data in tqdm(enumerate(train_loader)):
            ni = iteration + num_iters * (epo-1) # number integrated batches (since train start)
            # Warmup
            if ni <= 16464:
                xi = [0, 16464]  # x interp
                # model.gr = np.interp(ni, xi, [0.0, 1.0])  # giou loss ratio (obj_loss = 1.0 or giou)
                self.accumulate = max(1, np.interp(ni, xi, [1, 64 / self.args.batch_size]).round())
                for j, x in enumerate(self.optimizer.param_groups):
                    # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                    x['lr'] = np.interp(ni, xi, [0.1 if j == 2 else 0.0, self.args.lr* self.lf(epo-1)])
                   
                    if 'momentum' in x:
                        x['momentum'] = np.interp(ni, xi, [0.9, self.args.momentum])
            #print(self.optimizer)
            batch_imgs,targets=data
            targets_all = torch.cat(targets,dim=0)
            imgs = torch.stack(batch_imgs)
            imgs = imgs.float().cuda()
            #cuda = device.type != 'cpu'
            #print(targets)
            with amp.autocast(enabled=True):
            # for i in range(len(batch_imgs)):
                pred = self.model(imgs)
            #self.optimizer.zero_grad()
                loss, loss_items = loss_f(pred, targets_all, self.model)
            scaler.scale(loss).backward()
            if ni % self.accumulate == 0:
                scaler.step(self.optimizer)  # optimizer.step
                scaler.update()
                self.optimizer.zero_grad()
                if self.ema is not None:
                    self.ema.update(self.model)
        
            #loss=losses[-1]
            #loss_mean += total_loss
            #loss.backward()
            
            # self.optimizer.step()
            mloss = (mloss *  iteration+ loss_items) / ( iteration+ 1) 
       
            self.save_loggin_print(epo, { "loss": loss[0],"mloss": mloss[0]}, iteration, num_iters)
        self.scheduler.step()
        return loss

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
            # scale = data[1]
                targets = data[1][0]
                scale = data[1][2]
                img = data[0].cuda()
                img = img.unsqueeze(0).float()
                nb, _, height, width = img.shape 
                out=self.model(img)
                #clip_coords(out[0], (height, width))
                output = non_max_suppression(out[0], conf_thres=0.001, iou_thres=0.6, merge=False)
                
                if output == [None]:
                    continue
                #clip_coords(output[0], (height, width))
                boxes = output[0][:,:4].unsqueeze(0)
                #boxes/=scale
                #print(img.shape[2:])
                scale_coords(img.shape[2:], boxes, data[1][1][:2]) 
                #print(boxes)
                scores = output[0][:,4].unsqueeze(0) 
                classes = output[0][:,5].unsqueeze(0)
                #print(scores,classes)
                for j in range(self.num_classes):
                    inds = np.where(classes[0].cpu().detach().numpy()==j)[0]
                    if len(inds) == 0:
                        all_boxes[j][i] = np.empty([0, 5], dtype=np.float32)
                        continue
                    c_dets = np.hstack((boxes[0][inds].cpu().detach().numpy(),scores[0][inds].cpu().detach().numpy().reshape(-1,1)))
                    all_boxes[j][i] = c_dets
        if self.args.test_save:
                img_id, annotation = testset.pull_anno(i)
                test_result_file = os.path.join(self.save_qfolder, "test_result.txt")
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
            #self.detection_head=DetectHead().cuda()
            # self.clip_boxes=ClipBoxes().cuda()
            # scores,classes,boxes_= self.detection_head(out)
            # boxes=self.clip_boxes(img,boxes_)
            # #boxes = boxes/scale
            # for j in range(self.num_classes):
            #     inds = np.where(classes[0].cpu().numpy()==j)[0]
            #     if len(inds) == 0:
            #         all_boxes[j][i] = np.empty([0, 5], dtype=np.float32)
            #         continue
            #     c_dets = np.hstack((boxes[0][inds].cpu().detach().numpy(),scores[0][inds].cpu().detach().numpy().reshape(-1,1)))
            #     all_boxes[j][i] = c_dets

def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords
def clip_coords(boxes, img_shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    boxes[:, 0].clamp_(0, img_shape[1])  # x1
    boxes[:, 1].clamp_(0, img_shape[0])  # y1
    boxes[:, 2].clamp_(0, img_shape[1])  # x2
    boxes[:, 3].clamp_(0, img_shape[0])  # y2

        # if self.args.test_save:
        #         img_id, annotation = testset.pull_anno(i)
        #         test_result_file = os.path.join(self.save_folder, "test_result.txt")
        #         self.vis.write_gt(test_result_file, img_id, annotation)
        #         eval("from datasets.VOC import {}_CLASSES as CLASSES".format(self.args.detname))
        #         self.vis.write_bb(test_result_file, detections, CLASSES)
        # print("Evaluting detections...")
        # if self.args.det_dataset == 'VOC':VHR_CLASSES = ("airplane", "ship", "storage tank", "baseball diamond", "tennis court", "basketball court", 
#     
        #         #print(all_boxes[1])            
        #         evaluate_detections(self.args, from_train, all_boxes, self.save_folder, testset)
        #     else:
        #         evaluate_detections(self.args, "test", all_boxes, fold, testset)
        # else:
            
        #     testset.run_eval( testset, all_boxes, self.save_folder)




    
    


    

    

def non_max_suppression(prediction, conf_thres=0.001, iou_thres=0.6, merge=False, classes=None, agnostic=False):
    """Performs Non-Maximum Suppression (NMS) on inference results

    Returns:
         detections with shape: nx6 (x1, y1, x2, y2, conf, cls)
    """
    if prediction.dtype is torch.float16:
        prediction = prediction.float()  # to FP32

    nc = prediction[0].shape[1] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Settings
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    max_det = 300  # maximum number of detections per image
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label = nc > 1  # multiple labels per box (adds 0.5ms/img)

    t = time.time()
    output = [None] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # If none remain process next image
        n = x.shape[0]  # number of boxes
        if not n:
            continue

        # Sort by confidence
        # x = x[x[:, 4].argsort(descending=True)]

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.boxes.nms(boxes, scores, iou_thres)
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            try:  # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
                iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
                weights = iou * scores[None]  # box weights
                x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
                if redundant:
                    i = i[iou.sum(1) > 1]  # require redundancy
            except:  # possible CUDA error https://github.com/ultralytics/yolov3/issues/1139
                print(x, i, x.shape, i.shape)
                pass

        output[xi] = x[i]
        # if (time.time() - t) > time_limit:
        #     break  # time limit exceeded
    return output
def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y
def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)



from copy import deepcopy
def is_parallel(model):
    # is model is parallel with DP or DDP
    return type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)
class ModelEMA:
    """ Model Exponential Moving Average from https://github.com/rwightman/pytorch-image-models
    Keep a moving average of everything in the model state_dict (parameters and buffers).
    This is intended to allow functionality like
    https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    A smoothed version of the weights is necessary for some training schemes to perform well.
    This class is sensitive where it is initialized in the sequence of model init,
    GPU assignment and distributed training wrappers.
    """

    def __init__(self, model, decay=0.9999, updates=0):
        # Create EMA
        self.ema = deepcopy(model.module if is_parallel(model) else model).eval()  # FP32 EMA
        # if next(model.parameters()).device.type != 'cpu':
        #     self.ema.half()  # FP16 EMA
        self.updates = updates  # number of EMA updates
        self.decay = lambda x: decay * (1 - math.exp(-x / 2000))  # decay exponential ramp (to help early epochs)
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        # Update EMA parameters
        with torch.no_grad():
            self.updates += 1
            d = self.decay(self.updates)

            msd = model.module.state_dict() if is_parallel(model) else model.state_dict()  # model state_dict
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1. - d) * msd[k].detach()

    def update_attr(self, model, include=(), exclude=('process_group', 'reducer')):
        # Update EMA attributes
        copy_attr(self.ema, model, include, exclude)
def copy_attr(a, b, include=(), exclude=()):
    # Copy attributes from b to a, options to only include [...] and to exclude [...]
    for k, v in b.__dict__.items():
        if (len(include) and k not in include) or k.startswith('_') or k in exclude:
            continue
        else:
            setattr(a, k, v)