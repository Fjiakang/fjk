import os
from numpy.lib.function_base import select
import torch
import numpy as np
import torch.nn as nn
from torch.nn.modules import transformer
import torch.optim as optim
from torch.autograd import Variable
from models.base_model import BaseModel
from libs.Visualizer import Visualizer
from .net import retina_net, BBoxTransform, ClipBoxes
from .loss import FocalLoss
from torchvision.ops import nms
#from libs.utils.boxes_utils import bboxes_iou, nms, py_cpu_nms
from datasets.voc_eval import evaluate_detections
from datasets import coco_eval
import torch.nn.functional as F
from tqdm import tqdm
import collections
#from datasets.COCO import run_eval
from datasets.COCO import COCO_
assert torch.__version__.split('.')[0] == '1'

print('CUDA available: {}'.format(torch.cuda.is_available()))
def arguments():
    args = {
        "--depth": 50,  # 152,101,50,34,18
        "--keep_res": False
    }
    return args
#python  main.py --det_model retinanet --task train --detname VHR --det_dataset VOC --num_classes 10   --batch_size 2 --lr 1e-4 --num_epoch 100

class retinanet(BaseModel):
    def __init__(self, parser):
        super().__init__(parser)
        parser.add_args(arguments())
        self.args = parser.get_args()

        self.save_folder = os.path.join(self.args.outf, self.args.det_dataset, self.args.detname, self.args.det_model, "basize_{}epoch_{}lr_{}".format(self.args.batch_size, self.args.num_epoch, self.args.lr))
        self.train_folder = os.path.join(self.save_folder, "train")
        self.test_folder = os.path.join(self.save_folder, "test")
        self.val_folder = os.path.join(self.save_folder, "val")

        self.vis = Visualizer(self.args, self.save_folder)

        self.epoch = self.args.num_epoch
        self.drop_lr = self.args.lr_step
        self.num_classes = self.args.num_classes

        self.model = retina_net(self.args)
        self.model = self.model.cuda()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr) if self.args.task == "train" else None
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=3, verbose=True)
        # self.optimizer = optim.Adam(self.model.parameters(), lr = self.args.lr)
        self.net_dict = {"name": ["retinanet"], "network": [self.model]}
        self.best_trigger = False
        self.final_trigger = False
        self.wegiter_trigger = False

        self.criterion = FocalLoss()

    def train(self, train_loader, testset):
        if self.args.load_weights_dir != "":
            self.load_weights(self.args.load_weights_dir)
            print("Finished loading model!")
        self.model.train()
        #self.model.freeze_bn()
        #self.model.module.freeze_bn()
        best = float(0)
        print("Start training")
        loss_hist = collections.deque(maxlen=500)
        for epo in range(1, self.epoch + 1):
            loss_train = self.train_epoch(epo,loss_hist, train_loader)
            
            # if epo in self.lr_step:
            #     self.optimizer.param_groups[0]["lr"] *= 0.1
            # print(self.args.lr)
            self.best_trigger = True if loss_train < best else False
            best = loss_train if loss_train < best else best
            self.final_trigger = True if epo == self.epoch else False
            self.save_weights(epo, self.train_folder)

            if self.args.train_with_test > 0 and epo % self.args.train_with_test == 0:
            #if epo%5 == 0:
                self.inference(testset, from_train=epo)
            self.model.train()

    def train_epoch(self, epo,loss_hist, train_loader):
        self.model.train()
        epoch_loss = []
        num_iters = len(train_loader)
        for iteration, data in enumerate(train_loader):
            try:
                images = data['img'].cuda().float()
                targets = data['annot'].cuda()
                classification, regression, priors = self.model(images)
                self.optimizer.zero_grad()
                loss_cls, loss_reg = self.criterion(classification, regression, priors, targets)
                loss_cls = loss_cls.mean()
                loss_reg = loss_reg.mean()
                loss = loss_cls + loss_reg
                if bool(loss == 0):
                        continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.1)
                self.optimizer.step()
                loss_hist.append(float(loss))
                epoch_loss.append(float(loss))
                self.save_loggin_print(epo, {"loss": np.mean(loss_hist), "loss_cls": loss_cls, "loss_reg": loss_reg}, iteration, num_iters)
                del loss_reg
                del loss_cls
            except Exception as e:
                print(e)
                continue
        self.scheduler.step(np.mean(epoch_loss))
        return loss
        
    def test(self, testset):
        self.load_weights(self.args.load_weights_dir)
        print("Finished loading model!")
        self.inference(testset)

    def inference(self, testset, fold=None, from_train=-1):
        self.model.eval()
        #self.model.module.freeze_bn()
        fold = self.val_folder if fold == None else fold
        all_boxes = [[[] for _ in range(len(testset))] for _ in range(self.num_classes)]
       
        for i in tqdm(range(len(testset))):
            data = testset[i]
            #image, gt, h, w= testset.pull_item(i)
            img = data[0]
            scale = data[1][1]
            x = img.permute(2, 0, 1).cuda().float().unsqueeze(dim=0)
            
            x = x.cuda()
            classification, regression, anchors = self.model(x)
            regressBoxes = BBoxTransform()
            clipBoxes = ClipBoxes()
            transformed_anchors = regressBoxes(anchors, regression)
            transformed_anchors = clipBoxes(transformed_anchors, x)
            scores, labels, boxes = self.test_postprocess(classification, transformed_anchors)

            scores = scores.cuda()
            labels = labels.cuda()
            boxes = boxes.cuda()
            boxes /= scale
            for j in range(self.num_classes):
                inds = np.where(labels.cpu().numpy()==j)[0]
                if len(inds) == 0:
                    all_boxes[j][i] = np.empty([0, 5], dtype=np.float32)
                    continue
                c_dets = np.hstack((boxes[inds].cpu().detach().numpy(),scores[inds].cpu().detach().numpy().reshape(-1,1)))
                all_boxes[j][i] = c_dets
            # correct boxes for image scale
        print("Evaluting detections...")
        if self.args.det_dataset == 'VOC':
            if from_train != -1:
                #print(all_boxes[1])            
                evaluate_detections(self.args, from_train, all_boxes, self.save_folder, testset)
            else:
                evaluate_detections(self.args, "test", all_boxes, fold, testset)
        else:
            
            testset.run_eval( testset, all_boxes, self.save_folder)



            
    def test_postprocess(self, classification, transformed_anchors):
        finalResult = [[], [], []]
        finalScores = torch.Tensor([]).cuda()
        finalAnchorBoxesIndexes = torch.Tensor([]).long().cuda()
        finalAnchorBoxesCoordinates = torch.Tensor([]).cuda()

        for i in range(classification.shape[2]):
            scores = torch.squeeze(classification[:, :, i])
            scores_over_thresh = scores > 0.05
            if scores_over_thresh.sum() == 0:
                # no boxes to NMS, just continue
                continue

            scores = scores[scores_over_thresh]
            anchorBoxes = torch.squeeze(transformed_anchors)
            anchorBoxes = anchorBoxes[scores_over_thresh]
            anchors_nms_idx = nms(anchorBoxes, scores, 0.5)

            finalResult[0].extend(scores[anchors_nms_idx])
            finalResult[1].extend(torch.tensor([i] * anchors_nms_idx.shape[0]))
            finalResult[2].extend(anchorBoxes[anchors_nms_idx])

            finalScores = torch.cat((finalScores, scores[anchors_nms_idx]))
            finalAnchorBoxesIndexesValue = torch.tensor([i] * anchors_nms_idx.shape[0])
            finalAnchorBoxesIndexesValue = finalAnchorBoxesIndexesValue.cuda()

            finalAnchorBoxesIndexes = torch.cat((finalAnchorBoxesIndexes, finalAnchorBoxesIndexesValue))
            finalAnchorBoxesCoordinates = torch.cat((finalAnchorBoxesCoordinates, anchorBoxes[anchors_nms_idx]))
        return [finalScores, finalAnchorBoxesIndexes, finalAnchorBoxesCoordinates]
    




            #################################################################################################################################
            # regressBoxes = BBoxTransform()
            # clipBoxes = ClipBoxes()
            # transformed_anchors = regressBoxes(anchors, regression)
            # transformed_anchors = clipBoxes(transformed_anchors, x)
            # boxes = transformed_anchors[0]
            # scores = classification[0]
            # boxes /= scale
            # #detections_boxes = boxes.cpu().detach()
            # detections_scores = scores.cpu().detach().numpy()
            # #detections = self.test_postprocess( scale, classification, regression, transformed_anchors)
            # for j in range( self.num_classes):
            #     inds = np.where(detections_scores[:, j] > thresh)[0]
            #     if len(inds) == 0:
            #         all_boxes[j][i] = np.empty([0, 5], dtype=np.float32)
            #         continue
            #     c_dets = np.hstack(( (boxes.cpu().detach().numpy())[inds], detections_scores[inds, j][:, np.newaxis])).astype(np.float32, copy=False)

            #     keep= nms(torch.tensor(np.delete(c_dets,-1,1)),torch.tensor(c_dets[:,-1]), 0.5)
            #     c_dets = c_dets[keep, :]
            #     if len(c_dets) == 5:
            #         c_dets = (torch.tensor(all_boxes[j][i])).view(-1,5).numpy()
            #     all_boxes[j][i] = c_dets
            # #all_boxes[j][i] = np.array(all_boxes[j][i])
            # if max_per_image > 0:
            #     image_scores = np.hstack([all_boxes[j][i][:, -1] for j in range(self.num_classes)])
            #     if len(image_scores) > max_per_image:
            #         image_thresh = np.sort(image_scores)[-max_per_image]
            #         for j in range( self.num_classes):
            #             keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
            #             all_boxes[j][i] = all_boxes[j][i][keep, :]
            ##########################################################################################################################################


            
            #     dets = detections[0, j, :]
            #     mask = dets[:, 0].gt(0.0).expand(5, dets.size(0)).t()
            #     dets = torch.masked_select(dets, mask).view(-1, 5)
            #     if dets.size(0) == 0:
            #         continue
            #     boxes = dets[:, 1:]
            #     scores = dets[:, 0].cpu().detach().numpy()
            #     cls_dets = np.hstack((boxes.cpu().detach().numpy(), scores[:, np.newaxis])).astype(np.float32, copy=False)
            #     all_boxes[j][i] = cls_dets
            # if self.args.test_save:
            #     img_id, annotation = testset.pull_anno(i)
            #     test_result_file = os.path.join(self.save_folder, "test_result.txt")
            #     self.vis.write_gt(test_result_file, img_id, annotation)
            #     eval("from datasets.VOC import {}_CLASSES as CLASSES".format(self.args.detname))
            #     self.vis.write_bb(test_result_file, detections, CLASSES)
        #print(all_boxes[0])
       # print(all_boxes[1])
#        print(all_boxes[10])
        
       
            # scores = scores.cpu()
            # labels = labels.cpu()
            # boxes = boxes.cpu()
            # correct boxes for image scale
            
    # def test_postprocess(self, scale, classification, regression, transformed_anchors):
    #     finalResult = [[], [], []]
    #     finalScores = torch.Tensor([]).cuda()
    #     finalAnchorBoxesIndexes = torch.Tensor([]).long().cuda()
    #     finalAnchorBoxesCoordinates = torch.Tensor([]).cuda()
    #     top_k = 200
    #     nms_thresh = 0.5
    #     #num = regression.size(0)
    #     output = torch.zeros(1, classification.size(2), top_k, 5)
    #     #for i in range(num):
    #     for j in range(1, classification.shape[2]):
    #         scores = torch.squeeze(classification[:, :, j])
    #         scores_over_thresh = scores > 0.05
    #         if scores_over_thresh.sum() == 0:
    #             # no boxes to NMS, just continue
    #             continue

    #         scores = scores[scores_over_thresh]
    #         anchorBoxes = torch.squeeze(transformed_anchors)
    #         anchorBoxes = anchorBoxes[scores_over_thresh].detach()
    #         #anchors_nms_idx = nms(anchorBoxes, scores, 0.5)
            
    #         ids, count = nms(anchorBoxes, scores, nms_thresh, top_k)
    #         # finalResult[0].extend(scores[anchors_nms_idx])
    #         # finalResult[1].extend(torch.tensor([i] * anchors_nms_idx.shape[0]))
    #         # finalResult[2].extend(anchorBoxes[anchors_nms_idx])
    #         anchorBoxes/=scale
    #         output[i, j, :count] = torch.cat((scores[ids[:count]].unsqueeze(1),anchorBoxes[ids[:count]]), 1)
    #     flt = output.contiguous().view(num, -1, 5)
    #     _, idx = flt[:, :, 0].sort(1, descending=True)
    #     _, rank = idx.sort(1)
    #     flt[(rank < top_k).unsqueeze(-1).expand_as(flt)].fill_(0)
    #     return output
        #     finalScores = torch.cat((finalScores, scores[anchors_nms_idx]))
        #     finalAnchorBoxesIndexesValue = torch.tensor([i] * anchors_nms_idx.shape[0])
        #     finalAnchorBoxesIndexesValue = finalAnchorBoxesIndexesValue.cuda()

        #     finalAnchorBoxesIndexes = torch.cat((finalAnchorBoxesIndexes, finalAnchorBoxesIndexesValue))
        #     finalAnchorBoxesCoordinates = torch.cat((finalAnchorBoxesCoordinates, anchorBoxes[anchors_nms_idx]))
        # return [finalScores, finalAnchorBoxesIndexes, finalAnchorBoxesCoordinates]
