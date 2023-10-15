import os
import torch
import numpy as np
import torch.nn as nn
from .yolox_net import YOLOX as yolox_net
from .loss import get_loss
#from datasets.VOC import VOC_CLASSES
from datasets.voc_eval import evaluate_detections
from torch.autograd import Variable
from libs.Visualizer import Visualizer
from models.base_model import BaseModel

# from apex import amp
import torchvision


def arguments():
    args = {
        "--strides": [8, 16, 32],
        "--use_l1": True,
        "--amp_training": False,
        "--conf_thre": 0.7,
        "--nms_thre": 0.5,
        "--depth": 0.33,
        "--width": 0.50,
        "--n_anchors": 1,
        "--in_channels": [256, 512, 1024],
    }  # notsure  # not sure
    return args


class yolox(BaseModel):
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

        self.model = yolox_net(self.args)
        self.model = self.model.cuda()
        self.net_dict = {"name": ["yolox"], "network": [self.model]}
        self.num_classes = 10
        self.best_trigger = False
        self.final_trigger = False
        self.wegiter_trigger = False
        self.critrion = get_loss(self.args)

    def get_optimizer(self):
        pg0, pg1, pg2 = [], [], []
        for k, v in self.model.named_modules():
            if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
                pg2.append(v.bias)  # biases
            if isinstance(v, nn.BatchNorm2d) or "bn" in k:
                pg0.append(v.weight)  # no decay
            elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
                pg1.append(v.weight)  # apply decay

        optimizer = torch.optim.SGD(pg0, lr=self.args.lr, momentum=self.args.momentum, nesterov=True)
        optimizer.add_param_group({"params": pg1, "weight_decay": self.args.weight_decay})  # add pg1 with weight_decay
        optimizer.add_param_group({"params": pg2})
        return optimizer

    def yolo_init(self):
        def yolox_weights(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03

        self.model.apply(yolox_weights)
        self.model.head.initialize_biases(prior_prob=1e-2)

    def train(self, train_loader, testset):
        self.load_weights(self.args.load_weights_dir) if self.args.load_weights_dir != "" else self.yolo_init()
        self.optimizer = self.get_optimizer()
        self.model.train()
        print("Finished loading model!\nStart training")

        best = float(0)
        for epo in range(1, self.epoch + 1):
            loss_train = self.train_epoch(epo, train_loader)
            if epo in self.lr_step:
                self.optimizer.param_groups[0]["lr"] *= 0.1

            self.best_trigger = True if loss_train < best else False
            best = loss_train if loss_train < best else best
            self.final_trigger = True if epo == self.epoch else False
            self.save_weights(epo, self.train_folder)

            if self.args.train_with_test > 0 and epo % self.args.train_with_test == 0:
                self.inference(testset)
                self.model.train()

    def train_epoch(self, epo, train_loader):

        batch_iteration = iter(train_loader)
        loss_mean = float(0)
        num_iters = len(train_loader)
        for iteration, data in enumerate(train_loader):
            
            images = data[0].cuda().float()
            targets = data[1].cuda()

            images = Variable(images.cuda())
            # with torch.no_grad():
            #     targets = [Variable(ann.cuda()) for ann in targets]
            outputs = self.model(images)
            x_shifts, y_shifts, expanded_strides, outputs, origin_preds = self.train_postprocess(outputs)

            loss_stats, num_fg = self.critrion(x_shifts, y_shifts, expanded_strides, outputs, origin_preds, targets)
            loss = loss_stats["total_loss"]
            self.optimizer.zero_grad()
            loss.backward()
            # if self.args.amp_training:
            #     if self.amp_training:
            #         with amp.scale_loss(loss, self.optimizer) as scaled_loss:
            #             scaled_loss.backward()
            #     else:
            #         loss.backward()
            self.optimizer.step()
            self.save_loggin_print(epo, loss_stats, iteration, num_iters)

    def test(self, testset):
        self.load_weights(self.args.load_weights_dir)
        print("Finished loading model!")

        self.inference(testset)

    def inference(self, testset, fold=None):
        self.model.eval()
        fold = self.val_folder if fold == None else fold
        all_boxes = [[[] for _ in range(len(testset))] for _ in range(self.num_classes)]
        for i in range(len(testset)):
            img, gt, h, w = testset.pull_item(i)
            x = Variable(img.unsqueeze(0))
            x = x.cuda()
            outputs = self.model(x)
            outputs = self.test_postprocess(outputs)
            detections = self.post_process(outputs, self.args.conf_thre, self.args.nms_thre)
            img_id, annotation = testset.pull_anno(i)

            detections = self.convert_to_voc_format(detections, (h, w), img_id)

            if self.args.test_save:
                img_id, annotation = testset.pull_anno(i)
                test_result_file = os.path.join(self.save_folder, "test_result.txt")
                self.write_gt(test_result_file, img_id, annotation)
                scale = torch.Tensor([w, h, w, h])
                self.write_bb(test_result_file, detections, scale, VOC_CLASSES)
            else:
                for j in range(1, detections.size(1)):
                    dets = detections[0, j, :]
                    mask = dets[:, 0].gt(0.0).expand(5, dets.size(0)).t()
                    dets = torch.masked_select(dets, mask).view(-1, 5)
                    if dets.size(0) == 0:
                        continue
                    boxes = dets[:, 1:]
                    boxes[:, 0] *= w
                    boxes[:, 2] *= w
                    boxes[:, 1] *= h
                    boxes[:, 3] *= h
                    scores = dets[:, 0].cpu().numpy()
                    cls_dets = np.hstack((boxes.cpu().numpy(), scores[:, np.newaxis])).astype(np.float32, copy=False)
                    all_boxes[j][i] = cls_dets
        print("Evaluting detections...")
        evaluate_detections(all_boxes, fold, testset)

    def train_postprocess(self, outputs):
        cls_output = outputs["cls_output"]
        reg_output = outputs["reg_output"]
        obj_output = outputs["obj_output"]
        outputs = []
        x_shifts = []
        y_shifts = []
        origin_preds = []
        expanded_strides = []
        strides = self.args.strides
        for k in range(len(cls_output)):
            output = torch.cat([reg_output[k], obj_output[k], cls_output[k]], 1)
            output, grid = self.get_output_and_grid(output, k, strides[k])
            x_shifts.append(grid[:, :, 0])
            y_shifts.append(grid[:, :, 1])
            expanded_strides.append(torch.zeros(1, grid.shape[1]).fill_(strides[k]))
            if self.args.use_l1:
                batch_size = reg_output[k].shape[0]
                hsize, wsize = reg_output[k].shape[-2:]
                reg_output[k] = reg_output[k].view(batch_size, self.args.n_anchors, 4, hsize, wsize)
                reg_output[k] = reg_output[k].permute(0, 1, 3, 4, 2).reshape(batch_size, -1, 4)
                origin_preds.append(reg_output[k].clone())
            outputs.append(output)
        return x_shifts, y_shifts, expanded_strides, outputs, origin_preds

    def get_output_and_grid(self, output, k, stride):
        grids = [torch.zeros(1)] * len(self.args.in_channels)
        grid = grids[k]

        batch_size = output.shape[0]
        n_ch = 5 + self.num_classes
        hsize, wsize = output.shape[-2:]
        if grid.shape[2:4] != output.shape[2:4]:
            yv, xv = torch.meshgrid([torch.arange(hsize), torch.arange(wsize)])
            grid = torch.stack((xv, yv), 2).view(1, 1, hsize, wsize, 2)
            grids[k] = grid
        output = output.view(batch_size, self.args.n_anchors, n_ch, hsize, wsize)
        output = output.permute(0, 1, 3, 4, 2).reshape(batch_size, self.args.n_anchors * hsize * wsize, -1)
        grid = grid.view(1, -1, 2)
        grid = grid.to(self.device)
        output[..., :2] = ((output[..., :2] + grid) * stride)
        output[..., 2:4] = torch.exp(output[..., 2:4]) * stride
        return output, grid

    def test_postprocess(self, outputs):
        cls_output = outputs["cls_output"]
        reg_output = outputs["reg_output"]
        obj_output = outputs["obj_output"]
        for k in range(len(cls_output)):
            output = torch.cat([reg_output[k], obj_output[k].sigmoid(), cls_output[k].sigmoid()], 1)
            outputs.append(output)

        outputs = torch.cat([x.flatten(start_dim=2) for x in outputs], dim=2).permute(0, 2, 1)
        prediction = decode(outputs, self.args)

        box_corner = prediction.new(prediction.shape)
        box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
        box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
        box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
        box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
        prediction[:, :, :4] = box_corner[:, :, :4]

        output = [None for _ in range(len(prediction))]
        for i, image_pred in enumerate(prediction):

            # If none are remaining => process next image
            if not image_pred.size(0):
                continue
            # Get score and class with highest confidence
            class_conf, class_pred = torch.max(image_pred[:, 5 : 5 + self.num_classes], 1, keepdim=True)

            conf_mask = (image_pred[:, 4] * class_conf.squeeze() >= self.args.conf_thre).squeeze()
            # _, conf_mask = torch.topk((image_pred[:, 4] * class_conf.squeeze()), 1000)
            # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
            detections = torch.cat((image_pred[:, :5], class_conf, class_pred.float()), 1)
            detections = detections[conf_mask]
            if not detections.size(0):
                continue

            nms_out_index = torchvision.ops.batched_nms(
                detections[:, :4],
                detections[:, 4] * detections[:, 5],
                detections[:, 6],
                self.args.nms_thre,
            )
            detections = detections[nms_out_index]
            if output[i] is None:
                output[i] = detections
            else:
                output[i] = torch.cat((output[i], detections))

        return output

    def post_process(prediction, num_classes, conf_thre=0.7, nms_thre=0.45):
        box_corner = prediction.new(prediction.shape)
        box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
        box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
        box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
        box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
        prediction[:, :, :4] = box_corner[:, :, :4]

        output = [None for _ in range(len(prediction))]
        for i, image_pred in enumerate(prediction):

            # If none are remaining => process next image
            if not image_pred.size(0):
                continue
            # Get score and class with highest confidence
            class_conf, class_pred = torch.max(image_pred[:, 5 : 5 + num_classes], 1, keepdim=True)

            conf_mask = (image_pred[:, 4] * class_conf.squeeze() >= conf_thre).squeeze()
            # _, conf_mask = torch.topk((image_pred[:, 4] * class_conf.squeeze()), 1000)
            # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
            detections = torch.cat((image_pred[:, :5], class_conf, class_pred.float()), 1)
            detections = detections[conf_mask]
            if not detections.size(0):
                continue

            nms_out_index = torchvision.ops.batched_nms(
                detections[:, :4],
                detections[:, 4] * detections[:, 5],
                detections[:, 6],
                nms_thre,
            )
            detections = detections[nms_out_index]
            if output[i] is None:
                output[i] = detections
            else:
                output[i] = torch.cat((output[i], detections))

        return output

    def convert_to_voc_format(self, outputs, info_imgs, ids):
        predictions = {}
        for (output, img_h, img_w, img_id) in zip(outputs, info_imgs[0], info_imgs[1], ids):
            if output is None:
                predictions[int(img_id)] = (None, None, None)
                continue
            output = output.cpu()

            bboxes = output[:, 0:4]

            # preprocessing: resize
            scale = min(self.img_size[0] / float(img_h), self.img_size[1] / float(img_w))
            bboxes /= scale

            cls = output[:, 6]
            scores = output[:, 4] * output[:, 5]

            predictions[int(img_id)] = (bboxes, cls, scores)
        return predictions


def decode(outputs, args):
    grids = []
    strides = []
    hw = [x.shape[-2:] for x in outputs]
    for (hsize, wsize), stride in zip(hw, args.strides):
        yv, xv = torch.meshgrid([torch.arange(hsize), torch.arange(wsize)])
        grid = torch.stack((xv, yv), 2).view(1, -1, 2)
        grids.append(grid)
        shape = grid.shape[:2]
        strides.append(torch.full((*shape, 1), stride))

    grids = torch.cat(grids, dim=1)
    strides = torch.cat(strides, dim=1)

    outputs[..., :2] = (outputs[..., :2] + grids) * strides
    outputs[..., 2:4] = torch.exp(outputs[..., 2:4]) * strides
    return outputs
