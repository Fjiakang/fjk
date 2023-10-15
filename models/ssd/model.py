import os
import torch
import numpy as np
import torch.nn as nn
from .ssd_net import ssd_net
import torch.optim as optim
import torch.nn.init as init
import torch.nn.functional as F
from datasets.voc_eval import evaluate_detections
from torch.autograd import Variable
from torch.autograd import Function
from libs.Visualizer import Visualizer
from models.base_model import BaseModel
from libs.utils.boxes_utils import bboxes_iou, nms
from .loss import MultiBoxLoss


def arguments():
    args = {
        "--basenet_weights": "./weights/vgg16_reducedfc.pth",
        "--resume_net": None,
        "--overlap_thresh": 0.5,
        "--nms_thresh": 0.45,
        "--visual_threshold": 0.6,
        "--prior_for_matching": True,
        "--bkg_label": 0,
        "--neg_mining": True,
        "--neg_pos": 3,
        "--neg_overlap": 0.5,
        "--encode_target": False,
        "--top_k": 200,
        "--variance": [0.1, 0.2],
        # "--rgb_means": (104, 117, 123),
        "--swap": (2, 0, 1),
        "--conf_thresh": 0.01,
    }
    return args


class ssd(BaseModel):
    def __init__(self, parser):
        super().__init__(parser)
        parser.add_args(arguments())
        self.args = parser.get_args()
        torch.set_default_tensor_type("torch.cuda.FloatTensor") if not self.device == "cpu" else torch.set_default_tensor_type("torch.FloatTensor")

        self.save_folder = os.path.join(self.args.outf, self.args.det_dataset, self.args.detname, self.args.det_model, "basize_{}epoch_{}lr_{}".format(self.args.batch_size, self.args.num_epoch, self.args.lr))
        self.train_folder = os.path.join(self.save_folder, "train")
        self.test_folder = os.path.join(self.save_folder, "test")
        self.val_folder = os.path.join(self.args.load_weights_dir.split("train")[0], "val")

        self.vis = Visualizer(self.args, self.save_folder)

        self.epoch = self.args.num_epoch
        self.drop_lr = self.args.lr_step
        self.num_classes = self.args.num_classes + 1

        self.model = ssd_net(self.args)
        self.model = self.model.cuda()
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.args.lr, momentum=self.args.momentum, weight_decay=self.args.weight_decay) if self.args.task == "train" else None
        self.net_dict = {"name": ["ssd"], "network": [self.model]}
        self.best_trigger = False
        self.final_trigger = False
        self.wegiter_trigger = False

        self.criterion = MultiBoxLoss(self.args)
        self.lossdic = dict(loss=[])

    def vgg_weights(self):
        base_weight = torch.load(self.args.basenet_weights)
        self.model.backbone.vgg.load_state_dict(base_weight)

        def xavier(param):
            init.xavier_uniform_(param)

        def weights_init(m):
            if isinstance(m, nn.Conv2d):
                xavier(m.weight.data)
                m.bias.data.zero_()

        print("Initializing weights...")
        # initialize newly added layers' weights with kaiming_normal method
        self.model.backbone.extras.apply(weights_init)
        self.model.head.loc.apply(weights_init)
        self.model.head.conf.apply(weights_init)

    def train(self, train_loader, testset):
        # 若有断点重训，则doit，or 加载预训练模型

        self.load_weights(self.args.load_weights_dir) if self.args.load_weights_dir != "" else self.vgg_weights()
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
                self.inference(testset, from_train=epo)
                self.model.train()

    def train_epoch(self, epo, train_loader):
        step_index = 0
        batch_iteration = iter(train_loader)
        lr_steps = (80000, 100000, 120000)
        loss_mean = float(0)
        num_iters = len(train_loader)
        for iteration in range(len(train_loader)):
            if iteration in lr_steps:
                step_index += 1
                adjust_learning_rate(self,self.optimizer, self.args.gamma, step_index)
            try:
                images, targets = next(batch_iteration)
            except StopIteration:
                batch_iteration = iter(train_loader)

            images = Variable(images.cuda())
            with torch.no_grad():
                targets = [Variable(ann.cuda()) for ann in targets]

            outputs, priors = self.model(images)
            outputs = self.train_postprocess(outputs, priors)

            self.optimizer.zero_grad()
            loss_l, loss_c = self.criterion(outputs, targets)
            loss = loss_l + loss_c
            loss_mean += loss
            loss.backward()
            self.optimizer.step()
            self.save_loggin_print(epo, {"loss": loss, "loss_c": loss_c, "loss_l": loss_l}, iteration, num_iters)
        self.lossdic["loss"].append(loss.detach().item())
        return loss_mean

    def test(self, testset):
        self.load_weights(self.args.load_weights_dir)
        print("Finished loading model!")
        self.inference(testset)

    def inference(self, testset, fold=None, from_train=-1):
        self.model.eval()
        fold = self.val_folder if fold == None else fold
        all_boxes = [[[] for _ in range(len(testset))] for _ in range(self.num_classes)]
        for i in range(len(testset)):
            img, _, h, w = testset.pull_item(i)
            x = Variable(img.unsqueeze(0))
            x = x.cuda()
            outputs, priors = self.model(x)
            detections = self.test_postprocess(outputs, priors).data
            # # torch.zeros(batch, self.num_classes, self.top_k,    ``) detection
            detections = detections * torch.Tensor([1, w, h, w, h])
            for j in range(1, self.num_classes):
                dets = detections[0, j, :]
                mask = dets[:, 0].gt(0.0).expand(5, dets.size(0)).t()
                dets = torch.masked_select(dets, mask).view(-1, 5)
                if dets.size(0) == 0:
                    continue
                boxes = dets[:, 1:]
                scores = dets[:, 0].cpu().numpy()
                cls_dets = np.hstack((boxes.cpu().numpy(), scores[:, np.newaxis])).astype(np.float32, copy=False)
                all_boxes[j][i] = cls_dets

            if self.args.test_save:
                img_id, annotation = testset.pull_anno(i)
                test_result_file = os.path.join(self.save_folder, "test_result.txt")
                self.vis.write_gt(test_result_file, img_id, annotation)
                eval("from datasets.VOC import {}_CLASSES as CLASSES".format(self.args.detname))
                self.vis.write_bb(test_result_file, detections, CLASSES)

        print("Evaluting detections...")
        if from_train != -1:
            evaluate_detections(self.args, from_train, all_boxes, self.save_folder, testset)
        else:
            evaluate_detections(self.args, "test", all_boxes, fold, testset)

    def train_postprocess(self, outputs, priors):
        # {"loc": loc, "conf": conf}
        loc = outputs["loc"]
        conf = outputs["conf"]
        outputs = (
            loc.view(loc.size(0), -1, 4),
            conf.view(conf.size(0), -1, self.num_classes),
            priors,
        )
        return outputs

    def test_postprocess(self, outputs, priors):
        loc = outputs["loc"]
        conf = outputs["conf"]
        softmax = nn.Softmax(dim=1)
        outputs = Detect().apply(
            self.args,
            loc.view(loc.size(0), -1, 4),
            softmax(conf.view(-1, self.num_classes)),
            priors,
        )  # loc preds  # conf preds  # default boxes
        return outputs

def adjust_learning_rate(self,optimizer, gamma, step):
    """Sets the learning rate to the initial LR decayed by 10 at every
        specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = self.args.lr * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
class Detect(Function):
    """At test time, Detect is the final layer of SSD.  Decode location preds,
    apply non-maximum suppression to location predictions based on conf
    scores and threshold to a top_k number of output predictions for both
    confidence score and locations.
    """

    def forward(self, args, loc_data, conf_data, prior_data):
        """
        Args:
            loc_data: (tensor) Loc preds from loc layers
                Shape: [batch,num_priors*4]
            conf_data: (tensor) Shape: Conf preds from conf layers
                Shape: [batch*num_priors,num_classes]
            prior_data: (tensor) Prior boxes and variances from priorbox layers
                Shape: [1,num_priors,4]
        """
        # num_classes, bkg_label, top_k, conf_thresh, nms_thresh,
        # self.args.num_classes, 0, 200, 0.01, 0.45,
        self.num_classes = args.num_classes + 1
        self.background_label = args.bkg_label
        self.top_k = args.top_k
        # Parameters used in nms.
        self.nms_thresh = args.nms_thresh
        self.conf_thresh = args.conf_thresh
        self.variance = args.variance

        num = loc_data.size(0)  # batch size
        num_priors = prior_data.size(0)
        output = torch.zeros(num, self.num_classes, self.top_k, 5)
        conf_preds = conf_data.view(num, num_priors, self.num_classes).transpose(2, 1)

        # Decode predictions into bboxes.
        for i in range(num):
            decoded_boxes = decode(loc_data[i], prior_data, self.variance)
            # For each class, perform nms
            conf_scores = conf_preds[i].clone()

            for cl in range(1, self.num_classes):
                c_mask = conf_scores[cl].gt(self.conf_thresh)
                scores = conf_scores[cl][c_mask]
                if scores.size(0) == 0:
                    continue
                l_mask = c_mask.unsqueeze(1).expand_as(decoded_boxes)
                boxes = decoded_boxes[l_mask].view(-1, 4)
                # idx of highest scoring and non-overlapping boxes per class
                ids, count = nms(boxes, scores, self.nms_thresh, self.top_k)
                output[i, cl, :count] = torch.cat((scores[ids[:count]].unsqueeze(1), boxes[ids[:count]]), 1)
        flt = output.contiguous().view(num, -1, 5)
        _, idx = flt[:, :, 0].sort(1, descending=True)
        _, rank = idx.sort(1)
        flt[(rank < self.top_k).unsqueeze(-1).expand_as(flt)].fill_(0)
        return output


def decode(loc, priors, variances):
    """Decode locations from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        loc (tensor): location predictions for loc layers,
            Shape: [num_priors,4]
        priors (tensor): Prior boxes in center-offset form.
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        decoded bounding box predictions
    """

    boxes = torch.cat((priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:], priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])), 1)
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes
