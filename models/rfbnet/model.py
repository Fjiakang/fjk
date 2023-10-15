import os
from tkinter import N
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
from .rfb_net import rfb_net

from datasets.voc_eval import evaluate_detections
from torch.autograd import Variable
from torch.autograd import Function
from libs.Visualizer import Visualizer
from models.base_model import BaseModel
from libs.utils.boxes_utils import py_cpu_nms,nms
from .loss import MultiBoxLoss
from tqdm import tqdm
from .add_transform import BaseTransform
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
        "--top_k": 300,
        "--swap": (2, 0, 1),
        "--conf_thresh": 0.01,
        "--variance": [0.1, 0.2],
    }
    return args


class rfbnet(BaseModel):
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
        self.num_classes = self.args.num_classes+1

        self.model = rfb_net(self.args)
        self.model = self.model.cuda()
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.args.lr, momentum=self.args.momentum, weight_decay=self.args.weight_decay) if self.args.task == "train" else None
        self.net_dict = {"name": ["ssd"], "network": [self.model]}
        self.best_trigger = False
        self.final_trigger = False
        self.wegiter_trigger = False

        self.criterion = MultiBoxLoss(self.args)

    def vgg_weights(self):
        base_weight = torch.load(self.args.basenet_weights)
        self.model.backbone.vgg.load_state_dict(base_weight)

        def weights_init(m):
            for key in m.state_dict():
                if key.split(".")[-1] == "weight":
                    if "conv" in key:
                        init.kaiming_normal_(m.state_dict()[key], mode="fan_out")
                    if "bn" in key:
                        m.state_dict()[key][...] = 1
                elif key.split(".")[-1] == "bias":
                    m.state_dict()[key][...] = 0

        print("Initializing weights...")

        self.model.backbone.extras.apply(weights_init)
        self.model.head.loc.apply(weights_init)
        self.model.head.conf.apply(weights_init)
    def train(self, train_loader, testset):

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

            if epo%3==0:
                self.inference(testset, from_train=epo)
                self.model.train()

    def train_epoch(self, epo, train_loader):
        step_index = 0
        epoch_size = len(train_loader) // self.args.batch_size
        stepvalues = (150 * epoch_size, 200 * epoch_size, 250 * epoch_size)
        batch_iteration = iter(train_loader)
        loss_mean = float(0)
        num_iters = len(train_loader)
        for iteration in range(len(train_loader)):
            if iteration in stepvalues:
                step_index += 1
            lr = adjust_learning_rate(self, self.optimizer, self.args.gamma, epo, step_index, iteration, epoch_size)
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

        return loss_mean

    def test(self, testset):
        self.load_weights(self.args.load_weights_dir)

        print("Finished loading model!")
        
        self.inference(testset)

    def inference(self, testset,  fold=None, from_train=-1):
        self.model.eval()
        max_per_image=300
        size = int(512)
        rgb_means = (104, 117, 123)
        transform = BaseTransform(size, rgb_means, (2, 0, 1))
        fold = self.val_folder if fold == None else fold
        all_boxes = [[[] for _ in range(len(testset))] for _ in range(self.num_classes)]
        detector = Detect(self.num_classes,0)
        thresh = 0.005
        for i in tqdm(range(len(testset))):
            
            img = testset.pull_image(i)
            
            scale = torch.Tensor([img.shape[1], img.shape[0],
                             img.shape[1], img.shape[0]])
            with torch.no_grad():
                x = transform(img).unsqueeze(0)
            x = x.cuda()
            scale = scale.cuda()
            
            
            outputs, priors = self.model(x)
            outputs, priors = self.test_postprocess(outputs, priors)
            boxes, scores = detector.forward(outputs,priors)
            boxes = (boxes[0]*scale).cuda()
            scores=scores[0]
            
           
            scores = scores.cpu().detach().numpy()
           
            for j in range(1, self.num_classes):
                inds = np.where(scores[:, j] > thresh)[0]
                if len(inds) == 0:
                    all_boxes[j][i] = np.empty([0, 5], dtype=np.float32)
                    continue
               
                c_dets = np.hstack(( (boxes.cpu().detach().numpy())[inds], (scores[inds, j])[:, np.newaxis])).astype(np.float32, copy=False)

                keep= py_cpu_nms(c_dets, 0.45)
                c_dets = c_dets[keep, :]
                all_boxes[j][i] = c_dets
            #all_boxes[j][i] = np.array(all_boxes[j][i])
            if max_per_image > 0:
                image_scores = np.hstack([all_boxes[j][i][:, -1] for j in range(1,self.num_classes)])
                if len(image_scores) > max_per_image:
                    image_thresh = np.sort(image_scores)[-max_per_image]
                    for j in range(1, self.num_classes):
                        keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
                        all_boxes[j][i] = all_boxes[j][i][keep, :]

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
        softmax = nn.Softmax(dim=-1)
        outputs = (
                loc.view(loc.size(0), -1, 4),
                softmax(conf.view( -1, self.num_classes)),
            
        )  # loc preds  # conf preds  # default boxes
        return outputs, priors




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
    loc = loc.view(-1, 4)
    boxes = torch.cat((
        priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
        priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])), 1)
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes
class Detect(Function):
    """At test time, Detect is the final layer of SSD.  Decode location preds,
    apply non-maximum suppression to location predictions based on conf
    scores and threshold to a top_k number of output predictions for both
    confidence score and locations.
    """
    def __init__(self, num_classes, bkg_label):
        self.num_classes = num_classes
        self.background_label = bkg_label

        self.variance = [0.1, 0.2]

    def forward(self, predictions, prior):
        """
        Args:
            loc_data: (tensor) Loc preds from loc layers
                Shape: [batch,num_priors*4]
            conf_data: (tensor) Shape: Conf preds from conf layers
                Shape: [batch*num_priors,num_classes]
            prior_data: (tensor) Prior boxes and variances from priorbox layers
                Shape: [1,num_priors,4]
        """

        loc, conf = predictions

        loc_data = loc.data
        conf_data = conf.data
        prior_data = prior.data
        num = loc_data.size(0)  # batch size
        self.num_priors = prior_data.size(0)
        self.boxes = torch.zeros(1, self.num_priors, 4)
        self.scores = torch.zeros(1, self.num_priors, self.num_classes)
        if loc_data.is_cuda:
            self.boxes = self.boxes.cuda()
            self.scores = self.scores.cuda()

        if num == 1:
            # size batch x num_classes x num_priors
            conf_preds = conf_data.unsqueeze(0)

        else:
            conf_preds = conf_data.view(num, num_priors,
                                        self.num_classes)
            self.boxes.expand_(num, self.num_priors, 4)
            self.scores.expand_(num, self.num_priors, self.num_classes)

        # Decode predictions into bboxes.
        for i in range(num):
            decoded_boxes = decode(loc_data[i], prior_data, self.variance)
            conf_scores = conf_preds[i].clone()

            self.boxes[i] = decoded_boxes
            self.scores[i] = conf_scores

        return self.boxes, self.scores

def adjust_learning_rate(self, optimizer, gamma, epoch, step_index, iteration, epoch_size):
    """Sets the learning rate 
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    if epoch < 6:
        lr = 1e-6 + (self.args.lr-1e-6) * iteration / (epoch_size * 5) 
    else:
        lr = self.args.lr * (gamma ** (step_index))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr
