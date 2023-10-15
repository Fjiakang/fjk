import cv2
import os
import torch
import numpy as np
from .loss import CtdetLoss
import torch.optim as optim
from .net import get_large_hourglass_net
from .decode import _nms
from datasets.voc_eval import evaluate_detections
from libs.Visualizer import Visualizer
from models.base_model import BaseModel
from libs.utils.images import get_affine_transform
from libs.utils.centernet_aug import transform_preds, _topk
from datasets.VOC import VHR_std, VHR_means, SSDD_std, SSDD_means
from libs.utils.centernet_utils import _sigmoid, flip_tensor, _transpose_and_gather_feat
from tqdm import tqdm
from datasets import coco_eval
try:
    from external.nms import soft_nms
except:
    print("NMS not imported! do \n cd $ROOT/external \n make")


def arguments():
    args = {
        "--num_stacks": 2,
        "--downratio": 4,
        "--top_K": 100,
        "--head_conv": 64,
        "--keep_res": False,  # or False
        "--pad": 128,
        "--rand_crop": True,
        "--scale": 0.4,
        "--shift": 0.1,
        "--flip": 0.5,
        "--hm_weight": 1.0,
        "--off_weight": 1.0,
        "--wh_weight": 0.1,
        "--flip_test": True,  # True会涨点
        "--test_scales": [1],
        "--max_per_image": 100,
        "--nms": False,
    }

    return args


class centernet(BaseModel):
    def __init__(self, parser):
        super().__init__(parser)
        parser.add_args(arguments())
        self.args = parser.get_args()

        self.save_folder = os.path.join(self.args.outf, self.args.det_dataset, self.args.detname, self.args.det_model, "basize_{}epoch_{}lr_{}".format(self.args.batch_size, self.args.num_epoch, self.args.lr))
        self.train_folder = os.path.join(self.save_folder, "train")
        self.test_folder = os.path.join(self.save_folder, "test")
        self.val_folder = os.path.join(self.args.load_weights_dir.split("train")[0], "val")

        self.vis = Visualizer(self.args, self.save_folder)

        self.epoch = self.args.num_epoch
        self.drop_lr = self.args.lr_step
        self.num_classes = self.args.num_classes + 1
        heads = {"hm": self.args.num_classes, "wh": 2, "reg": 2}
        self.model = get_large_hourglass_net(num_layers=0, heads=heads, head_conv=64)
        self.model = self.model.cuda()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr) if self.args.task == "train" else None
        self.net_dict = {"name": ["centernet"], "network": [self.model]}
        self.best_trigger = False
        self.final_trigger = False
        self.wegiter_trigger = False

        self.criterion = CtdetLoss(self.args)

    def train(self, train_loader, testset):

        if self.args.load_weights_dir != "":
            self.load_state_dict(self.args.load_weights_dir)
            print("Finished loading model!\nStart training")

        # TODO 什么作用
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device=self.device, non_blocking=True)
        self.model.train()

        best = float(0)
        for epo in range(1, self.epoch + 1):
            loss_train = self.train_epoch(epo, train_loader)
            if epo in self.lr_step:
                self.optimizer.param_groups[0]["lr"] *= 0.1

            self.best_trigger = True if loss_train < best else False
            best = loss_train if loss_train < best else best
            self.final_trigger = True if epo == self.epoch else False
            self.wegiter_trigger = True if epo % self.args.weight_iter == 0 else False
            self.save_weights(epo, self.train_folder)

            if self.args.train_with_test > 0 and epo % self.args.train_with_test == 0:
                self.inference(testset, from_train=epo)
                self.model.train()

    def train_epoch(self, epo, train_loader):

        batch_iteration = iter(train_loader)
        num_iters = len(train_loader)
        loss_mean = float(0)
        for iteration in range(len(train_loader)):
            try:
                images, targets = next(batch_iteration)
            except StopIteration:
                batch_iteration = iter(train_loader)

            for k in targets:
                targets[k] = targets[k].to(device=self.device, non_blocking=True)
            images = images.to(device=self.device, non_blocking=True)
            images = images.type(torch.cuda.FloatTensor)

            outputs = self.model(images)

            self.optimizer.zero_grad()
            loss, loss_stats = self.criterion(outputs, targets)
            loss = loss.mean()
            loss_mean += loss
            loss.backward()
            self.optimizer.step()
            self.save_loggin_print(epo, loss_stats, iteration, num_iters)
        return loss_mean

    def test(self, testset):
        self.load_weights(self.args.load_weights_dir)
        print("Finished loading model!")
        self.inference(testset)

    def inference(self, testset, fold=None, from_train=-1):
        self.model.eval()
        fold = self.val_folder if fold == None else fold
        all_boxes = [[[] for _ in range(len(testset))] for _ in range(self.num_classes)]

        for i in tqdm(range(len(testset))):
            detection = []
            image, meta, _, _ = testset.pull_item(i)
            image = image.to(self.device)
            _, dets = self.test_process(image)
            det = self.test_post_process(dets, meta, 1)
            detection.append(det)
            detection = self.merge_output(detection, self.args.test_scales)  # {"1":[[],[]],"2":[[],[]]}
            for j in range(1, self.num_classes):
                all_boxes[j][i] = detection[j]

            if self.args.test_save:
                img_id, annotation = testset.pull_anno(i)
                test_result_file = os.path.join(self.save_loader, "test_result.txt")
                self.vis.write_gt(test_result_file, img_id, annotation)
                eval("from libs.data import {}_CLASSES as CLASSES".format(self.args.detname))
                self.vis.write_bb(test_result_file, detection, CLASSES)  # TODO 此detection有误，格式应为# detection:(batch, self.num_classes, self.top_k, 5)

        print("Evaluting detections...")
        #if self.args.det_dataset == 'VOC':
        if from_train != -1:
            evaluate_detections(self.args, from_train, all_boxes, self.save_folder, testset)
        else:
            evaluate_detections(self.args, "test", all_boxes, fold, testset)
        # else:
        #     coco_eval.evaluate_coco(testset, self.model)
    def test_process(self, image):
        with torch.no_grad():
            output = self.model(image)[-1]
            hm = output["hm"].sigmoid_()
            wh = output["wh"]
            reg = output["reg"]
            if self.args.flip_test:
                hm = (hm[0:1] + flip_tensor(hm[1:2])) / 2
                wh = (wh[0:1] + flip_tensor(wh[1:2])) / 2
            dets = decode(hm, wh, reg, K=self.args.top_K)
        return output, dets

    def test_post_process(self, dets, meta, scale=1):
        dets = dets.detach().cpu().numpy()
        dets = dets.reshape(1, -1, dets.shape[2])
        dets = ctdet_post_process(dets.copy(), [meta["c"]], [meta["s"]], meta["out_height"], meta["out_width"], self.args.num_classes)
        for j in range(1, self.num_classes):
            dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 5)
            dets[0][j][:, :4] /= scale
        return dets[0]

    def merge_output(self, detection, scales):
        result = {}
        ddd = np.zeros((1, self.args.num_classes, self.args.max_per_image, 5))
        for j in range(1, self.num_classes):
            result[j] = np.concatenate([det[j] for det in detection], axis=0).astype(np.float32)
            if self.args.nms or len(scales) > 1:
                soft_nms(result[j], Nt=0.5, method=2)
        scores = np.hstack([result[j][:, 4] for j in range(1, self.num_classes)])
        if len(scores) > self.args.max_per_image:
            kth = len(scores) - self.args.max_per_image
            thresh = np.partition(scores, kth)[kth]
            for j in range(1, self.num_classes):
                keep_inds = result[j][:, 4] >= thresh
                result[j] = result[j][keep_inds]

        return result  # {"1":[[],[]],"2":[[],[]]}


def decode(heat, wh, reg, K=100):
    """
    Args:
        heat:heatmap
        wh  : width and height
        reg : reg value
    Returns:
        detections : conclued bboxes, scores, classes
    Reference:
        This code is based on
        Objects as Points （https://github.com/xingyizhou/CenterNet）
        Copyright (c) 2019,  University of Texas at Austin
    """
    batch, cat, height, width = heat.size()

    heat = _nms(heat)

    scores, inds, clses, ys, xs = _topk(heat, K=K)

    reg = _transpose_and_gather_feat(reg, inds)
    reg = reg.view(batch, K, 2)
    xs = xs.view(batch, K, 1) + reg[:, :, 0:1]
    ys = ys.view(batch, K, 1) + reg[:, :, 1:2]

    wh = _transpose_and_gather_feat(wh, inds)

    wh = wh.view(batch, K, 2)
    clses = clses.view(batch, K, 1).float()
    scores = scores.view(batch, K, 1)
    bboxes = torch.cat([xs - wh[..., 0:1] / 2, ys - wh[..., 1:2] / 2, xs + wh[..., 0:1] / 2, ys + wh[..., 1:2] / 2], dim=2)
    detections = torch.cat([bboxes, scores, clses], dim=2)

    # detections:batch,K,4+1+1
    return detections


def ctdet_post_process(dets, c, s, h, w, num_classes):
    # dets:batch x max_dets x dim

    ret = []
    for i in range(dets.shape[0]):
        top_preds = {}
        dets[i, :, :2] = transform_preds(dets[i, :, 0:2], c[i], s[i], (w, h))
        dets[i, :, 2:4] = transform_preds(dets[i, :, 2:4], c[i], s[i], (w, h))
        label = dets[i, :, -1]
        for j in range(num_classes):
            inds = label == j
            top_preds[j + 1] = np.concatenate([dets[i, inds, :4].astype(np.float32), dets[i, inds, 4:5].astype(np.float32)], axis=1).tolist()
        ret.append(top_preds)
    return ret
