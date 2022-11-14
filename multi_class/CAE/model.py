# 引用文章：Deep Neural Network Initialization Methods for Micro-Doppler Classification With Low Training Sample Support
# Seyfioğlu M S, Gürbüz S Z. Deep neural network initialization methods for micro-Doppler classification with low training sample support[J]. IEEE Geoscience and Remote Sensing Letters, 2017, 14(12): 2462-2466.
# https://ieeexplore.ieee.org/document/8119733

# @article{seyfiouglu2017deep,
#   title={Deep Neural Network Initialization Methods for Micro-Doppler Classification With Low Training Sample Support},
#   author={Seyfio{\u{g}}lu, Mehmet Sayg{\i}n and G{\"u}rb{\"u}z, Sevgi Z{\"u}beyde},
#   journal={IEEE Geoscience and Remote Sensing Letters},
#   volume={14},
#   number={12},
#   pages={2462--2466},
#   year={2017}
# }

import torch
import torch.nn as nn
import torch.optim as optim
import torchextractor as tx
from libs.base_model_d import base_model_d
from libs.metric import metric
from multi_class.CAE.networks import CAE_te, CAE_tr  # ,weights_init
from tqdm import tqdm


def arguments():
    args = {
        "--pretrain_epochs":30,
    }
    return args


class CAE(base_model_d):
    def __init__(self, parser):
        super().__init__(parser)
        parser.add_args(arguments())
        self.args = parser.get_args()

        self.imsize = 110
        parser.change_args("cls_imageSize", self.imsize)
        self.args = parser.get_args()

        self.model_pre = CAE_tr(self.args.cls_nums, self.nc).to(self.device)
        self.model = CAE_te(self.args.cls_nums, self.nc).to(self.device)
        self.net_dict = {"name": ["ende_tr", "ende_te"], "network": [self.model_pre, self.model]}

        self.loss_ce = nn.CrossEntropyLoss().to(self.device)
        self.loss_mse = nn.MSELoss().to(self.device)

        self.opt_pre = optim.Adam(self.model_pre.parameters(), lr=self.args.cls_lr)
        self.opt = optim.Adam(self.model.parameters(), lr=self.args.cls_lr)

        self.macs, self.params = self.netinfo.compute_macs_params(self.model, "CAE_te", [self.nc, self.imsize, self.imsize])
        # self.gpu_time = self.get_inference_time()
        # self.fps = self.get_fps()
        self.lossdic = dict(loss=[])
        self.paramdict = dict(epoch=[])
        self.valuedictlist = [["loss", self.lossdic], ["param", self.paramdict]]


    def train(self, train_loader, test_loader):
        self.save_args()
        if self.args.load_weights:
            self.load_weights()
        print("Unsupervised Pre-training...")
        best = float(0)
        for e in range(self.args.pretrain_epochs):
            self.train_epoch_pre(train_loader)
        self.model.load_state_dict(self.model_pre.state_dict())
        print("Training...")

        best = float(0)
        for e in range(self.epochs):
            self.paramdict["epoch"].append(e + 1)
            self.train_epoch(train_loader)
            if (self.train_with_test) and (e % self.train_with_test == 0 or e == self.epochs - 1):
                pred, real, score, embedding = self.inference(test_loader)
                metric_cluster = metric(pred, real, score, self.metric, self.args.metric_average, self.args.metric_pos_label)
                best, self.best_trigger, self.indicator_for_best = metric_cluster.best_value_indicator(best, self.indicator_for_best)
                self.save_loggin_print(e + 1, metric_cluster, best, self.valuedictlist)
                self.save_weights(e + 1)
                self.save_result_img(e + 1, self.best_trigger, metric_cluster, embedding, real, self.valuedictlist, model_layer_list=[{"model": self.model, "layerlist": ["layer1_a"]}])



    def test(self, train_loader, test_loader):
        self.load_weights()
        self.save_args()
        pred, real, score, embedding = self.inference(test_loader)
        metric_cluster = metric(pred, real, score, self.metric, self.args.metric_average, self.args.metric_pos_label)
        best, self.best_trigger, self.indicator_for_best = metric_cluster.best_value_indicator(float(0), self.indicator_for_best)
        self.save_loggin_print("test", metric_cluster, best, self.valuedictlist)
        self.save_result_img("test", self.best_trigger, metric_cluster, embedding, real, self.lossdic, model_layer_list=[{"model": self.model, "layerlist": ["layer1_a"]}])

    def train_epoch(self, dataloader):
        self.model.train()
        for data in tqdm(dataloader, total=len(dataloader), leave=False):
            img, label = self.set_input(data)
            img, label = img.to(self.device), label.to(self.device)

            pred = self.model(img)
            loss = self.loss_ce(pred, label)

            self.model.zero_grad()
            loss.backward(retain_graph=True)
            self.opt.step()
        self.lossdic["loss"].append(loss.detach().item())
        # self.scheduler.step()

    def train_epoch_pre(self, dataloader):
        self.model_pre.train()
        for data in tqdm(dataloader, total=len(dataloader), leave=False):
            img, label = self.set_input(data)
            img, label = img.to(self.device), label.to(self.device)

            pred = self.model_pre(img)
            loss = self.loss_mse(img, pred)

            self.model_pre.zero_grad()
            loss.backward(retain_graph=True)
            self.opt_pre.step()
        # self.scheduler.step()

    def inference(self, dataloader):
        with torch.no_grad():
            pred = torch.zeros(size=(len(dataloader.dataset),), dtype=torch.float32)
            real = torch.empty(size=(len(dataloader.dataset),), dtype=torch.long)
            score = torch.zeros(size=(len(dataloader.dataset), self.args.cls_nums), dtype=torch.float32)
            embedding = torch.zeros(size=(len(dataloader.dataset), self.args.cls_nums), dtype=torch.float32)

            self.model.eval()
            for (i, data) in enumerate(tqdm(dataloader, total=len(dataloader), leave=False)):
                img, label = self.set_input(data)
                img, label = img.to(self.device), label.to(self.device)
                out = self.model(img)
                out_d = torch.argmax(out, 1)
                pred[(i * dataloader.batch_size) : i * dataloader.batch_size + img.size(0)] = out_d
                score[(i * dataloader.batch_size) : i * dataloader.batch_size + img.size(0)] = nn.functional.softmax(out, dim=1)
                real[(i * dataloader.batch_size) : i * dataloader.batch_size + img.size(0)] = label
                embedding[(i * dataloader.batch_size) : i * dataloader.batch_size + img.size(0)] = out

            return pred, real, score, embedding
