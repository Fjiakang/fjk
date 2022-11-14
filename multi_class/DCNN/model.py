# 引用文章：Human Detection and Activity Classification Based on Micro-Doppler Signatures Using Deep Convolutional Neural Networks
# Kim Y, Moon T. Human detection and activity classification based on micro-Doppler signatures using deep convolutional neural networks[J]. IEEE geoscience and remote sensing letters, 2015, 13(1): 8-12.
import torch
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
#import torch.optim.lr_scheduler as lr_scheduler
from multi_class.DCNN.networks import D_CNN
from libs.base_model_d import base_model_d
from libs.metric import metric



class DCNN(base_model_d):
    def __init__(self, parser):
        super().__init__(parser)
        # parser.add_args(arguments())
        # self.args = parser.get_args()

        self.imsize = 100
        parser.change_args("cls_imageSize", self.imsize)
        self.args = parser.get_args()

        self.model = D_CNN(self.args.cls_nums, self.imsize,self.nc).to(self.device)
        self.net_dict = {"name": ['DCNN'],
                         "network": [self.model]}

        self.loss_ce = nn.CrossEntropyLoss().to(self.device)
        self.opt = optim.Adam(self.model.parameters(), lr=self.args.cls_lr)
        # self.scheduler = lr_scheduler.MultiStepLR(self.opt, milestones=self.args.cls_milestones, gamma=self.args.cls_gamma)

        self.macs, self.params = self.compute_macs_params(self.model, "DCNN")
        # self.gpu_time = self.get_inference_time()
        # self.fps = self.get_fps()

    def train(self, train_loader, test_loader):
        if self.args.load_weights:
            self.load_weights()
        best = float(0)
        for e in range(self.epochs):
            self.train_epoch(train_loader)
            if (self.train_with_test) and (e % self.train_with_test == 0 or e == self.epochs - 1):
                pred, real, score = self.inference(test_loader)
                metric_cluster = metric(pred, real, score, self.metric, self.args)
                best, self.best_trigger, self.indicator_for_best = metric_cluster.best_value_indicator(best,
                                                                                                       self.indicator_for_best)
                self.save_loggin_print(e + 1, metric_cluster, best)
                self.save_weights(e + 1)

    def test(self, train_loader, test_loader):
        self.load_weights()
        pred, real, score = self.inference(test_loader)
        metric_cluster = metric(pred, real, score, self.metric, self.args)
        self.best, _, self.indicator_for_best = metric_cluster.best_value_indicator(float(0), self.indicator_for_best)
        assert self.args.phase == "test", ''' Call test function but phase is not testing. '''
        self.save_loggin_print("test", metric_cluster, self.best)

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
        # self.scheduler.step()

    def inference(self, dataloader):
        with torch.no_grad():
            pred = torch.zeros(size=(len(dataloader.dataset),), dtype=torch.float32)
            real = torch.empty(size=(len(dataloader.dataset),), dtype=torch.long)
            score = torch.zeros(size=(len(dataloader.dataset), self.args.cls_nums), dtype=torch.float32)

            self.model.eval()
            for (i, data) in enumerate(tqdm(dataloader, total=len(dataloader), leave=False)):
                img, label = self.set_input(data)
                img, label = img.to(self.device), label.to(self.device)
                out = self.model(img)
                out_d = torch.argmax(out, 1)
                pred[(i * dataloader.batch_size): i * dataloader.batch_size + img.size(0)] = out_d
                score[(i * dataloader.batch_size): i * dataloader.batch_size + img.size(0)] = nn.functional.softmax(out,
                                                                                                                    dim=1)
                real[(i * dataloader.batch_size): i * dataloader.batch_size + img.size(0)] = label
            return pred, real, score
