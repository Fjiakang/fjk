# Human Motion Recognition With Limited Radar Micro-Doppler Signatures
# X. Li, Y. He, F. Fioranelli, X. Jing, A. Yarovoy and Y. Yang, "Human Motion Recognition With Limited Radar Micro-Doppler Signatures," in IEEE Transactions on Geoscience and Remote Sensing, doi: 10.1109/TGRS.2020.3028223.

import torch
import torch.nn as nn
import torch.optim as optim
from libs.base_model_d import base_model_d
from libs.metric import metric
from multi_class.mnet.networks import m_net  # ,weights_init
from tqdm import tqdm


class mnet(base_model_d):
    def __init__(self, parser):
        super().__init__(parser)
        self.imsize = 100
        parser.change_args("cls_imageSize", self.imsize)
        self.args = parser.get_args()
        self.model = m_net(self.args.cls_nums, self.imsize, self.nc).to(self.device)
        self.net_dict = {"name": ['mnet'], "network": [self.model]}
        self.loss_ce = nn.CrossEntropyLoss().to(self.device)
        self.opt = optim.Adam(self.model.parameters(), lr=self.args.cls_lr)
        self.macs, self.params = self.netinfo.compute_macs_params(self.model, "mnet",
                                                                  [self.nc, self.imsize, self.imsize])  #
        self.lossdict = dict(loss=[])
        self.paramdict = dict(epoch=[])
        self.valuedictlist = [['loss', self.lossdict], ['param', self.paramdict]]  #

    def train(self, train_loader, test_loader):
        self.save_args()
        if self.args.load_weights:
            self.load_weights()
        best = float(0)
        for e in range(self.epochs):
            self.paramdict['epoch'].append(e + 1)
            self.train_epoch(train_loader)
            if (self.train_with_test) and (e % self.train_with_test == 0 or e == self.epochs - 1):
                pred, real, score, embedding = self.inference(test_loader)
                metric_cluster = metric(pred, real, score, self.metric, self.args.metric_average,
                                        self.args.metric_pos_label)
                best, self.best_trigger, self.indicator_for_best = metric_cluster.best_value_indicator(best,
                                                                                                       self.indicator_for_best)
                self.save_loggin_print(e + 1, metric_cluster, best, self.valuedictlist)  #
                self.save_weights(e + 1)
                self.save_result_img(e + 1, best, metric_cluster, embedding, real, self.valuedictlist,
                                     model_layer_list=[{'model': self.model, 'layerlist': ["conv6"]}])  #

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
        self.lossdict['loss'].append(loss.detach().item())
        # self.scheduler.step()

    def test(self, test_loader):
        self.load_weights()
        pred, real, score, embedding = self.inference(test_loader)
        metric_cluster = metric(pred, real, score, self.metric, self.args)
        self.best, _, self.indicator_for_best = metric_cluster.best_value_indicator(float(0), self.indicator_for_best)
        self.save_loggin_print("test", metric_cluster, self.best, self.valuedictlist)
        self.save_result_img('test', self.best_trigger, metric_cluster, embedding, real, self.valuedictlist,
                             model_layer_list={"model": self.model, "layerlist": ["conv5"]})
        assert self.args.phase == "test", """ Call test function but phase is not testing. """

    def inference(self, dataloader):
        with torch.no_grad():
            pred = torch.zeros(size=(len(dataloader.dataset),), dtype=torch.float32)
            real = torch.empty(size=(len(dataloader.dataset),), dtype=torch.long)
            score = torch.zeros(size=(len(dataloader.dataset), self.args.cls_nums), dtype=torch.float32)
            embedding = torch.zeros(size=(len(dataloader.dataset), self.args.cls_nums), dtype=torch.float32)

            self.model.eval()

            for (i, data) in enumerate(dataloader):
                img, label = self.set_input(data)
                img, label = img.to(self.device), label.to(self.device)
                out = self.model(img)
                out_d = torch.argmax(out, 1)  # 索引
                pred[i * dataloader.batch_size: i * dataloader.batch_size + img.size(0)] = out_d
                real[i * dataloader.batch_size: i * dataloader.batch_size + img.size(0)] = label
                score[(i * dataloader.batch_size): i * dataloader.batch_size + img.size(0)] = nn.functional.softmax(out,
                                                                                                                    dim=1)
                embedding[(i * dataloader.batch_size): i * dataloader.batch_size + img.size(0)] = out

            return pred, real, score, embedding