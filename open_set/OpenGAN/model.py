import sys
import os
import time
import torch
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from libs.metric import metric
from libs.base_model_d import base_model_d


def arguments():
    args = {"--tensorimg":100}
    return args

class OpenGAN(base_model_d):
    def __init__(self, parser):
        super().__init__(parser)
        self.imsize = 32
        self.batchsize=20
        parser.change_args("cls_imageSize", self.imsize)
        parser.change_args("cls_batchsize", self.batchsize)
        parser.add_args(arguments())
        self.args = parser.get_args()

        from open_set.OpenGAN.network import Generator, Discriminator
        self.netg = Generator(self.nc,self.args.tensorimg).to(self.device)
        self.netd = Discriminator(self.nc,self.args.known_classes_nums).to(self.device)
        self.net_dict = {"name": ['netg', 'netd'],
                        "network":[self.netg, self.netd]}

        self.loss_mse = nn.MSELoss()

        self.opt_g = optim.Adam(self.netg.parameters(), lr=0.02)
        self.opt_d = optim.Adam(self.netd.parameters(), lr=0.0003)    

    def train(self, train_loader, test_loader):
        if self.args.load_weights:
            self.load_weights(self.net_dict)
        best = float(0)
        for e in range(self.epochs):
            self.train_epoch(train_loader)
            if self.train_with_test:
                pred, real, score= self.inference(test_loader)
                metric_cluster = metric(pred, real, score, self.metric, self.args)
                best, self.best_trigger, self.indicator_for_best = metric_cluster.best_value_indicator(best, self.indicator_for_best)
                self.save_loggin_print(e+1, metric_cluster, best)
                self.save_weights(e+1)

    def train_epoch(self, dataloader):
        self.netg.train()
        self.netd.train()
        for data in tqdm(dataloader, total = len(dataloader), leave = False):            
            img, label = self.set_input(data)
            img, label = img.to(self.device), label.to(self.device)
            tensorimg = torch.rand(img.size(0),self.args.tensorimg,1,1)
            tensorimg = tensorimg.to(self.device)
            fake_img = self.netg(tensorimg).detach()
            labelf = (self.args.known_classes_nums)*torch.ones(size = (img.size(0),1),dtype = torch.int64,device = self.device)

            out_d = torch.cat([self.netd(img), self.netd(fake_img)], 0).to(self.device)
            
            label_d = torch.cat([label.reshape((img.size(0),1)), labelf], 0).to(self.device)
            label_d = torch.zeros(2*img.size(0),self.args.known_classes_nums+1,dtype = torch.int64,device = self.device).scatter_(1,label_d,1)
            
            loss_d_mse = self.loss_mse(out_d.float(), label_d.float())

            self.netd.zero_grad()                                                                                  
            loss_d_mse.backward(retain_graph = True)
            self.opt_d.step()

            label_g = torch.randint(0,self.args.known_classes_nums,size = (img.size(0),1),dtype = torch.int64,device = self.device)
            label_g = torch.zeros(img.size(0),self.args.known_classes_nums+1,dtype = torch.int64,device = self.device).scatter_(1,label_g,1)

            loss_g_mse = self.loss_mse(self.netd(fake_img).float(), label_g.float())

            self.netg.zero_grad()
            loss_g_mse.backward(retain_graph = True)
            self.opt_g.step()

    def test(self, dataloader):
        self.load_weights(self.net_dict)
        pred, real, score= self.inference(test_loader)
        metric_cluster = metric(pred, real, score, self.metric, self.args)
        self.best, _, self.indicator_for_best = metric_cluster.best_value_indicator(float(0), self.indicator_for_best)
        self.save_loggin_print("test", metric_cluster)

    def inference(self, dataloader):
        with torch.no_grad():
            pred = torch.zeros(size = (len(dataloader.dataset),), dtype=torch.float32)
            real = torch.empty(size = (len(dataloader.dataset),), dtype=torch.long)
            score = torch.zeros(size = (len(dataloader.dataset),self.args.known_classes_nums+1), dtype=torch.float32)

            self.netd.eval()
            for (i,data) in enumerate(dataloader):
                img, label = self.set_input(data)
                img, label = img.to(self.device), label.to(self.device)

                out = self.netd(img)
                out_d = torch.argmax(out, 1)
                out_D = torch.softmax(out,dim=1)

                pred[(i * dataloader.batch_size): i * dataloader.batch_size + img.size(0)] = out_d
                real[(i * dataloader.batch_size): i * dataloader.batch_size + img.size(0)] = label
                score[(i * dataloader.batch_size): i * dataloader.batch_size + img.size(0)] = out_D

            return pred, real, score