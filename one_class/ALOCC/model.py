from tqdm import tqdm
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn as nn
from libs.metric import metric
from libs.base_model_d import base_model_d
from one_class.ALOCC.networks import NetD, NetR, weights_init
import torchextractor as tx
import os
import time
def arguments():
    args = {"--net_type": 'base',
            "--test_mode": 'd'
            }
    return args
class ALOCC(base_model_d):
    def __init__(self, parser):
        super().__init__(parser)
        self.imsize = 61
        parser.change_args("cls_imageSize", self.imsize)
        parser.add_args(arguments())
        self.args = parser.get_args()

        self.netr = NetR(self.args).to(self.device)
        self.netd = NetD(self.args).to(self.device)
        self.netr.apply(weights_init)
        self.netd.apply(weights_init)

        self.net_dict = {"name": ['netr', 'netd'],
                        "network":[self.netr, self.netd]}

        self.test_mode = self.args.test_mode
        self.loss_bce = nn.BCELoss()
        self.loss_mse = nn.MSELoss()

        self.opt_r = optim.Adam(self.netr.parameters(), lr=self.lr, betas=(0.5, 0.999))
        self.opt_d = optim.Adam(self.netd.parameters(), lr=self.lr, betas=(0.5, 0.999))

        self.macs_r, self.params_r = self.compute_macs_params(self.netr,"netr")
        self.macs_d, self.params_d = self.compute_macs_params(self.netd,"netd")

    def train(self, train_loader, test_loader):
        
        if self.args.load_weights:
            self.load_weights(self.net_dict)
        best = float(0)
        for e in range(self.epochs):
            self.train_epoch(train_loader)
            if (self.train_with_test) and (e%self.train_with_test==0 or e == self.epochs-1):
                pred, real, score = self.inference(test_loader)
                metric_cluster = metric(pred, real, score, self.metric, self.args)
                best, self.best_trigger, self.indicator_for_best = metric_cluster.best_value_indicator(best, self.indicator_for_best)
                self.save_loggin_print(e+1, metric_cluster, best)
                self.save_weights(e+1)
    def test(self, train_loader, test_loader):
        self.load_weights(self.net_dict)
        pred, real, score= self.inference(test_loader)
        metric_cluster = metric(pred, real, score, self.metric, self.args)
        self.best, _, self.indicator_for_best = metric_cluster.best_value_indicator(float(0), self.indicator_for_best)
        assert self.args.phase == "test", ''' Call test function but phase is not testing. '''
        self.save_loggin_print("test", metric_cluster, self.best)

    def train_epoch(self, dataloader):
        self.netr.train()
        self.netd.train()
        for data in tqdm(dataloader, total = len(dataloader), leave = False):
            #import pdb;pdb.set_trace()
            img, _ = self.set_input(data)    
                
            img = img.to(self.device)
            randn_noise = torch.randn(img.size(0), self.args.nc, self.imsize, self.imsize).to(self.device)
            img_denoise = self.netr(img + randn_noise).to(self.device)

            out_d_denoise = self.netd(img_denoise)
            out_d_img = self.netd(img)
            out_d = torch.cat([out_d_denoise, out_d_img], 0)

            label = torch.ones(size = (img.size(0),),dtype = torch.float32, device = self.device)
            labelf = torch.zeros(size = (img.size(0),),dtype = torch.float32, device = self.device)
            label_d = torch.cat([labelf, label], 0)

            loss_d = self.loss_bce(out_d, label_d)
            
            self.opt_d.zero_grad()
            loss_d.backward(retain_graph = True)
            self.opt_d.step()

            out_d_denoise_1 = self.netd(img_denoise)
            label = torch.ones(size = (img.size(0),),dtype = torch.float32, device = self.device)
            loss_r_mse = self.loss_mse(img, img_denoise)
            loss_r_bce = self.loss_bce(out_d_denoise_1, label)
            loss_r = 0.4*loss_r_mse + loss_r_bce

            self.opt_r.zero_grad()
            loss_r.backward(retain_graph = True)
            self.opt_r.step()

        return self.netr, self.netd
    
    def inference(self, dataloader):
        self.netd.eval()
        self.netr.eval()
        if self.test_mode == "dr":
            score, real = self.test_dr(dataloader)
        elif self.test_mode == "d":
            score, real = self.test_d(dataloader)
        elif self.test_mode == "r":
            score, real = self.test_r(dataloader)
        pred = torch.zeros(size = (len(dataloader.dataset),), dtype=torch.float32)
        for i in range(real.size(0)):
            pred[i] = 1 if score[i] > score.median().item() else 0    
       
        return pred,real,score

    def test_dr(self,dataloader):
        with torch.no_grad():
            score = torch.zeros(size = (len(dataloader.dataset),), dtype=torch.float32)
            real = torch.empty(size = (len(dataloader.dataset),), dtype=torch.long)
            for (i,data) in enumerate(dataloader):
               
                img,label = self.set_input(data)
                img, label = img.to(self.device), label.to(self.device)
                randn_noise = torch.randn(img.size(0), self.args.nc, self.imsize, self.imsize).to(self.device)
                img_denoise = self.netr(img + randn_noise)
                out_d = self.netd(img_denoise).squeeze()
                
                score[i * dataloader.batch_size: i * dataloader.batch_size + img.size(0)] = out_d
                real[i * dataloader.batch_size: i * dataloader.batch_size + img.size(0)] = label.reshape(img.size(0))
        return score, real

    def test_d(self,dataloader):
        #import pdb;pdb.set_trace()
        with torch.no_grad():
            score = torch.zeros(size = (len(dataloader.dataset),), dtype=torch.float32)
            real = torch.empty(size = (len(dataloader.dataset),), dtype=torch.long)
            for (i,data) in enumerate(dataloader):
               
                img,label = self.set_input(data)
                img, label = img.to(self.device), label.to(self.device)
                randn_noise = torch.randn(img.size(0), self.args.nc, self.imsize, self.imsize).to(self.device)
                img_denoise = self.netr(img + randn_noise)
                out_d = self.netd(img + randn_noise)
                
                score[i * dataloader.batch_size: i * dataloader.batch_size + img.size(0)] = out_d
                real[i * dataloader.batch_size: i * dataloader.batch_size + img.size(0)] = label.reshape(img.size(0))
               
        return score, real

    def test_r(self,dataloader):
        with torch.no_grad():
            score = torch.empty(size = (len(dataloader.dataset),), dtype=torch.float32)
            real = torch.empty(size = (len(dataloader.dataset),), dtype=torch.long)
            for (i,data) in enumerate(dataloader):
                
                img,label = self.set_input(data)
                img, label = img.to(self.device), label.to(self.device)
                randn_noise = torch.randn(img.size(0), self.args.nc, self.imsize, self.imsize).to(self.device)
                img_denoise = self.netr(img + randn_noise)
                out_d = self.loss_mse(img, img_denoise)

                score[i * self.batchsize: i * self.batchsize + img.size(0)] = out_d
                score[i * dataloader.batch_size: i * dataloader.batch_size + img.size(0)] = out_d
                real[i * dataloader.batch_size: i * dataloader.batch_size + img.size(0)] = label.reshape(img.size(0))
        return score, real
#python main.py one_class --cls_network ALOCC --control_print --cls_dataroot /home/tju --cls_dataset data