from tqdm import tqdm
import torch
from typing import Tuple
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn as nn
from libs.metric import metric
from libs.base_model_d import base_model_d
from libs.base_model_t import base_model_t

from sklearn.metrics import roc_auc_score
import torchextractor as tx
import os
from os.path import join
import time
from prettytable import PrettyTable
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import numpy as np
from libs.data import data
from one_class.LSA.networks import LS_A,BaseModule


def arguments():
    args = {"--net_type": 'base',
            "--test_mode": 'd'
            }
    return args

class LSA(base_model_d):
    
  

   
    def __init__(self,parser):
       
        super().__init__(parser)
        self.imsize = 100
        parser.change_args("cls_imageSize", self.imsize)
        self.args = parser.get_args()
        
        
        self.model = LS_A(input_shape=(1,100,100),code_length=64, cpd_channels=100).cuda().eval()
        self.net_dict = {"name": ['LSAMNIST'],
                        "network":[self.model]}
        # Set up loss function
        self.loss = LSALoss(cpd_channels=100)
        self.opt = optim.Adam(self.model.parameters(), lr=self.args.cls_lr)
        # self.scheduler = lr_scheduler.MultiStepLR(self.opt, milestones=self.args.cls_milestones, gamma=self.args.cls_gamma)
        
        self.macs, self.params = self.compute_macs_params(self.model,"LSAMNIST")
        # self.gpu_time = self.get_inference_time()
        # self.fps = self.get_fps()

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
       
        self.model.train()
        
        
        for data  in tqdm(dataloader,  total = len(dataloader), leave = False):
        
            
            img,label= self.set_input(data)
            
            
            
            img = img.to(self.device)
            
            x_r, z, z_dist = self.model(img)
            
            loss = self.loss(img, x_r, z, z_dist)

            self.model.zero_grad()
            loss.backward(retain_graph = True)
            self.opt.step()

        print(loss)
    def test(self, train_loader, test_loader):
        self.load_weights()
        pred, real, score= self.inference(test_loader)
        metric_cluster = metric(pred, real, score, self.metric, self.args)
        self.best, _, self.indicator_for_best = metric_cluster.best_value_indicator(float(0), self.indicator_for_best)
        assert self.args.phase == "test", ''' Call test function but phase is not testing. '''
        self.save_loggin_print("test", metric_cluster, self.best)
    def inference(self, dataloader):
        self.model.eval()
    
        self.model.eval()
        with torch.no_grad():
            pred = torch.zeros(size = (len(dataloader.dataset),), dtype=torch.float32)
            real = torch.empty(size = (len(dataloader.dataset),), dtype=torch.long)
            score = torch.zeros(size = (len(dataloader.dataset),), dtype=torch.float32)
            
            sample_llk = torch.zeros(size = (len(dataloader),))
            sample_rec = torch.zeros(size = (len(dataloader),))
            min_llk, max_llk, min_rec, max_rec = self.compute_normalizing_coefficients(dataloader)
          
        for  (i,data) in enumerate(tqdm(dataloader)):
      
            img,label= self.set_input(data)
      
            img, label = img.to(self.device), label.to(self.device)
           
            x_r, z, z_dist = self.model(img)
            
            self.loss(img, x_r, z, z_dist)
            
            sample_llk[i] = - self.loss.autoregression_loss
            sample_rec[i] = - self.loss.reconstruction_loss
              
            # Normalize scores
            sample_llk = normalize(sample_llk, min_llk, max_llk)
            sample_rec = normalize(sample_rec, min_rec, max_rec)
            # Compute the normalized novelty score
            sample_ns = novelty_score(sample_llk, sample_rec)
            
            out_ns=sample_ns[i]

            score[(i * dataloader.batch_size): i * dataloader.batch_size + img.size(0)] =out_ns
            real[(i * dataloader.batch_size): i * dataloader.batch_size + img.size(0)] = label.reshape(img.size(0))
          
        pred = torch.zeros(size = (len(dataloader.dataset),), dtype=torch.float32)
        for i in range(real.size(0)):
            pred[i] = 1 if score[i] > score.median().item() else 0 
    
        return  pred,real,score


    
    
    def compute_normalizing_coefficients(self,dataloader):
      
        """
        Computes normalizing coeffients for the computation of the Novelty score (Eq. 9-10).
        :param cl: the class to be considered normal.
        :return: a tuple of normalizing coefficients in the form (llk_min, llk_max, rec_min, rec_max).
        """
        
        
        
        sample_llk = torch.zeros(size =(len(dataloader),))
        sample_rec = torch.zeros(size=(len(dataloader),))
        for  (i,data)  in enumerate(dataloader):
          
            img,label= self.set_input(data)
            
            
            
            img = img.to(self.device)
          
            x_r, z, z_dist = self.model(img)
           
            
            
            self.loss(img, x_r, z, z_dist)  

            sample_llk[i] = - self.loss.autoregression_loss
            sample_rec[i] = - self.loss.reconstruction_loss
           
        return sample_llk.min(), sample_llk.max(), sample_rec.min(), sample_rec.max()
    

def normalize(samples, min, max):
    # type: (np.ndarray, float, float) -> np.ndarray
  
    
    return (samples - min) / (max - min)    
def normalize_ns(orign_data):
    d_min=orign_data.min()
    if d_min<0:
        orign_data+=torch.abs(d_min)
        d_min-orign_data.min()
    d_max=orign_data.max()
    dst=d_max-d_min
    norm_data=(orign_data-d_min)/dst
    return norm_data

    
def novelty_score(sample_llk_norm, sample_rec_norm):
  

    # Sum
    ns = (sample_llk_norm + sample_rec_norm)

    return ns
class AutoregressionLoss(BaseModule):
    """
    Implements the autoregression loss.
    Given a representation and the estimated cpds, provides
    the log-likelihood of the representation under the estimated prior.
    """
    def __init__(self, cpd_channels):
        # type: (int) -> None
        """
        Class constructor.

        :param cpd_channels: number of bins in which the multinomial works.
        """
        super(AutoregressionLoss, self).__init__()

        self.cpd_channels = cpd_channels

        # Avoid nans
        self.eps = torch.finfo(float).eps

    def forward(self, z, z_dist):
        # type: (torch.Tensor, torch.Tensor) -> torch.Tensor
        """
        Forward propagation.

        :param z: the batch of latent representations.
        :param z_dist: the batch of estimated cpds.
        :return: the mean log-likelihood (averaged along the batch axis).
        """
        z_d = z.detach()
        import torch.nn.functional as F
        # Apply softmax
        z_dist = F.softmax(z_dist, dim=1)

        # Flatten out codes and distributions
        z_d = z_d.view(len(z_d), -1).contiguous()
        z_dist = z_dist.view(len(z_d), self.cpd_channels, -1).contiguous()

        # Log (regularized), pick the right ones
        z_dist = torch.clamp(z_dist, self.eps, 1 - self.eps)
        log_z_dist = torch.log(z_dist)
        index = torch.clamp(torch.unsqueeze(z_d, dim=1) * self.cpd_channels, min=0,
                            max=(self.cpd_channels - 1)).long()
        selected = torch.gather(log_z_dist, dim=1, index=index)
        selected = torch.squeeze(selected, dim=1)

        # Sum and mean
        S = torch.sum(selected, dim=-1)
        nll = - torch.mean(S)

        return nll
class ReconstructionLoss(BaseModule):
    """
    Implements the reconstruction loss.
    """
    def __init__(self):
        # type: () -> None
        """
        Class constructor.
        """
        super(ReconstructionLoss, self).__init__()

    def forward(self, x, x_r):
        # type: (torch.Tensor, torch.Tensor) -> torch.Tensor
        """
        Forward propagation.

        :param x: the batch of input samples.
        :param x_r: the batch of reconstructions.
        :return: the mean reconstruction loss (averaged along the batch axis).
        """
        L = torch.pow((x - x_r), 2)

        while L.dim() > 1:
            L = torch.sum(L, dim=-1)

        return torch.mean(L)



class LSALoss(nn.Module):
    """
    Implements the loss of a LSA model.
    It is a sum of the reconstruction loss and the autoregression loss.
    """
    def __init__(self, cpd_channels, lam=1):
        # type: (int, float) -> None
        """
        Class constructor.

        :param cpd_channels: number of bins in which the multinomial works.
        :param lam: weight of the autoregression loss.
        """
        super(LSALoss, self).__init__()

        self.cpd_channels = cpd_channels
        self.lam = lam

        # Set up loss modules
        self.reconstruction_loss_fn = ReconstructionLoss()
        self.autoregression_loss_fn = AutoregressionLoss(cpd_channels)

        # Numerical variables
        self.reconstruction_loss = None
        self.autoregression_loss = None
        self.total_loss = None

    def forward(self, x, x_r, z, z_dist):
        # type: (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor) -> torch.Tensor
        """
        Forward propagation.

        :param x: the batch of input samples.
        :param x_r: the batch of reconstructions.
        :param z: the batch of latent representations.
        :param z_dist: the batch of estimated cpds.
        :return: the loss of the model (averaged along the batch axis).
        """
        # Compute pytorch loss
        rec_loss = self.reconstruction_loss_fn(x, x_r)
        arg_loss = self.autoregression_loss_fn(z, z_dist)
        tot_loss = rec_loss + self.lam * arg_loss

        # Store numerical
        self.reconstruction_loss = rec_loss.item()
        self.autoregression_loss = arg_loss.item()
        self.total_loss = tot_loss.item()

        return tot_loss
