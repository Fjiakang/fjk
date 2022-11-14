import torch
import torchextractor as tx
import os
import torch.nn as nn
from libs.metric import metric
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from ptflops import get_model_complexity_info
from libs.Visualizer import Visualizer

def arguments():
    args = {"--cls_epochs": 30,
    "--cls_lr": 2e-4,
    "--metric_init": 0,
    "--time_avg":10,
    "--test_weight_choose":"best",
    "--weight_path":"None",
    "--test_interval":1}
    return args

def arguments_multi_class():
    args = {'--cls_nums': 10
            }
    return args

def arguments_one_class():
    args = {'--known_lbl': '0'
            }
    return args
def arguments_open_set():
    args = {'--known_lbl': 'a01'
            }
    return args

class base_model_d(nn.Module):
    def __init__(self, parser):
        super(base_model_d, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        torch.backends.cudnn.benchmark = True if not self.device == "cpu" else False
        parser.add_args(arguments())
        self.args = parser.get_args()
        self.is_deep_model = True
        self.nc = self.args.nc
        if self.args.cls_type == "multi_class":
            parser.add_args(arguments_multi_class())
            self.args = parser.get_args()
            parser.cls_nums = self.args.cls_nums
        elif self.args.cls_type == 'one_class':
            parser.add_args(arguments_one_class())
            self.args = parser.get_args()
            assert "[" not in self.args.known_lbl[0], '''your known_lbl type is wrong. e.g. '0' '1' '''
            if isinstance(self.args.known_lbl, str):
                parser.change_args("known_lbl", self.args.known_lbl.split(' '))
                # parser.parser.set_defaults(known_lbl = self.args.known_lbl.split(' '))
        elif self.args.cls_type == 'open_set':
            parser.add_args(arguments_open_set())
            self.args = parser.get_args()
            assert "[" not in self.args.known_lbl[0], '''your known_lbl type is wrong. e.g. '0' '1' '''
            if isinstance(self.args.known_lbl, str):
                parser.change_args("known_lbl", self.args.known_lbl.split(' '))
            parser.add_args({'--known_classes_nums': len(self.args.known_lbl)})
            self.args = parser.get_args()
            self.known_classes_nums = self.args.known_classes_nums
            
        self.train_with_test = self.args.test_interval
        self.starter,self.ender = torch.cuda.Event(enable_timing = True), torch.cuda.Event(enable_timing = True) 
        self.batchsize = self.args.cls_batchsize
        self.dataset = self.args.cls_dataset
        self.network = self.args.cls_network
        self.epochs = self.args.cls_epochs
        self.metric = self.args.metric    
        self.lr = self.args.cls_lr
        self.time_avg = self.args.time_avg
        self.class_type = self.args.cls_type
        self.indicator_for_best = None
        self.vis = Visualizer(parser)

    # def addvis(self, vis):
    #     self.vis = vis
    #     pass

    def train(self, train_dataloader, test_dataloader):

        pass

    def test(self, train_dataloader, test_dataloader):

        pass

    def inference(self, dataloader):

        raise NotImplementedError

    def save_weights(self,epoch):
        dir_name = ''
        if self.args.control_save_end:
            if self.best_trigger or self.final_epoch_trigger:
                dir_name = ''
                if self.args.cls_type  in ["one_class", "open_set"]:
                    for i in range(len(self.args.known_lbl)):
                        dir_name += self.args.known_lbl[i] if i == len(self.args.known_lbl)-1 else self.args.known_lbl[i]+'_'
                    weight_dir = os.path.join(self.args.outf, os.path.basename(os.getcwd()), self.args.cls_dataset, self.args.cls_type, self.args.cls_network, dir_name,'weight')
                else:
                    weight_dir = os.path.join(self.args.outf, os.path.basename(os.getcwd()), self.args.cls_dataset, self.args.cls_type, self.args.cls_network, 'weight')
                if not os.path.exists(weight_dir):
                    os.makedirs(weight_dir)
                for i in range(len(self.net_dict['name'])):
                    torch.save({'epoch':epoch + 1, 'net_params':self.net_dict['network'][i].state_dict()},'%s/{}_current.pth'.format(self.net_dict['name'][i]) % (weight_dir))
                if self.best_trigger:
                    for i in range(len(self.net_dict['name'])):
                        cmd = "cp {}/{}_current.pth {}/{}_best.pth".format(weight_dir, self.net_dict['name'][i], weight_dir, self.net_dict['name'][i])
                        os.system(cmd)
                if self.final_epoch_trigger:
                    for i in range(len(self.net_dict['name'])):
                        cmd = "mv {}/{}_current.pth {}/{}_final.pth".format(weight_dir, self.net_dict['name'][i], weight_dir, self.net_dict['name'][i])
                        os.system(cmd)

    def load_weights(self):
        dir_name = ''
        for i in range(len(self.net_dict['name'])):
            if self.args.weight_path == "None":
                if self.args.cls_type  in ["one_class", "open_set"]:
                    for i in range(len(self.args.known_lbl)):
                        dir_name += self.args.known_lbl[i]+'_'
                    weight_dir = os.path.join(self.args.outf, os.path.basename(os.getcwd()), self.args.cls_dataset, self.args.cls_type, self.args.cls_network, dir_name,'weight')         
                else:
                    weight_dir = os.path.join(self.args.outf, os.path.basename(os.getcwd()), self.args.cls_dataset, self.args.cls_type, self.args.cls_network, 'weight')
                weight_dir = os.path.join(weight_dir,self.net_dict['name'][i]+'_'+self.args.test_weight_choose+'.pth') if self.args.test_weight_choose == "best" or self.args.test_weight_choose == "final" else os.path.join(weight_dir,self.net_dict['name'][i]+'_current.pth')
            else:
                weight_dir = self.args.weight_path
            self.pretrain_dict = torch.load(weight_dir)
            self.net_dict['network'][i].load_state_dict(self.pretrain_dict['net_params'])

    # def best_result_saver(self):
    #     higher_is_best = ['ACC','AUC']
    #     lower_is_best = ['EER']
    #     if self.indicator_for_best is None:
    #         for self.indicator_for_best in range(0,len(self.metric_values)):
    #             if not self.metric_values[self.indicator_for_best] is None:
    #                 self.best = float(0) if self.metric[self.indicator_for_best] in higher_is_best else float(1)
    #                 break

    #     comparison_operator = '<' if self.metric[self.indicator_for_best] in higher_is_best else '>'

    #     exec("self.best_trigger = True if self.best {} self.metric_values[self.indicator_for_best] else False".format(comparison_operator))
    #     self.best = self.metric_values[self.indicator_for_best] if self.best_trigger else self.best

    # def compute_metric(self,epoch,pred,real):
    #     self.metric_class = metric(pred,real)
    #     self.metric_values = self.metric_class.get_metric(self.metric)

    #     self.best_result_saver()
    #     self.final_epoch_trigger = True if epoch == self.epochs else False

    def save_loggin_print(self, current_epoch, metric_cluster, best):
        self.final_epoch_trigger = True if current_epoch == self.epochs else False
        if self.args.control_print or self.args.control_save:
            self.vis.loggin(metric_cluster, current_epoch, best, self.indicator_for_best)
        if self.args.control_save:
            self.vis.save()
        if self.args.control_print:
            self.vis.output()
        if self.final_epoch_trigger and self.args.control_save:
            self.vis.plot_menu(best)

    def set_input(self,input):
        if isinstance(input[1],tuple):
            import numpy as np
            input[1] = torch.from_numpy(np.array([int(x) for x in np.array(input[1])]))
        return input

    def compute_macs_params(self, net, net_name):    
        self.imsize2 = self.args.cls_imageSize if self.args.cls_imageSize2 ==-1 else self.args.cls_imageSize2
        macs, params = get_model_complexity_info(net, (self.nc, self.args.cls_imageSize, self.imsize2))
        self.vis.plot_macs_params(macs, params, net_name)
        return macs, params

    def create_empty_loader(self):
        self.imsize2 = self.args.cls_imageSize if self.args.cls_imageSize2 ==-1 else self.args.cls_imageSize2
        sample_tensor = torch.zeros(self.batchsize*self.time_avg, self.nc, self.args.cls_imageSize,self.imsize2)
        label_tensor = torch.zeros(self.batchsize*self.time_avg)
        empty_dataset = TensorDataset(sample_tensor,label_tensor)
        empty_dataloader = DataLoader(empty_dataset,shuffle = True,batch_size=self.batchsize)
        return empty_dataloader

    def compute_time_fps(self, empty_dataloader):

        if self.inference(empty_dataloader) == None:
            raise Exception("You need define an inference function in your model.")

        else:
            self.starter.record()
            self.inference(empty_dataloader)
            self.ender.record()
            torch.cuda.synchronize()
            self.inference_time = self.starter.elapsed_time(self.ender)
            self.gpu_time = self.inference_time/self.time_avg
            self.fps = 1000/ self.gpu_time

    def get_inference_time(self):
        self.compute_time_fps(self.create_empty_loader())
        self.vis.plot_inference_time(self.gpu_time)
        return self.gpu_time

    def get_fps(self):
        self.compute_time_fps(self.create_empty_loader())
        self.vis.plot_fps(self.fps)
        return self.fps

    # def get_last_