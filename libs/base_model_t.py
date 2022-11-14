import os, pickle
import numpy as np
from libs.Visualizer import Visualizer
from libs.metric import metric
from tqdm import tqdm
import time

def arguments():
    args = {"--cls_epochs": 1,
    "--metric_init": 0,
    "--time_avg":10,
    "--weight_path":"None",
    "--test_interval":1}
    return args

def arguments_multi_class():
    args = {'--cls_nums': 2}
    return args

class base_model_t():
    def __init__(self, parser):
        super(base_model_t, self).__init__()
        parser.add_args(arguments())
        self.args = parser.get_args()
        self.is_deep_model = False
        self.nc = self.args.nc

        if self.args.cls_type == "multi_class":
            parser.add_args(arguments_multi_class())
            self.args = parser.get_args()
            parser.cls_nums = self.args.cls_nums

        self.train_with_test = self.args.test_interval
        # self.starter,self.ender = torch.cuda.Event(enable_timing = True), torch.cuda.Event(enable_timing = True) 
        self.network = self.args.cls_network
        self.class_type = self.args.cls_type
        self.dataset = self.args.cls_dataset
        self.metric = self.args.metric
        self.time_avg = self.args.time_avg
        self.indicator_for_best = None
        self.vis = Visualizer(parser)


        # self.train_type_num = []   #保存训练集中每类图片的数量
        # self.test_type_num = []    #保存测试集中每类图片的数量

    def save_weights(self):
        if self.args.control_save_end:
            weight_dir = os.path.join(self.args.outf, os.path.basename(os.getcwd()), self.args.cls_dataset, self.args.cls_type, self.args.cls_network, 'weight')
            if not os.path.exists(weight_dir):
                os.makedirs(weight_dir)

            for i in range(len(self.net_dict['name'])):
                with open('%s/{}.pth'.format(self.net_dict['name'][i]) % (weight_dir),"wb") as file:
                    pickle.dump(self.net_dict['network'][i],file)

    def load_weights(self,net_dict):
        for i in range(len(net_dict['name'])):
            weight_dir = os.path.join(self.args.outf, os.path.basename(os.getcwd()), self.args.cls_dataset, self.args.cls_type, self.args.cls_network, 'weight')
            weight_dir = os.path.join(weight_dir,self.net_dict['name'][i]+'.pth')
            with open(weight_dir,"rb")as file:
                self.net_dict['network'][i]=pickle.load(file)
        return self.net_dict['network']

    def train(self, train_dataloader, test_dataloader):

        pass

    def test(self, train_dataloader, test_dataloader):

        pass

    def inference(self, dataloader):

        raise NotImplementedError

    def feature_extraction(self,img):

        pass

    def get_feature(self,data_loader):
        feature = []
        for i in tqdm(range(0,len(data_loader[0])), total=len(data_loader[0]), leave=False):
            feature.append(self.feature_extraction(data_loader[0][i]))
        return feature

    def save_loggin_print(self, current_epoch, metric_cluster, best):
        self.final_epoch_trigger = True
        if self.args.control_print or self.args.control_save:
            self.vis.loggin(metric_cluster, current_epoch, best, self.indicator_for_best)
        if self.args.control_save:
            self.vis.save()
        if self.args.control_print:
            self.vis.output()
        if self.final_epoch_trigger and self.args.control_save:
            self.vis.plot_menu(best)

    def create_empty_loader(self):
        self.imsize2 = self.args.cls_imageSize if self.args.cls_imageSize2 ==-1 else self.args.cls_imageSize2
        sample_tensor = np.ones((self.time_avg, self.nc, self.args.cls_imageSize, self.imsize2), dtype=np.float32)
        label_tensor = np.ones((self.time_avg), dtype=np.float32)
        empty_dataloader = (sample_tensor,label_tensor)
        return empty_dataloader

    def compute_time_fps(self, empty_dataloader):
        time_start = time.time()
        pred, real, score = self.inference(empty_dataloader)
        time_end = time.time()
        self.inference_time = time_end - time_start
        self.gpu_time = self.inference_time/self.time_avg
        self.fps = 1000/ self.gpu_time

    def get_inference_time(self):
        self.train_with_test = 0
        self.train(self.create_empty_loader(), self.create_empty_loader())
        self.compute_time_fps(self.create_empty_loader())
        self.vis.plot_inference_time(self.gpu_time)
        self.train_with_test = self.args.test_interval
        return self.gpu_time

    def get_fps(self):
        self.train_with_test = 0
        self.train(self.create_empty_loader(), self.create_empty_loader())
        self.compute_time_fps(self.create_empty_loader())
        self.vis.plot_fps(self.fps)
        self.train_with_test = self.args.test_interval
        return self.fps

    # def get_pic_type_num(self, loader, loadertype):
    #     flag = loader[1][0]
    #     num = 0
    #     for i in range(0, len(loader[1])):
    #         if(flag==loader[1][i]):
    #             num += 1
    #         else:
    #             if(loadertype==0):
    #                 self.train_type_num.append(num)
    #                 num = 0
    #                 flag = loader[1][i]
    #             else:
    #                 self.test_type_num.append(num)
    #                 num = 0
    #                 flag = loader[1][i]
    #     self.train_type_num.append(num)
    #     self.test_type_num.append(num)
    #     return self.train_type_num,self.test_type_num

