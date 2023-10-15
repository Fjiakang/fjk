import os
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader


class BaseModel(nn.Module):
    def __init__(self, parser):
        super().__init__()
        self.args = parser.get_args()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.indicator_for_best = None

        self.starter, self.ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.backends.cudnn.benchmark = True if not self.device == "cpu" else False
        # torch.set_default_tensor_type("torch.cuda.FloatTensor") if not self.device == "cpu" else torch.set_default_tensor_type("torch.FloatTensor")

        torch.manual_seed(self.args.seed)
        self.lr_step = [int(i) for i in self.args.lr_step.split(",")]

    def train(self, train_loader, testset):
        raise NotImplementedError

    def test(self, testset):
        raise NotImplementedError

    def inference(self):
        raise NotImplementedError

    def save_weights(self, epoch, folder=None):
        weight_dir = self.save_folder if folder == None else folder
        if not os.path.exists(weight_dir):
            os.makedirs(weight_dir)
        for i in range(len(self.net_dict["name"])):
            torch.save({"epoch": epoch, "net_params": self.net_dict["network"][i].state_dict()}, "%s/{}_current.pth".format(self.net_dict["name"][i]) % (weight_dir))

        if self.wegiter_trigger:
            for i in range(len(self.net_dict["name"])):
                self.wegsave_dir = "{}/epo_{}.pth".format(weight_dir, epoch)
                cmd_wegsave = "cp {}/{}_current.pth {}/epo_{}.pth".format(weight_dir, self.net_dict["name"][i], weight_dir, epoch)
                os.system(cmd_wegsave)
        if self.best_trigger:
            for i in range(len(self.net_dict["name"])):
                cmd_best = "cp {}/{}_current.pth {}/{}_best.pth".format(weight_dir, self.net_dict["name"][i], weight_dir, self.net_dict["name"][i])
                os.system(cmd_best)
        if self.final_trigger:
            for i in range(len(self.net_dict["name"])):
                cmd_final = "mv {}/{}_current.pth {}/{}_final.pth".format(weight_dir, self.net_dict["name"][i], weight_dir, self.net_dict["name"][i])
                os.system(cmd_final)

    def load_weights(self, load_modeldir):
        for i in range(len(self.net_dict["name"])):
            self.pretrain_dict = torch.load(load_modeldir)
            self.net_dict["network"][i].load_state_dict(self.pretrain_dict["net_params"])

    def save_loggin_print(self, epo, avg_loss_stats, iter_id, num_iters):
        if epo == 1 and iter_id == 0:
            self.vis.plot_txt("config", vars(self.args))
        self.vis.loggin(epo, avg_loss_stats, iter_id, num_iters)
        self.vis.save()
        self.vis.output()

    def create_empty_loader(self):
        self.imsize2 = self.args.inp_imgsize
        sample_tensor = torch.zeros(self.args.batch_size * self.time_avg, 3, self.args.inp_imgsize, self.imsize2)
        label_tensor = torch.zeros(self.args.batch_size * self.time_avg)
        empty_dataset = TensorDataset(sample_tensor, label_tensor)
        empty_dataloader = DataLoader(empty_dataset, shuffle=True, batch_size=self.batchsize)
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
            self.gpu_time = self.inference_time / self.time_avg
            self.fps = 1000 / self.gpu_time

    def get_inference_time(self):
        self.compute_time_fps(self.create_empty_loader())
        self.vis.plot_inference_time(self.gpu_time)
        return self.gpu_time

    def get_fps(self):
        self.compute_time_fps(self.create_empty_loader())
        self.vis.plot_fps(self.fps)
        return self.fps

    # self.save_result_img(e + 1, self.best_trigger, metric_cluster, embedding, real, self.lossdic, model_layer_list=[{"model": self.model, "layerlist": ["layer1"]}])

    def save_result_img(self, current_epoch, best_trigger, real, lossdict, model_layer_list=None):
        self.final_epoch_trigger = True if current_epoch == self.epochs else False
        if self.args.control_save_img_type is not None:

            if self.final_epoch_trigger:  ####最后
                if "lossepoch" in self.args.control_save_img_type:
                    self.vis.save_lossepochimg(lossdict)
            if self.final_epoch_trigger or current_epoch == "test":  #####最后或test
                if self.final_epoch_trigger:
                    self.load_weights()
                if "attentionmap" in self.args.control_save_img_type:
                    self.vis.save_attentionmap(model_layer_list, image_size=self.args.inp_imgsize)
                if "featuremap" in self.args.control_save_img_type:
                    self.vis.save_featuremap(model_layer_list, image_size=self.args.inp_imgsize)
        if self.args.control_monitor and current_epoch != "test":  ######实时,不要更改顺序
            self.vis.monitor(self.viz, current_epoch, lossdict)
