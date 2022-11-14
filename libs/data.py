import os
import torch
import numpy as np
import torchvision.datasets as datasets
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torch.utils.data import DataLoader as dataloader 
import copy


def arguments_randomcrop():
    args = {'--random_crop_w':1,
    '--random_crop_h':1}
    return args

def arguments_centercrop():
    args = {'--center_crop':16}
    return args

def arguments_colorjitter():
    args = {'--bright':1,
    '--contrast':1,
    '--hue':0}
    return args

def arguments_graychannels():
    args = {'--gray_channels':1}
    return args

class data():
    def __init__(self,parser,is_deep_model):

        if parser.get_args().gray_image:
            parser.add_args(arguments_graychannels())
        self.args = parser.get_args() 
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        torch.backends.cudnn.benchmark = True if not self.device == "cpu" else False
        self.cls_dataset = self.args.cls_dataroot+'/' + self.args.cls_dataset + '/'
        self.is_deep_model = is_deep_model
        self.imsize2 = self.args.cls_imageSize if self.args.cls_imageSize2 ==-1 else self.args.cls_imageSize2

        exec('''{}'''.format(self.transform_all(parser,'train')))
        exec('''{}'''.format(self.transform_all(parser,'test')))

        if self.args.cls_dataset == "mnist":
            self.classes = {
            '0': 0, '1': 1, '2': 2, '3': 3, '4': 4,
            '5': 5, '6': 6, '7': 7, '8': 8, '9': 9
        }
            self.train_set = datasets.MNIST(os.path.expanduser(self.cls_dataset), train=True, download=True,
                                            transform=self.transform_train)
            self.test_set = datasets.MNIST(os.path.expanduser(self.cls_dataset), train=False, download=True,
                                            transform=self.transform_test)
        elif self.args.cls_dataset == "cifar10":
            self.classes = {
            'plane': 0, 'car': 1, 'bird': 2, 'cat': 3, 'deer': 4,
            'dog': 5, 'frog': 6, 'horse': 7, 'ship': 8, 'truck': 9
        }
            self.train_set = datasets.CIFAR10(os.path.expanduser(self.cls_dataset), train=True, download=True,
                                                transform=self.transform_train)
            self.test_set = datasets.CIFAR10(os.path.expanduser(self.cls_dataset), train=False, download=True,
                                                transform=self.transform_test)
        else:
            self.classes = {}
            self.test_folder_names = os.listdir(self.cls_dataset + 'test/')
            self.train_folder_names = os.listdir(self.cls_dataset + 'train/')
            self.test_folder_names.sort()
            self.train_folder_names.sort()
            assert len(self.test_folder_names) == len(self.train_folder_names),  '''your train folder contain more or less categories than test. '''
            for i in range(len(self.test_folder_names)):
                self.classes[self.test_folder_names[i]] = i

            self.train_set = ImageFolder(os.path.join(self.cls_dataset, 'train'), 
                                        transform=self.transform_train)
            self.test_set = ImageFolder(os.path.join(self.cls_dataset, 'test'), 
                                        transform=self.transform_test)

    def transform_all(self, parser, mode_str):
        methods = {'random_crop':'arguments_randomcrop',
                    'center_crop':'arguments_centercrop',
                    'color':'arguments_colorjitter'}

        for tran in self.args.aug_methods:
            exec("parser.add_args({}())".format(methods[tran]))
            self.args = parser.get_args()

        transform_statement = '''self.transform_''' + mode_str + '''= transforms.Compose(['''

        if self.args.gray_image:
            transform_statement += '''transforms.Grayscale(num_output_channels={}),'''.format(self.args.gray_channels)

        if mode_str == 'train':
            for tran in self.args.aug_methods:
                if tran == 'random_crop':
                    transform_statement += '''transforms.RandomCrop(({},{})),'''.format(self.args.random_crop_w, self.args.random_crop_h)
                elif tran == 'center_crop':
                    transform_statement += '''transforms.CenterCrop(({},{})),'''.format(self.args.center_crop, self.args.center_crop)
                elif tran == 'color':
                    transform_statement += '''transforms.ColorJitter(brightness={},contrast={},hue={}),'''.format(self.args.bright, self.args.contrast,self.args.hue)

        transform_statement += '''transforms.Resize((self.args.cls_imageSize,self.imsize2)),transforms.ToTensor()])'''

        return transform_statement

    def get_data(self):
        self.change_label()
        self.choose_pos_label(self.test_set)
        if self.is_deep_model:
             self.train_set = dataloader(self.train_set, batch_size = self.args.cls_batchsize, shuffle = True, drop_last = True)
             self.test_set = dataloader(self.test_set, batch_size = 1, shuffle = False, drop_last = False)
        else:
            self.train_set, self.test_set = self.get_numpy()

        return self.train_set, self.test_set

    def change_label(self):
        if self.args.cls_type in ["one_class","open_set"]:
            data_type = 'data' if self.args.cls_dataset in ['mnist','cifar10'] else 'samples'
            label_type = 'targets'
            known_lbl = copy.deepcopy(self.args.known_lbl)

            for i in range(len(known_lbl)):
                known_lbl[i] = self.classes[known_lbl[i]]

            exec("""self.train_set.{img}, self.train_set.{lbl},\
                self.test_set.{img},self.test_set.{lbl}=self.{type}_label(self.train_set.{img},
                    self.train_set.{lbl},
                    self.test_set.{img},
                    self.test_set.{lbl},
                    known_lbl)""".format(img=data_type,lbl=label_type,type=self.args.cls_type))


    # def change_dset(self, known_lbl):

        
    #     # data_type = {"mnist":'data',
    #     # "cifar10":'data',
    #     # "folder":'samples'}
    #     # label_type = {"mnist":'targets',
    #     # "cifar10":'targets',
    #     # "folder":'targets'}
    #     for i in range(len(known_lbl)):
    #         known_lbl[i] = self.classes[known_lbl[i]]

    #     exec("""self.train_set.{img}, self.train_set.{lbl},\
    #         self.test_set.{img},self.test_set.{lbl}=self.{type}_label(self.train_set.{img},
    #             self.train_set.{lbl},
    #             self.test_set.{img},
    #             self.test_set.{lbl},
    #             known_lbl)""".format(img=data_type,lbl=label_type,type=self.args.cls_type))

    def one_class_label(self,trn_img,trn_lbl,tst_img,tst_lbl,known):
        unknown_trn_idx, known_trn_idx, unknown_tst_idx,known_tst_idx = [],[],[],[]
        if isinstance(trn_img, np.ndarray) :
            trn_img = torch.tensor(trn_img)
            trn_lbl = torch.tensor(trn_lbl)
            tst_img = torch.tensor(tst_img)
            tst_lbl = torch.tensor(tst_lbl)
        if isinstance(trn_img, list) :
            trn_img = np.array(trn_img)
            tst_img = np.array(tst_img)    
            trn_lbl = np.array(trn_lbl)
            tst_lbl = np.array(tst_lbl)
            for i in range(len(known)):
                unknown_trn_idx.extend(np.where(trn_lbl != known[i])[0].tolist())
                known_trn_idx.extend(np.where(trn_lbl == known[i])[0].tolist())
                unknown_tst_idx.extend(np.where(tst_lbl != known[i])[0].tolist())
                known_tst_idx .extend(np.where(tst_lbl == known[i])[0].tolist())
        else:
            for i in range(len(known)):
                unknown_trn_idx.extend(torch.from_numpy(np.where(trn_lbl.numpy() != known[i])[0]))
                known_trn_idx.extend(torch.from_numpy(np.where(trn_lbl.numpy() == known[i])[0]))
                unknown_tst_idx.extend(torch.from_numpy(np.where(tst_lbl.numpy() != known[i])[0]))
                known_tst_idx.extend(torch.from_numpy(np.where(tst_lbl.numpy() ==known[i])[0]))

        unknown_trn_img = trn_img[unknown_trn_idx]
        known_trn_img = trn_img[known_trn_idx]
        unknown_trn_lbl = trn_lbl[unknown_trn_idx]
        known_trn_lbl = trn_lbl[known_trn_idx]
        unknown_tst_img = tst_img[unknown_tst_idx]
        known_tst_img = tst_img[known_tst_idx]
        unknown_tst_lbl = tst_lbl[unknown_tst_idx]
        known_tst_lbl = tst_lbl[known_tst_idx]

        unknown_trn_lbl[:] = 0
        unknown_tst_lbl[:] = 0
        known_trn_lbl[:] = 1
        known_tst_lbl[:] = 1

        if self.args.cls_dataset not in ['mnist','cifar10']:
            new_trn_img = np.concatenate((np.expand_dims(known_trn_img[:,0], axis=1), np.expand_dims(known_trn_lbl, axis=1)), axis=1)
            new_trn_img = [tuple(new_trn_img[x]) for x in range(0,len(new_trn_img))]
            new_trn_lbl = np.copy(known_trn_lbl)
            new_tst_img = np.concatenate((np.concatenate((np.expand_dims(unknown_tst_img[:,0], axis=1), np.expand_dims(unknown_tst_lbl, axis=1)), axis=1), 
                np.concatenate((np.expand_dims(known_tst_img[:,0], axis=1), np.expand_dims(known_tst_lbl, axis=1)), axis=1)), axis=0)
            new_tst_img = [tuple(new_tst_img[x]) for x in range(0,len(new_tst_img))]
            new_tst_lbl = np.concatenate((unknown_tst_lbl, known_tst_lbl), axis=0)

        else:
            new_trn_img = known_trn_img.clone()
            new_trn_lbl = known_trn_lbl.clone()
            new_tst_img = torch.cat((known_tst_img, unknown_tst_img), dim=0)
            new_tst_lbl = torch.cat((known_tst_lbl, unknown_tst_lbl), dim=0)

        if self.args.cls_dataset in ['cifar10']:
            return new_trn_img.numpy(), new_trn_lbl.tolist(), new_tst_img.numpy(), new_tst_lbl.tolist()
        else:
            return new_trn_img, new_trn_lbl, new_tst_img, new_tst_lbl

    def open_set_label(self,trn_img,trn_lbl,tst_img,tst_lbl,lbls):
        unknown_lbl = []
        for v in self.classes.values():
            if v not in lbls:
                unknown_lbl.append(v)
        known_trn_idx ,known_tst_idx ,unknown_trn_idx,unknown_tst_idx= [],[],[],[]
        if isinstance(trn_img, np.ndarray) :
            trn_img = torch.tensor(trn_img)
            trn_lbl = torch.tensor(trn_lbl)
            tst_img = torch.tensor(tst_img)
            tst_lbl = torch.tensor(tst_lbl)
        trn_lbl_change = copy.deepcopy(trn_lbl)
        tst_lbl_change = copy.deepcopy(tst_lbl)
        if isinstance(trn_img, list) :
            trn_img = np.array(trn_img)
            tst_img = np.array(tst_img)    
            trn_lbl = np.array(trn_lbl)
            tst_lbl = np.array(tst_lbl)
            trn_lbl_change = copy.deepcopy(trn_lbl)
            tst_lbl_change = copy.deepcopy(tst_lbl)    
            for i in range(len(lbls)):
                    known_trn_idx.extend(np.where(trn_lbl == lbls[i])[0].tolist())
                    known_tst_idx.extend(np.where(tst_lbl == lbls[i])[0].tolist()) 
                    trn_lbl_change[np.where(trn_lbl == lbls[i])[0].tolist()] = i
                    tst_lbl_change[np.where(tst_lbl == lbls[i])[0].tolist()] = i

            for j in range(len(unknown_lbl)):                    
                    unknown_tst_idx.extend(np.where(tst_lbl == unknown_lbl[j])[0].tolist())

        else:
            for i in range(len(lbls)):  
                    known_trn_idx.extend(torch.from_numpy(np.where(trn_lbl.numpy() == lbls[i])[0]))
                    known_tst_idx.extend(torch.from_numpy(np.where(tst_lbl.numpy() == lbls[i])[0]))
                    trn_lbl_change[np.where(trn_lbl == lbls[i])[0].tolist()] = i
                    tst_lbl_change[np.where(tst_lbl == lbls[i])[0].tolist()] = i

            for j in range(len(unknown_lbl)):                    
                    unknown_tst_idx.extend(torch.from_numpy(np.where(tst_lbl.numpy() == unknown_lbl[j])[0]))

        known_trn_img = trn_img[known_trn_idx]
        known_trn_lbl = trn_lbl_change[known_trn_idx]

        known_tst_img = tst_img[known_tst_idx]
        known_tst_lbl = tst_lbl_change[known_tst_idx]
        unknown_tst_img = tst_img[unknown_tst_idx]
        unknown_tst_lbl = tst_lbl_change[unknown_tst_idx]

        unknown_tst_lbl[:] = self.args.known_classes_nums

        key = list(self.classes.keys())
        value = list(self.classes.values())

        for i in range(len(key)):
            if i < len(lbls):
                self.classes[key[value.index(lbls[i])]] = i 
            else: 
                self.classes[key[value.index(unknown_lbl[i - len(lbls)])]] =  self.args.known_classes_nums

        if self.args.cls_dataset not in ['mnist','cifar10']:
            new_trn_img = np.concatenate((np.expand_dims(known_trn_img[:,0], axis=1), np.expand_dims(known_trn_lbl, axis=1)), axis=1)
            new_trn_img = [tuple(new_trn_img[x]) for x in range(0,len(new_trn_img))]
            new_trn_lbl = np.copy(known_trn_lbl)
            new_tst_img = np.concatenate((np.concatenate((np.expand_dims(known_tst_img[:,0], axis=1), np.expand_dims(known_tst_lbl, axis=1)), axis=1), 
                np.concatenate((np.expand_dims(unknown_tst_img[:,0], axis=1), np.expand_dims(unknown_tst_lbl, axis=1)), axis=1)), axis=0)
            new_tst_img = [tuple(new_tst_img[x]) for x in range(0,len(new_tst_img))]
            new_tst_lbl = np.concatenate((known_tst_lbl, unknown_tst_lbl), axis=0)

        else:
            new_trn_img = known_trn_img.clone()
            new_trn_lbl = known_trn_lbl.clone()
            new_tst_img = torch.cat((known_tst_img, unknown_tst_img), dim=0)
            new_tst_lbl = torch.cat((known_tst_lbl, unknown_tst_lbl), dim=0)

        if self.args.cls_dataset in ['cifar10']:
            return new_trn_img.numpy(), new_trn_lbl.tolist(), new_tst_img.numpy(), new_tst_lbl.tolist()
        else:
            return new_trn_img, new_trn_lbl, new_tst_img, new_tst_lbl

    def get_numpy(self):
        data_type = 'data' if self.args.cls_dataset in ['mnist','cifar10'] else 'samples'
        label_type = 'targets'

        # data_type = {"mnist":'data', "cifar10":'data', "folder":'samples'}
        # label_type = {"mnist":'targets', "cifar10":'targets', "folder":'target'}

        if self.args.cls_dataset == "mnist":
            exec("""self.train_set_numpy = self.train_set.{}.numpy()""".format(data_type))
            exec("""self.train_label_numpy = self.train_set.{}.numpy()""".format(label_type))
            exec("""self.test_set_numpy = self.test_set.{}.numpy()""".format(data_type))
            exec("""self.test_label_numpy = self.test_set.{}.numpy()""".format(label_type))

        elif self.args.cls_dataset == "cifar10":
            exec("""self.train_set_numpy = self.train_set.{}""".format(data_type))
            exec("""self.train_label_numpy = self.train_set.{}""".format(label_type))
            exec("""self.test_set_numpy = self.test_set.{}""".format(data_type))
            exec("""self.test_label_numpy = self.test_set.{}""".format(label_type))
        
        else:
            self.train_set_numpy = next(iter(dataloader(self.train_set, batch_size=self.train_set.__len__(), shuffle=False)))[0].numpy()
            self.train_label_numpy = next(iter(dataloader(self.train_set, batch_size=self.train_set.__len__(), shuffle=False)))[1].numpy()
            self.test_set_numpy = next(iter(dataloader(self.test_set, batch_size=self.test_set.__len__(), shuffle=False)))[0].numpy()
            self.test_label_numpy = next(iter(dataloader(self.test_set, batch_size=self.test_set.__len__(), shuffle=False)))[1].numpy()


        self.train_set_numpy = np.expand_dims(self.train_set_numpy, axis = 1) if len(self.train_set_numpy.shape) == 3 else self.train_set_numpy
        self.test_set_numpy = np.expand_dims(self.test_set_numpy, axis = 1) if len(self.test_set_numpy.shape) == 3 else self.test_set_numpy

        assert((len(self.train_set_numpy.shape) == 4) and (len(self.test_set_numpy.shape) == 4))

        return (self.train_set_numpy, self.train_label_numpy), (self.test_set_numpy, self.test_label_numpy)

    def choose_pos_label(self, test_set):
        label = torch.tensor(test_set.targets) if self.args.cls_dataset not in ['mnist','cifar10'] else test_set.targets
        if self.args.cls_type == 'one_class':
            self.args.metric_pos_label = 1  if self.args.metric_pos_label == self.args.known_lbl else 0 
        elif (self.args.cls_type == 'multi_class') and (self.args.cls_nums != 2):
            self.args.metric_pos_label = None
        else:
            self.args.metric_pos_label = self.classes[self.args.metric_pos_label]