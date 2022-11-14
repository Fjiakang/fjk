#引用文章：基于微多普勒包络密度特征的手语信号感知与识别
import numpy as np
from tqdm import tqdm
from sklearn.neighbors import KNeighborsClassifier
import cv2
from libs.metric import metric
from libs.base_model_t import base_model_t

def arguments(): #分类器参数
    args = {"--imsize":100,
            "--cls_env_threshold" : [0.1,0.15, 0.85,0.9, 0.5,0.6, 0.18, 0.22, 0.75,0.82],
            "--cls_T_env_threshold" : [0.05,0.3, 0.7,0.98, 0.5,0.6, 0.3, 0.4, 0.6,0.7],
            "--cls_knn_n_neighbors" : 1,
            "--cls_knn_weights" : 'uniform',
            "--cls_knn_algorithm" : 'auto',
            "--cls_knn_metric" : 'manhattan',
            "--cls_knn_metric_params" : None,
            "--cls_knn_n_jobs" : None
    }
    return args

class Den_Band(base_model_t):
    def __init__(self, parser):
        super().__init__(parser)
        parser.add_args(arguments())
        self.args = parser.get_args()
        parser.change_args("cls_imageSize", self.args.imsize)
        self.args = parser.get_args()
        self.model = KNeighborsClassifier(n_neighbors=self.args.cls_knn_n_neighbors, 
                                            weights=self.args.cls_knn_weights, 
                                            algorithm=self.args.cls_knn_algorithm, 
                                            metric=self.args.cls_knn_metric, 
                                            metric_params=self.args.cls_knn_metric_params, 
                                            n_jobs=self.args.cls_knn_n_jobs)       
        self.net_dict = {"name": ['envelope_den'],
                        "network":[self.model]}
        # self.gpu_time = self.get_inference_time()
        # self.fps = self.get_fps()

    def train(self, train_loader, test_loader):
        best = float(0)
        train_feature = self.get_feature(train_loader)
        self.model.fit(train_feature,train_loader[1])
        if (self.train_with_test):
            pred, real, score = self.inference(test_loader)
            metric_cluster = metric(pred, real, score, self.metric, self.args)
            best, self.best_trigger, self.indicator_for_best = metric_cluster.best_value_indicator(best, self.indicator_for_best)
            self.save_loggin_print(1, metric_cluster, best)
            self.save_weights()

    def inference(self, test_loader):
        test_feature = self.get_feature(test_loader)
        pred = self.model.predict(test_feature)
        score = self.model.predict_proba(test_feature)
        return pred, test_loader[1], score

    def test(self, train_loader, test_loader):
        self.load_weights(self.net_dict)
        pred, real, score = self.inference(test_loader)
        metric_cluster = metric(pred, real, score, self.metric, self.args)
        self.best, _, self.indicator_for_best = metric_cluster.best_value_indicator(float(0), self.indicator_for_best)
        assert self.args.phase == "test", ''' Call test function but phase is not testing. '''
        self.save_loggin_print("test", metric_cluster, self.best)

    def feature_extraction(self, img):
        #envelope methods
        den = self.per_envelope(img, self.args.cls_env_threshold)
        denT = self.per_envelope(np.transpose(img),self.args.cls_T_env_threshold)
        features = np.concatenate([den,denT], axis=0).flatten()
        
        return features

    def per_envelope(self, img, per_env_threshold):
        img = img.transpose(1, 2, 0) if (np.shape(img)[0]==1) or (np.shape(img)[0]==3) else img#将CHW转换为HWC
        img = img if np.shape(img)[2] == 1 else cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) #灰度处理
        Pv = np.cumsum(img,axis=0)
        percent = Pv/Pv[-1]
        band_num = len(per_env_threshold) // 2
        assert len(per_env_threshold) % 2 == 0, "Your threshold is not symmetry."

        den = np.zeros((band_num, np.shape(img)[1]))
        for i in range(0,np.shape(img)[1]):
            for j in range(band_num):
                B = np.where((per_env_threshold[j*2]<percent[:,i])&(percent[:,i]<per_env_threshold[j*2+1]))
                den[j,i] = len(B[0])
        return den
