# 引用文章：Knowledge Exploitation for Human Micro-Doppler Classification
# C. Karabacak, S. Z. Gurbuz, A. C. Gurbuz, M. B. Guldogan, G. Hendeby, and F. Gustafsson, “Knowledge exploitation for human micro-Doppler classification,” IEEE Geosci. Remote Sens. Lett., vol. 12, no. 10, pp. 2125–2129, Oct. 2015.
import numpy as np
from tqdm import tqdm
from sklearn.neighbors import KNeighborsClassifier
import cv2
from libs.metric import metric
from libs.base_model_t import base_model_t


def arguments():
    args = {"--cls_env_threshold":[0.22,0.51,0.83]}
    return args

class EFM(base_model_t):
    def __init__(self, parser):
        super().__init__(parser)
        parser.add_args(arguments())
        self.args = parser.get_args()
        self.imsize = 100
        parser.change_args("cls_imageSize", self.imsize)
        self.args = parser.get_args()
        self.model = KNeighborsClassifier(n_neighbors=10, weights='uniform', algorithm='auto', metric='euclidean', metric_params=None, n_jobs=None)       
        self.net_dict = {"name": ['EFM_model'],
                        "network":[self.model]}

        self.gpu_time = self.get_inference_time()
        self.fps = self.get_fps()

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
        features = np.zeros((len(self.args.cls_env_threshold)))
        envelopes=self.per_envelope(img, self.args.cls_env_threshold)
        [Upper_env,Lower_env]=self.ULenv(envelopes)
        features[0],features[1],features[2]=np.mean(Upper_env),np.mean(envelopes[1,:]),np.mean(Lower_env)
        return features

    def per_envelope(self, img,per_env_threshold):
        img = img.transpose(1, 2, 0) if (np.shape(img)[0]==1) or (np.shape(img)[0]==3) else img#将CHW转换为HWC
        img = img if np.shape(img)[2] == 1 else cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) #灰度处理
        Pv = np.cumsum(img,axis=0)
        percent = Pv/Pv[-1]

        envelopes=np.zeros((len(self.args.cls_env_threshold),np.shape(img)[1]))
        for i in range(0,np.shape(img)[1]):
            for j in range(len(self.args.cls_env_threshold)):
                B = np.where(percent[:,i]<per_env_threshold[j])
                envelopes[j,i] = B[0][-1] if len(B[0]) !=0 else 0

        return envelopes

    def ULenv(self,envelopes):

        return [envelopes[0,:],envelopes[np.shape(envelopes)[0]-1,:]]

    # def get_learning_type(self):

    #     return self.args.learning_type

