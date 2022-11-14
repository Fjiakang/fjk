# 引用文章：Sign Language Recognition with CW Radar and Machine Learning
# Lu Y, Lang Y. Sign Language Recognition with CW Radar and Machine Learning[C]//2020 21st International Radar Symposium (IRS). IEEE, 2020: 31-34.
import numpy as np
from tqdm import tqdm
from sklearn.neighbors import KNeighborsClassifier
import cv2
from libs.metric import metric
from libs.base_model_t import base_model_t
from skimage.feature import hog

def arguments():
    args = {"--imsize":100}
    return args

class SLR_KNN(base_model_t):
    def __init__(self, parser):
        super().__init__(parser)
        parser.add_args(arguments())
        self.args = parser.get_args()
        parser.change_args("cls_imageSize", self.args.imsize)
        self.args = parser.get_args()
        self.model = KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='auto', metric='euclidean', metric_params=None, n_jobs=None)       
        self.net_dict = {"name": ['SLR_KNN_model'],
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
        img = img.transpose(1, 2, 0) if (np.shape(img)[0]==1) or (np.shape(img)[0]==3) else img#将CHW转换为HWC
        features = hog(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(8, 8))
        return features

