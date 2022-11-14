from sklearn.metrics import roc_curve, auc, accuracy_score, roc_auc_score, confusion_matrix, f1_score,precision_recall_curve,average_precision_score,precision_score,recall_score
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from libs.data import data
import csv
import os
import torch
import numpy as np
    
def class_num(real):
    assert len(set(real.tolist())) > 1
    if len(set(real.tolist())) == 2:
        return True
    else:
        return False

class metric(object):
    def __init__(self, pred, real, score, metric_list,args):
        super(metric, self).__init__()

        pred = torch.from_numpy(pred) if isinstance(pred, np.ndarray) else pred
        real = torch.from_numpy(real) if isinstance(real, np.ndarray) else real
        score = torch.from_numpy(score) if isinstance(score, np.ndarray) else score

        self.metric_list = metric_list
        cls_type = args.cls_type
        metric_average = args.metric_average
        pos_label = args.metric_pos_label

        open_set_score = score
        score = score if torch.squeeze(torch.tensor(score.size())).dim() == 0  else score.transpose(0,1)[pos_label]

        self.fpr = roc_curve(real, score)[0] if class_num(real) else None
        self.tpr = roc_curve(real, score)[1] if class_num(real) else None
        self.frr = 1.-self.tpr if class_num(real) else None
        self.far = self.fpr if class_num(real) else None
        self.DET = np.vstack((self.far, self.frr)) if class_num(real) else None
        self.auc = auc(self.fpr, self.tpr) if class_num(real) else None
        self.eer = brentq(lambda x: 1. - x - interp1d(self.far, 1-self.frr, axis=0, fill_value="extrapolate")(x), 0.0001, 1.) if class_num(real) else None
        self.ROC = np.vstack((self.fpr, self.tpr)) if class_num(real) else None
        
        self.precision_curve = precision_recall_curve(real, score, pos_label = pos_label)[0] if class_num(real) else None
        self.recall_curve = precision_recall_curve(real, score, pos_label = pos_label)[1] if class_num(real) else None
        self.PR = np.vstack((self.recall_curve, self.precision_curve)) if class_num(real) else None
        self.AP = average_precision_score(real, score, pos_label = pos_label) if class_num(real) else None
        
        self.precision = precision_score(real, pred, pos_label = pos_label, average = metric_average) 
        self.recall = recall_score(real, pred, pos_label = pos_label, average = metric_average)
        self.F1_score = f1_score(real, pred, pos_label = pos_label, average = metric_average)

        self.acc = accuracy_score(real, pred)
        self.confusion_matrix = confusion_matrix(real, pred)

        if cls_type == 'open_set':
            self.auc = self.open_set_AUC(open_set_score, real)
            self.precision = self.open_set_Precision_Recall_F1()[0]
            self.recall = self.open_set_Precision_Recall_F1()[1]
            self.F1_score = self.open_set_Precision_Recall_F1()[2]

        self.metric_dict = {'AUC': self.auc,
                            'ACC': self.acc,
                            'EER': self.eer,
                            'confusion_matrix': self.confusion_matrix,
                            'ROC': self.ROC,
                            'DET': self.DET,
                            'F1_score': self.F1_score,
                            'PR':self.PR,
                            'AP': self.AP,
                            'precision': self.precision,
                            'recall': self.recall}

        self.metric_properties = {
                            'confusion_matrix': 'matrix',
                            'ROC':'curve',
                            'DET':'curve',
                            'PR':'curve'}

        if metric_average is None or cls_type != 'open_set':
            self.metric_properties['F1_score'] = 'matrix'
            self.metric_properties['precision'] = 'matrix'
            self.metric_properties['recall'] = 'matrix'

        self.values = self.get_metric(self.metric_list)

        self.indicator_for_best = None
        self.best = float(0)
        self.best_trigger = False

        self.higher_is_best = ['ACC','AUC','F1_score','AP','precision','recall']
        self.lower_is_best = ['EER']

    def open_set_Precision_Recall_F1(self):
        list_end = self.confusion_matrix.shape[0]-1
        TP,FP,FN = 0,0,0
        for i in range(list_end):
            TP +=self.confusion_matrix[i][i]
        FP = self.confusion_matrix[:,0:list_end].sum()-TP
        FN = self.confusion_matrix[0:list_end,-1].sum()
        precision = TP/(TP+FP)
        recall = TP/(TP+FN)
        return precision, recall, 2*precision*recall/(precision+recall)

    def open_set_AUC(self,score,real):
        known_classes_nums = score.shape[1]-1
        score = score[:, -1].numpy()
        score_known,score_unknown=[],[]
        score_known.extend(score[:len(np.where(real != known_classes_nums)[0])])
        score_unknown.extend(score[len(np.where(real != known_classes_nums)[0]):])
        y_true = np.array([0]*len(score_known)+[1]*len(score_unknown))
        y_score = np.concatenate([score_known,score_unknown])
        auc = roc_auc_score(y_true,y_score)
        return auc
        
    def metric_for_sorting(self, best, indicator_for_best):
        if indicator_for_best is None:
            for self.indicator_for_best in range(0,len(self.values)):
                if not self.values[self.indicator_for_best] is None:
                    self.best = float(0) if self.metric_list[self.indicator_for_best] in self.higher_is_best else float(1)                    
                    break
        else:
            self.indicator_for_best = indicator_for_best
            self.best = best        


    def best_value_indicator(self, best, indicator_for_best):
        self.metric_for_sorting(best, indicator_for_best)
        comparison_operator = '<' if self.metric_list[self.indicator_for_best] in self.higher_is_best else '>'
        exec("self.best_trigger = True if self.best {} self.values[self.indicator_for_best] else False".format(comparison_operator))
        self.best = self.values[self.indicator_for_best] if self.best_trigger else self.best
        return self.best, self.best_trigger, self.indicator_for_best

    def get_metric(self, metric):
        metric_res = []
        for i in range(len(metric)):

            if self.metric_dict[metric[i]] is None:
                raise Exception(self.metric_dict[metric[i]]+" can't be compute.")

            if isinstance(self.metric_dict[metric[i]],np.ndarray): ####指标为数组时候的占位符
                if not self.metric_dict[metric[i]].size == 1:
                    metric_res.append(None)
                    continue

            metric_res.append(self.metric_dict[metric[i]])
        return metric_res

