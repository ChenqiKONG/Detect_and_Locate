import numpy as np
from sklearn.metrics import roc_curve,  auc
from scipy.interpolate import interp1d
from scipy.optimize import brentq

def calculate_accuracy(threshold, dist, actual_issame):
    predict_issame = np.less(1-dist, 1-threshold)
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))
    tpr = 0 if (tp +fn==0) else float(tp) / float(tp +fn)
    fpr = 0 if (fp +tn==0) else float(fp) / float(fp +tn)
    acc = float(tp +tn ) /dist.shape[0]
    return tpr, fpr, acc

def eer_auc(y, y_score):
    fpr, tpr, _ = roc_curve(y, y_score, pos_label=1)
    eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    AUC = auc(fpr, tpr)
    return eer, AUC

def get_metrics(pred, gt, thre):
    gt_bool = (gt>0.5)
    _,_,ACC = calculate_accuracy(thre, pred, gt_bool)
    _, AUC = eer_auc(gt, pred)
    return AUC, ACC