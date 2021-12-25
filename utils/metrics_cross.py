import numpy as np
from sklearn.metrics import roc_curve, auc, average_precision_score
from scipy.interpolate import interp1d
from scipy.optimize import brentq

def calculate(threshold, dist, actual_issame):
    predict_issame = np.less(1-dist, 1-threshold)
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))
    return tp,fp,tn,fn

def pbca(msk_pred, msk_gt, thre):
    gt_bool = (msk_gt>0.5)
    tp, fp, tn, fn = calculate(thre, msk_pred, gt_bool)
    PBCA = (tp+tn)/((tp+fp+tn+fn)*1.0)
    return PBCA

def I_U(pred, gt):
    intersection = (pred * gt).sum()
    union = pred.sum() + gt.sum() - intersection
    return intersection, union

def iinc(msk_pred, msk_gt, thre):
    pixel_num = np.prod(msk_pred.shape)
    gt_bool = (msk_gt > thre)
    pred_bool = np.less(1-msk_pred, 1-thre)
    Mgt_mean = np.mean(gt_bool)
    Mgt_l1 = np.sum(gt_bool) * 1.0
    Mpred_mean = np.mean(pred_bool)
    Mpred_l1 = np.sum(pred_bool) * 1.0
    if Mgt_mean == 0 and Mpred_mean == 0:
        IINC = 0
    elif Mgt_mean == 0 and Mpred_mean != 0:
        _, U = I_U(pred_bool, gt_bool)
        U_norm = U/pixel_num
        IINC = 1/(3-U_norm)
    elif Mgt_mean != 0 and Mpred_mean == 0:
        _, U = I_U(pred_bool, gt_bool)
        U_norm = U / pixel_num
        IINC = 1 / (3 - U_norm)
    else:
        I, U = I_U(pred_bool, gt_bool)
        U_norm = U / pixel_num
        IINC = (2-I/Mpred_l1-I/Mgt_l1)*(1/(3-U_norm))
    return IINC

def get_loc_metrics(pred, gt, thre):
    PBCA = pbca(pred, gt, thre)
    IINC = iinc(pred, gt, thre)
    return PBCA, IINC

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

def calculate(threshold, dist, actual_issame):
    predict_issame = np.less(1-dist, 1-threshold)
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))
    return tp,fp,tn,fn

def FPR_FNR(threshold, dist, actual_issame):
    tp, fp, tn, fn = calculate(threshold, dist, actual_issame)
    FPR = fp / (tn*1.0 + fp*1.0)
    FNR = fn / (fn * 1.0 + tp * 1.0)
    return FPR, FNR

def eer_auc(y, y_score):
    fpr, tpr, thresholds = roc_curve(y, y_score, pos_label=1)
    eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    AUC = auc(fpr, tpr)
    return eer, AUC

def compute_mAP(y_true, y_pred):
    return average_precision_score(y_true, y_pred)

def get_metrics(pred, gt, thre):
    mAP = compute_mAP(gt, pred)
    gt_bool = (gt>0.5)
    _,_,ACC = calculate_accuracy(thre, pred, gt_bool)
    FPR, FNR = FPR_FNR(thre, pred, gt_bool)
    EER, AUC = eer_auc(gt, pred)
    return AUC, ACC, FPR, FNR, EER, mAP