""" Evaluate ROC

Returns:
    auc, eer: Area under the curve, Equal Error Rate
"""

# pylint: disable=C0103,C0301

##
# LIBRARIES
from __future__ import print_function

import os
import torch
import torch.nn.functional as F
import torch.nn as nn
from tqdm import trange
import numpy as np
from sklearn.metrics import roc_curve, auc, average_precision_score, f1_score
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt


def evaluate(labels, scores, metric='roc'):
    if metric == 'roc':
        return roc(labels, scores)
    elif metric == 'auprc':
        return auprc(labels, scores)
    elif metric == 'online_search':
        return find_best_cri(labels, scores)
    elif metric == 'combine':
        auc = roc(labels, scores)
        best_cri, best_rec, best_threshold = find_best_cri(labels, scores)
        return auc, best_cri, best_rec, best_threshold

    else:
        raise NotImplementedError("Check the evaluation metric.")

##
def roc(labels, scores, saveto=None):
    """Compute ROC curve and ROC area for each class"""
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    # labels = labels.cpu()
    # scores = scores.cpu()

    # True/False Positive Rates.
    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)

    # Equal Error Rate
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)

    if saveto:
        plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='darkorange', lw=lw, label='(AUC = %0.2f, EER = %0.2f)' % (roc_auc, eer))
        plt.plot([eer], [1-eer], marker='o', markersize=5, color="navy")
        plt.plot([0, 1], [1, 0], color='navy', lw=1, linestyle=':')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(saveto, "ROC.pdf"))
        plt.close()

    return roc_auc

def auprc(labels, scores):
    ap = average_precision_score(labels, scores)
    return ap


def find_best_cri(labels, scores):
    best_cri = -1
    best_rec = []
    best_threshold = -1
    for threshold in np.arange(0,1.001,0.001):
    # for threshold in np.arange(0.5, 0.51, 0.01):
        scores_tmp = (scores >= threshold).astype(np.float32)
        # f1 = 2 * acc * rec / (acc + rec)
        rec_abno = np.sum(labels*scores_tmp) / np.sum(labels)
        rec_norm = np.sum((1-labels)*(1-scores_tmp)) / np.sum(1-labels)
        cri = (rec_norm+rec_abno) / 2
        # print(rec_abno, rec_norm)
        if cri>best_cri:
            best_cri = cri
            best_rec = [rec_norm,rec_abno]
            best_threshold = threshold
        # if rec_abno>best_cri:
        #     best_cri = rec_abno
        #     best_rec = [rec_abno, rec_norm]
        #     best_threshold = threshold

    return best_cri, best_rec, best_threshold


##
def l1_loss(input, target):
    """ L1 Loss without reduce flag.

    Args:
        input (FloatTensor): Input tensor
        target (FloatTensor): Output tensor

    Returns:
        [FloatTensor]: L1 distance between input and output
    """

    return torch.mean(torch.abs(input - target))

##
def l2_loss(input, target, size_average=True):
    """ L2 Loss without reduce flag.

    Args:
        input (FloatTensor): Input tensor
        target (FloatTensor): Output tensor

    Returns:
        [FloatTensor]: L2 distance between input and output
    """
    if size_average:
        return torch.mean(torch.pow((input-target), 2))
    else:
        return torch.pow((input-target), 2)





class Meter_AnoGAN:
    def __init__(self):
        self.l_enc = []
        self.gt_labels = []

    # def update(self,latent_i,latent_o,label):
    #     l_enc = l2_loss(latent_i,latent_o)
    #     self.l_enc.append(l_enc.item())
    #     self.gt_labels.append(label)

    # TODO: for multi_layer
    def update(self,loss,label):
        self.l_enc.append(loss.item())
        self.gt_labels.append(label)


    def get_metrics(self):
        self.gt_labels = torch.Tensor(self.gt_labels)
        self.l_enc = torch.Tensor(self.l_enc)
        # print(torch.max(self.l_enc),torch.min(self.l_enc))
        abnormal_score = (self.l_enc - torch.min(self.l_enc)) / (
            torch.max(self.l_enc) - torch.min(self.l_enc))


        self.gt_labels = self.gt_labels.numpy()
        abnormal_score = abnormal_score.numpy()

        criterion = evaluate(self.gt_labels, abnormal_score, metric='combine')  # mean_rec
        best_threshold = criterion[-1]
        pred_labels = (abnormal_score>best_threshold).astype(np.float32)

        assert pred_labels.shape == self.gt_labels.shape
        res = []
        for (pred,gt) in zip(pred_labels,self.gt_labels):
            if   gt == 0 and pred == 0: res.append('TP')
            elif gt == 0 and pred == 1: res.append('FP')
            elif gt == 1 and pred == 0: res.append('FN')
            elif gt == 1 and pred == 1: res.append('TN')

        return criterion, res

