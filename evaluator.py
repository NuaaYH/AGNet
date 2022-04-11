import os
import time

import numpy as np
import torch
from torchvision import transforms


cuda=True
def _eval_pr(y_pred, y, num):
    if cuda:
        prec, recall = torch.zeros(num).cuda(), torch.zeros(num).cuda()
        thlist = torch.linspace(0, 1 - 1e-10, num).cuda()
    else:
        prec, recall = torch.zeros(num), torch.zeros(num)
        thlist = torch.linspace(0, 1 - 1e-10, num)
    for i in range(num):
        y_temp = (y_pred >= thlist[i]).float()
        tp = (y_temp * y).sum()
        prec[i], recall[i] = tp / (y_temp.sum() + 1e-20), tp / (y.sum() + 1e-20)

    return prec, recall

def Eval_Fmeasure(pred,gt):
    beta2 = 0.3
    prec, recall = _eval_pr(pred, gt, 255)
    f_score = (1 + beta2) * prec * recall / (beta2 * prec + recall+1e-20)
    f_score[f_score != f_score] = 0 # for Nan
    return f_score

def Eval_mae(pred,gt):
    pred=pred.cuda()
    gt=gt.cuda()
    with torch.no_grad():
        mae = torch.abs(pred - gt).mean()
        if mae == mae:  # for Nan
            return mae.item()