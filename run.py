from dataset import *
from res2net import *
from modules import *
import os
import torch
import numpy as np
import torch.utils.data as data
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import time
from evaluator import Eval_Fmeasure,Eval_mae

os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'
gpu_devices = list(np.arange(torch.cuda.device_count()))
multi_gpu = len(gpu_devices) > 1

output_folder = r'./Outputs/pred/AGNet/EORSSD/Test'
ckpt_folder = r'./Checkpoints'
dataset_root = r'../Dataset/EORSSD'

batch_size = 8
 

class FLoss(torch.nn.Module):
    def __init__(self, beta=0.3, log_like=False):
        super(FLoss, self).__init__()
        self.beta = beta
        self.log_like = log_like

    def forward(self, prediction, target):
        EPS = 1e-10
        N = prediction.size(0)
        TP = (prediction * target).view(N, -1).sum(dim=1)
        H = self.beta * target.view(N, -1).sum(dim=1) + prediction.view(N, -1).sum(dim=1)
        fmeasure = (1 + self.beta) * TP / (H + EPS)
        if self.log_like:
            floss = -torch.log(fmeasure)
        else:
            floss  = (1 - fmeasure)
        return floss.mean()

def iou(pred, mask):
    inter = (pred * mask) .sum(dim=(2, 3))
    union = (pred + mask) .sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return wiou.mean()

class BCEloss(nn.Module):
    def __init__(self):
        super(BCEloss, self).__init__()
        self.bce = nn.BCELoss()
        self.floss=FLoss()

    def forward(self, sm,se,label,edge):
        mask_loss = self.bce(sm,label) + 0.6*iou(sm,label) +self.floss(sm,label)
        edge_loss = self.bce(se, edge) + 0.6*iou(se,edge) +self.floss(se,edge)
        total_loss = mask_loss+0.5*edge_loss
        return [total_loss, mask_loss, mask_loss]

def count_param(model):
    param_count = 0
    for param in model.parameters():
        param_count += param.view(-1).size()[0]
    return param_count

class Run:
    def __init__(self):
        self.train_set = EORSSD(dataset_root, 'train', aug=True)
        self.train_loader = data.DataLoader(self.train_set, shuffle=True, batch_size=batch_size, num_workers=4, drop_last=True)  #true
        self.test_set = EORSSD(dataset_root, 'test', aug=False)
        self.test_loader = data.DataLoader(self.test_set, shuffle=False, batch_size=1, num_workers=4, drop_last=False)

        self.init_lr = 1e-4
        self.min_lr = 1e-5         
        self.train_epoch = 60

        self.net = AGNet()
        self.net.load_state_dict(torch.load(os.path.join(ckpt_folder, 'trained', 'eorssd.pth')))
        self.loss=BCEloss()

    def train(self):
        self.net.train().cuda()
        max_F,mF=0.80,0.85
        base, head = [], []
        for name, param in self.net.named_parameters():
            if 'bkbone' in name:
                base.append(param)
            else:
                head.append(param)
        optimizer = optim.Adam([{'params': base}, {'params': head}], lr=self.init_lr, weight_decay=5e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.train_epoch,eta_min=self.min_lr)
        for epc in range(1, self.train_epoch + 1):
            records = [0] * 3
            N = 0
            optimizer.param_groups[0]['lr'] = 0.1 * optimizer.param_groups[1]['lr']  # for backbone
            optimizer.param_groups[1]['lr'] = optimizer.param_groups[1]['lr']
            for image, label, edge in tqdm(self.train_loader):
                # prepare input data\n",
                image, label, edge = image.cuda(), label.cuda(), edge.cuda()
                B = image.size(0)
                # forward\n",
                optimizer.zero_grad()
                sm,se= self.net(image)
                losses_list = self.loss(sm,se,label,edge)
                # compute loss\n",
                total_loss = losses_list[0].mean()
                # record loss\n",
                N += B
                for i in range(len(records)):
                    records[i] += losses_list[i].mean().item() * B
                # backward\n",
                total_loss.backward()
                optimizer.step()
            # update learning rate\n",
            scheduler.step()
            F,mf=self.test(epc)
            if F>max_F:
                cache_model(self.net, os.path.join(ckpt_folder, 'trained', 'trained.pth'), multi_gpu)
                max_F=F
            if mf>mF:
                cache_model(self.net, os.path.join(ckpt_folder, 'trained', 'trained_m.pth'), multi_gpu)
                mF=mf
            if epc==self.train_epoch:
                cache_model(self.net, os.path.join(ckpt_folder, 'trained', 'last.pth'), multi_gpu)
            # print training information\n",
            records = proc_loss(records, N, 4)
            print('epoch: {} || total loss: {} || mask loss: {} || cur_mean_F: {:.4f} || cur_max_F:{:.4f}'
                  .format(epc, records[0], records[1], mF, max_F))
        print('finish training.'+'maxF:',max_F)


    def test(self,ep):
        self.net.eval().cuda()
        print("params:",(count_param(self.net)/1e6))
        avg_f, mae,img_num = 0.0,0.0, 0
        score = torch.zeros(255)
        for image, label, prefix in self.test_loader:
            with torch.no_grad():
                image, label = image.cuda(), label.cuda()
                B=image.size(0)
                smap,_= self.net(image)
                if ep%4==0:
                    for b in range(B):
                        path = os.path.join(output_folder, prefix[b] + '.png')
                        save_smap(smap[b, ...], path)
                if torch.mean(label) == 0.0:
                   continue
                if smap.shape != label.shape:
                    smap = F.interpolate(smap, size=(label.shape[2],label.shape[3]), mode='bilinear')
                img_num+=1
                mae += Eval_mae(smap, label)
                f_score= Eval_Fmeasure(smap, label)
                avg_f+=f_score
                score = avg_f / img_num
                """if ep%4==0:
                    for b in range(B):
                        path = os.path.join(output_folder, prefix[b]+'_'+str(round(f_score.mean().item(),4))[2:] + '.png')
                        save_smap(smap[b, ...], path)"""
        maxF=score.max().item()
        mean=score.mean().item()
        mae=(mae/img_num)
        print('finish testing.', 'F—value : {:.4f}\t'.format(maxF),'mF—value : {:.4f}\t'.format(mean),'MAE : {:.4f}\t'.format(mae))
        return maxF,mean



if __name__=='__main__':
    run=Run()
    #run.train()
    run.test(8)
