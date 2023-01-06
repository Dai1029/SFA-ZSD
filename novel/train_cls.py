from __future__ import print_function
import torch
import torch.optim as optim
from util import *
import torch.nn as nn
# from mmdetection.tools.faster_rcnn_utils import *
from torch.utils.data import DataLoader
import numpy as np
from dataset import *
from cls_models import ClsModelTrain
from mmdetection.splits import get_unseen_class_ids, get_unseen_class_labels
from torch.nn import functional as F


class TrainCls():
    def __init__(self, opt):

        self.classes_to_train = np.concatenate(([0], get_unseen_class_ids(opt.dataset, split=opt.classes_split)))
        self.opt = opt
        self.classes = get_unseen_class_labels(self.opt.dataset, split=opt.classes_split)
        self.best_acc = -100000
        self.best_acc_now_epoch = -100000
        self.isBestIter = False
        self.criterion = nn.CrossEntropyLoss()
        # self.criterion = nn.NLLLoss()

        self.dataset = None 
        self.val_accuracies = []
        self.test_cls = []
        self.init_model()
        self.best_epoch = 0
        self.best_cls_epoch = 0

    def init_model(self):
        self.classifier = ClsModelTrain(num_classes=len(self.classes_to_train))
        self.classifier.cuda()
        self.optimizer = optim.Adam(self.classifier.parameters(), lr=self.opt.lr_cls, betas=(0.5, 0.999))
        self.best_acc_now_epoch = -100000
        self.best_cls_epoch = 0

    def initDataSet(self, features, labels):
        self.dataset = FeaturesCls(self.opt, features=features, labels=labels, split='train', classes_to_train=self.classes_to_train)
        self.test_dataset = FeaturesCls(self.opt, split='test', classes_to_train=self.classes_to_train)
        self.dataloader = DataLoader(dataset=self.dataset, batch_size=self.opt.batch_size, num_workers=4, shuffle=True, pin_memory=True)
        self.test_dataloader = DataLoader(dataset=self.test_dataset, batch_size=self.opt.batch_size*50, num_workers=4, shuffle=True, pin_memory=True)
        
    def updateDataSet(self, features, labels):
        self.dataloader.dataset.replace(features, labels)

    def __call__(self, features=None, labels=None, gan_epoch=0):
        self.isBestIter = False
        self.gan_epoch = gan_epoch

        if self.dataset is None:
            self.initDataSet(features, labels)
            self.valacc, self.all_acc, _ = val(self.test_dataloader, self.classifier, self.criterion, self.opt, 0, verbose="Test")
            self.val_accuracies.append(self.all_acc)
        else:
            self.updateDataSet(features, labels)
        
        self.init_model()
        self.trainEpochs()
        self.best_acc = max(self.best_acc, self.valacc)

    def margin_loss(self, preds, in_label):
        # normal = F.softmax(preds, -1)
        in_label = in_label.cpu().numpy()
        class_uni = np.unique(in_label)
        pwd_loss = 0
        for i in class_uni:
            p_ind = np.where(in_label == i)[0]
            # n_ind = np.where(in_label != i)[0]
            min_loss = min(preds[p_ind, i])
            max_loss = max(preds[p_ind, i])
            pwd_loss += F.relu(0.4 - max_loss + min_loss)
        return pwd_loss/len(class_uni)

    def trainEpochs(self):
        for epoch in range(self.opt.nepoch_cls):
            self.classifier.train()
            loss_epoch = 0
            preds_all = []
            gt_all = []
            for ite, (in_feat, in_label) in enumerate(self.dataloader):
                in_feat = in_feat.type(torch.float).cuda()
                in_label = in_label.cuda()
                preds = self.classifier(feats=in_feat, classifier_only=True)
                # add wei biao qian
                loss_cls = self.criterion(preds, in_label)
                loss_m = self.margin_loss(preds, in_label)
                loss = loss_cls + loss_m
                loss_epoch += loss.item()

                # orign
                # loss = self.criterion(preds, in_label)
                # loss_epoch += loss.item()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                preds_all.append(preds.data.cpu().numpy())
                gt_all.append(in_label.data.cpu().numpy())

                if ite + 1 % 100 == 0 or ite + 1 == len(self.dataloader):
                    print(f'Cls Train Epoch [{epoch + 1:02}/{self.opt.nepoch_cls}] Iter [{ite + 1 :04}/{len(self.dataloader)}] {(ite + 1) * 100 / len(self.dataloader):0.2f}% Loss: {loss_epoch / ite :0.4f} Loss-cls: {loss_cls :0.4f} Loss-m: {loss_m :0.4f} lr: {get_lr(self.optimizer):0.5f}')
                    # print(f'Cls Train Epoch [{epoch + 1:02}/{self.opt.nepoch_cls}] Iter [{ite + 1 :04}/{len(self.dataloader)}] {(ite + 1) * 100 / len(self.dataloader):0.2f}% Loss: {loss_epoch / ite :0.4f} Loss-cls: {loss :0.4f} lr: {get_lr(self.optimizer):0.5f}')

                # validate on test set
            adjust_learning_rate(self.optimizer, epoch, self.opt)

            self.valacc, self.all_acc, c_mat_test = val(self.test_dataloader, self.classifier, self.criterion, self.opt, epoch, verbose="Test")
            self.val_accuracies.append(self.all_acc)

            if self.best_acc <= self.valacc:
                print(f"saved best model best accuracy : {self.valacc:0.4f}")
                torch.save({'state_dict': self.classifier.state_dict(), 'epoch': epoch}, f"{self.opt.outname}/classifier_best.pth")
                self.isBestIter = True
                # np.save(f'{self.opt.outname}/confusion_matrix_Test.npy', c_mat_test)
                self.best_acc = self.valacc
                self.best_epoch = self.gan_epoch+1
            if self.best_acc_now_epoch <= self.valacc:
                self.best_acc_now_epoch = self.valacc
                self.best_cls_epoch = epoch+1
                print(f"saved this epoch the best model best accuracy :[{self.gan_epoch+1}]-[{epoch+1}] {self.best_acc_now_epoch:0.4f}")
                if self.gan_epoch+1 >= 45:
                    torch.save({'state_dict': self.classifier.state_dict(), 'epoch': epoch}, f"{self.opt.outname}/{self.gan_epoch+1}.pth")
                # torch.save({'state_dict': self.classifier.state_dict(), 'epoch': epoch}, f"{self.opt.outname}/classifier_best_latest.pth")
        if self.gan_epoch+1 >= 45:
            self.test_cls.append([self.gan_epoch+1, self.best_cls_epoch, self.best_acc_now_epoch])   # dxm

        # _,_, c_mat_train = compute_per_class_acc(np.concatenate(gt_all), np.concatenate(preds_all), self.opt, verbose='Train')
        # np.save(f'{self.opt.outname}/confusion_matrix_Train.npy', c_mat_train)
        # torch.save({'state_dict': self.classifier.state_dict(), 'epoch': epoch}, f"{self.opt.outname}/classifier_latest.pth")

        print(f"In {self.best_cls_epoch:02} epoch, the model best accuracy : {self.best_acc_now_epoch:0.4f}")
        print(f"[{self.best_epoch:04}] best model accuracy {self.best_acc}")






