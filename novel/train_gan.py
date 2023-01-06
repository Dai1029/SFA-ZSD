from __future__ import print_function
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
from torch.autograd import Variable
import math

from splits import get_unseen_class_ids
from util import *
import model
from cls_models import ClsModel, ClsUnseen
import torch.nn.functional as F

class TrainGAN():
    def __init__(self, opt, attributes, unseenAtt, unseenLabels, gen_type='FG'):
        
        '''
        CLSWGAN trainer
        Inputs:
            opt -- arguments
            unseenAtt -- embeddings vector of unseen classes
            unseenLabels -- labels of unseen classes
            attributes -- embeddings vector of all classes
        '''
        self.opt = opt

        self.gen_type = gen_type
        self.Wu_Labels = np.array([i for i, l in enumerate(unseenLabels)])
        print(f"Wu_Labels {self.Wu_Labels}")
        self.Wu = unseenAtt

        self.unseen_classifier = ClsUnseen(unseenAtt)
        self.unseen_classifier.cuda()
        self.unseen_classifier = loadUnseenWeights(opt.pretrain_classifier_unseen, self.unseen_classifier)

        self.classifier = ClsModel(num_classes=opt.nclass_all)
        self.classifier.cuda()
        self.classifier = loadFasterRcnnCLSHead(opt.pretrain_classifier, self.classifier)

        # self.classifier_unseen_cls = ClsModel(num_classes=15)
        # self.classifier_unseen_cls.cuda()
        # self.classifier = loadFasterRcnnCLSHead(opt.outname+'/classifier_best.pth', self.classifier_unseen_cls)

        for p in self.classifier.parameters():
            p.requires_grad = False
        
        for p in self.unseen_classifier.parameters():
            p.requires_grad = False

        self.ntrain = opt.gan_epoch_budget
        self.attributes = attributes.data.numpy()

        print(f"# of training samples: {self.ntrain}")
        # initialize generator and discriminator
        self.netG = model.MLP_G(self.opt)
        self.netD = model.MLP_CRITIC(self.opt)

        if self.opt.cuda and torch.cuda.is_available():
            self.netG = self.netG.cuda()
            self.netD = self.netD.cuda()

        # print('\n\n#############################################################\n')
        # print(self.netG, '\n')
        # print(self.netD)
        # print('\n#############################################################\n\n')

        # classification loss, Equation (4) of the paper
        self.cls_criterion = nn.NLLLoss()

        if self.opt.cuda:
            self.cls_criterion.cuda()

        self.optimizerD = optim.Adam(self.netD.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
        self.optimizerG = optim.Adam(self.netG.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, 0.999))

    def __call__(self, epoch, features, labels):
        """
        Train GAN for one epoch
        Inputs:
            epoch: current epoch
            features: current epoch subset of features
            labels: ground truth labels
        """
        self.epoch = epoch
        self.features = features
        self.labels = labels
        self.ntrain = len(self.labels)
        self.trainEpoch()
    
    def load_checkpoint(self):
        checkpoint = torch.load(self.opt.netG)
        self.netG.load_state_dict(checkpoint['state_dict'])
        epoch = checkpoint['epoch']
        self.netD.load_state_dict(torch.load(self.opt.netD)['state_dict'])
        print(f"loaded weights from epoch: {epoch} \n{self.opt.netD} \n{self.opt.netG} \n")
        return epoch

    def save_checkpoint(self, state='latest'):
        torch.save({'state_dict': self.netD.state_dict(), 'epoch': self.epoch}, f'{self.opt.outname}/disc_{state}.pth')
        torch.save({'state_dict': self.netG.state_dict(), 'epoch': self.epoch}, f'{self.opt.outname}/gen_{state}.pth')

    def generate_syn_feature(self, labels, attribute, num=100, no_grad=True):
        """
        generates features
        inputs:
            labels: features labels to generate nx1 n is number of objects 
            attributes: attributes of objects to generate (nxd) d is attribute dimensions
            num: number of features to generate for each object
        returns:
            1) synthesised features 
            2) labels of synthesised  features 
        """

        nclass = labels.shape[0]
        syn_feature = torch.FloatTensor(nclass * num , self.opt.resSize)
        syn_label = torch.LongTensor(nclass*num)

        syn_att = torch.FloatTensor(num, self.opt.attSize)
        syn_noise = torch.FloatTensor(num, self.opt.nz)
        
        if self.opt.cuda:
            syn_att = syn_att.cuda()
            syn_noise = syn_noise.cuda()
        if no_grad is True:
            with torch.no_grad():
                for i in range(nclass):
                    label = labels[i]
                    iclass_att = attribute[i]
                    syn_att.copy_(iclass_att.repeat(num, 1))
                    syn_noise.normal_(0, 1)
                    output = self.netG(Variable(syn_noise), Variable(syn_att))
                
                    syn_feature.narrow(0, i*num, num).copy_(output.data.cpu())
                    syn_label.narrow(0, i*num, num).fill_(label)
        else:
            for i in range(nclass):
                label = labels[i]
                iclass_att = attribute[i]
                syn_att.copy_(iclass_att.repeat(num, 1))
                syn_noise.normal_(0, 1)
                output = self.netG(Variable(syn_noise), Variable(syn_att))
            
                syn_feature.narrow(0, i*num, num).copy_(output.data.cpu())
                syn_label.narrow(0, i*num, num).fill_(label)

        return syn_feature, syn_label

    def sample(self):
        """
        randomaly samples one batch of data
        returns (1)real features, (2)labels (3) attributes embeddings
        """
        idx = torch.randperm(self.ntrain)[0:self.opt.batch_size]
        batch_feature = torch.from_numpy(self.features[idx])
        batch_label = torch.from_numpy(self.labels[idx])
        batch_att = torch.from_numpy(self.attributes[batch_label])
        if 'BG' == self.gen_type:
            batch_label *= 0
        return batch_feature, batch_label, batch_att

    def calc_gradient_penalty(self, real_data, fake_data, input_att):
        alpha = torch.rand(self.opt.batch_size, 1)
        alpha = alpha.expand(real_data.size())
        if self.opt.cuda:
            alpha = alpha.cuda()

        interpolates = alpha * real_data + ((1 - alpha) * fake_data)

        if self.opt.cuda:
            interpolates = interpolates.cuda()

        interpolates = Variable(interpolates, requires_grad=True)

        disc_interpolates = self.netD(interpolates, Variable(input_att))

        ones = torch.ones(disc_interpolates.size())
        if self.opt.cuda:
            ones = ones.cuda()

        gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                grad_outputs=ones,
                                create_graph=True, retain_graph=True, only_inputs=True)[0]

        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.opt.lambda1
        return gradient_penalty

    def get_z_random(self):
        """
        returns normal initialized noise tensor 
        """
        z = torch.cuda.FloatTensor(self.opt.batch_size, self.opt.nz)
        z.normal_(0, 1)
        return z

    def contras_criterion(self, features, labels):
        feature_z = F.normalize(features, dim=1)
        temperature = 0.05
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        batch_size = features.shape[0]
        labels = labels.contiguous().view(-1, 1)  # contiguous()重新定义结构，输出为1列的标签
        mask = torch.eq(labels, labels.t()).float().to(device)  # 得到0,1的矩阵

        anchor_dot_contrast = torch.div(
            torch.matmul(feature_z, feature_z.t()),
            temperature)
        #   特征矩阵自乘 除以temperature

        # normalize the logits for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        # dim=1取每行的最大 1*batch   keepdim=True表示输出和输入的维度一样

        logits = anchor_dot_contrast - logits_max.detach()

        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )
        # .scatter索引
        mask = mask * logits_mask
        single_samples = (mask.sum(1) == 0).float()

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        # invoid to devide the zero
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + single_samples)

        # loss
        # filter those single sample
        loss = - mean_log_prob_pos * (1 - single_samples)
        loss = loss.sum() / (loss.shape[0] - single_samples.sum())

        return loss

    def trainEpoch(self):
        for i in range(0, self.ntrain, self.opt.batch_size):
            # import pdb; pdb.set_trace()
            input_res, input_label, input_att = self.sample()  # take batch_size seen feature label att

            if self.opt.batch_size != input_res.shape[0]:
                continue
            input_res, input_label, input_att = input_res.type(torch.FloatTensor).cuda(), input_label.type(
                torch.LongTensor).cuda(), input_att.type(torch.FloatTensor).cuda()
            ############################
            # (1) Update D network: optimize WGAN-GP objective, Equation (2)
            ###########################
            for p in self.netD.parameters():  # reset requires_grad
                p.requires_grad = True  # they are set to False below in netG update

            for iter_d in range(self.opt.critic_iter):
                self.netD.zero_grad()
                # train with realG
                # sample a mini-batch
                # sparse_real = self.opt.resSize - input_res[1].gt(0).sum()
                input_resv = Variable(input_res)
                input_attv = Variable(input_att)

                criticD_real = self.netD(input_resv, input_attv)
                criticD_real = criticD_real.mean()

                # train with fakeG
                noise = self.get_z_random()
                noisev = Variable(noise)
                fake = self.netG(noisev, input_attv)
                criticD_fake = self.netD(fake.detach(), input_attv)
                criticD_fake = criticD_fake.mean()

                # gradient penalty
                gradient_penalty = self.calc_gradient_penalty(input_res, fake.data, input_att)
                # gradient_penalty.backward()
                # unseenc_errG.backward()
                Wasserstein_D = criticD_real - criticD_fake
                D_cost = criticD_fake - criticD_real + gradient_penalty
                # cls loss? dxm+ c_real

                D_cost.backward()
                self.optimizerD.step()

            ############################
            # (2) Update G network: optimize WGAN-GP objective, Equation (2)
            ###########################
            for p in self.netD.parameters():  # reset requires_grad
                p.requires_grad = False  # avoid computation

            self.netG.zero_grad()
            input_attv = Variable(input_att)
            noise = self.get_z_random()
            noisev = Variable(noise)
            fake = self.netG(noisev, input_attv)
            criticG_fake = self.netD(fake, input_attv)
            criticG_fake = criticG_fake.mean()
            G_cost = -criticG_fake

            # ---------------------
            noise2 = self.get_z_random()
            noise2v = Variable(noise2)
            fake2 = self.netG(noise2v, input_attv)

            lz = torch.mean(torch.abs(fake2 - fake)) / torch.mean(torch.abs(noise2v - noisev))
            eps = 1 * 1e-5
            loss_lz = 1 / (lz + eps)
            loss_lz *= self.opt.lz_ratio
            # ---------------------

            # classification loss
            # seen
            c_errG = self.cls_criterion(self.classifier(feats=fake, classifier_only=True), Variable(input_label))
            c_errG = self.opt.cls_weight * c_errG
            # --------------------------------------------

            # unseen
            fake_unseen_f, fake_unseen_l = self.generate_syn_feature(self.Wu_Labels, self.Wu,
                                                                     num=self.opt.batch_size // 4, no_grad=False)

            fake_pred = self.unseen_classifier(feats=fake_unseen_f.cuda(), classifier_only=True)
            unseenc_errG = self.opt.cls_weight_unseen * self.cls_criterion(fake_pred, Variable(fake_unseen_l.cuda()))

            # contrastive loss  dxm
            unseen_ca = get_unseen_class_ids(self.opt.dataset, self.opt.classes_split)
            fake_unseen_l_id = torch.tensor([unseen_ca[i] for i in fake_unseen_l])
            contras_loss = self.contras_criterion(
                torch.cat([fake_unseen_f.cuda(), input_res], 0),
                torch.cat([fake_unseen_l_id.cuda(), input_label], 0))
            # # contras_loss *= 0.005
            # # contras_loss *= 0.001
            contras_loss *= 0.0005


            # unseen_cls
            # ---------------------------------------------
            # Total loss
            if self.epoch >= 5:
                errG = G_cost + c_errG + loss_lz + unseenc_errG + contras_loss
                # + contras_loss
            else:
                errG = G_cost + c_errG + loss_lz + contras_loss
            errG.backward()
            self.optimizerG.step()

            print(f"{self.gen_type} [{self.epoch + 1:02}/{self.opt.nepoch:02}] [{i:05}/{int(self.ntrain)}] "
                  f"Loss:{errG.item():06.4f} D-loss:{D_cost.data.item():06.4f} G-loss:{G_cost.data.item():06.4f} W-dist:{Wasserstein_D.data.item():06.4f} "
                  f"seen-loss:{c_errG.data.item():06.4f} unseen-loss:{unseenc_errG.data.item():06.4f} loss-div:{loss_lz.item():06.4f} contras_loss:{contras_loss:06.4f}")
            # contras_loss:{contras_loss:06.4f}
        self.netG.eval()