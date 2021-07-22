# Copyright (C) 2017 NVIDIA Corporation. All rights reserved.
# Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import numpy as np
import torch
import os
from torch.autograd import Variable
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import scipy.io as sio
import random

class DynaAWGNModel(BaseModel):

    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_L2', 'G_reward', 'G_soft_reward', 'G_entropy']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_A', 'fake', 'real_B']

        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        self.model_names = ['E', 'G', 'P']

        # define networks (both generator and discriminator)
        self.netE = networks.define_dynaE(input_nc=opt.input_nc, ngf=opt.ngf, max_ngf=opt.max_ngf,
                                          n_downsample=opt.n_downsample, C_channel=opt.C_channel,
                                          n_blocks=opt.n_blocks, norm=opt.norm_EG, init_type=opt.init_type,
                                          init_gain=opt.init_gain, gpu_ids=self.gpu_ids, first_kernel=opt.first_kernel)

        self.netG = networks.define_dynaG(output_nc=opt.output_nc, ngf=opt.ngf, max_ngf=opt.max_ngf,
                                          n_downsample=opt.n_downsample, C_channel=opt.C_channel,
                                          n_blocks=opt.n_blocks, norm=opt.norm_EG, init_type=opt.init_type,
                                          init_gain=opt.init_gain, gpu_ids=self.gpu_ids, first_kernel=opt.first_kernel, activation=opt.activation)

        self.netP = networks.define_dynaP(ngf=opt.ngf, max_ngf=opt.max_ngf,
                                          n_downsample=opt.n_downsample, C_channel=opt.C_channel,
                                          norm=opt.norm_EG, init_type=opt.init_type,
                                          init_gain=opt.init_gain, gpu_ids=self.gpu_ids,
                                          image_W=opt.image_size, image_H=opt.image_size, is_infer=opt.is_infer, method=opt.method)

        # if self.isTrain and self.is_GAN:  # define a discriminator;
        self.size_out = (opt.size // (2**opt.n_downsample))**2

        print('---------- Networks initialized -------------')

        # set loss functions and optimizers
        if self.isTrain:

            self.criterionL2 = torch.nn.MSELoss()
            params = list(self.netE.parameters()) + list(self.netG.parameters()) + list(self.netP.parameters())

            self.optimizer_G = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))
            #self.optimizer_G = torch.optim.SGD(params, lr=opt.lr, momentum=0.9)
            self.optimizers.append(self.optimizer_G)

        self.opt = opt
        self.temp = opt.temp

    def name(self):
        return 'DynaAWGN_Model'

    def set_input(self, image):
        self.real_A = image.clone().to(self.device)
        self.real_B = image.clone().to(self.device)

    def set_encode(self, image):
        self.real_A = image.clone().to(self.device)
        self.real_B = image.clone().to(self.device)

    def set_decode(self, latent):
        self.latent = latent.to(self.device)

    def set_img_path(self, path):
        self.image_paths = path

    def forward(self):

        # Generate latent vector
        latent, z = self.netE(self.real_A)

        if not self.opt.is_infer and self.opt.method == 'gumbel':
            #hard_mask, prob, soft = self.netP(z, self.temp)
            hard_mask, soft_mask, prob, count = self.netP(z, self.temp)
            
            count = count.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            self.hard_mask = hard_mask.squeeze()
            self.soft_mask = soft_mask.squeeze()
            self.prob = prob.squeeze()
            #self.soft_mask = soft.squeeze()
            
            #dummy = torch.ones(self.hard_mask.shape[0], 4).to(self.hard_mask.device)
            #self.hard_mask = torch.cat((dummy, self.hard_mask), -1)
            #self.soft_mask = torch.cat((dummy, self.soft), -1)
        else:
            hard_mask, prob = self.netP(z, self.temp)
            self.hard_mask = hard_mask.squeeze()
            self.prob = prob.squeeze()
            count = self.hard_mask.sum(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            #dummy = torch.ones(self.hard_mask.shape[0], 4).to(self.hard_mask.device)
            #self.hard_mask = torch.cat((dummy, self.hard_mask), -1)
        
        if self.opt.selection:

            N, self.C, _, _ = latent.shape
            #count = self.hard_mask.sum(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

            # zero out the redundant latents
            #latent[self.hard_mask == 0, :, :] = 0
            # with torch.no_grad():
            if not self.opt.is_infer and self.opt.method == 'gumbel':
                latent = latent * self.soft_mask.unsqueeze(-1).unsqueeze(-1)
            else:
                latent = latent * self.hard_mask.unsqueeze(-1).unsqueeze(-1)

            # Normalization
            latent_sum = torch.sqrt((latent**2).mean((-3, -2, -1), keepdim=True) * self.C / count)
            latent = latent / (latent_sum + 1e-5)

        else:
            # Normalization
            latent_sum = torch.sqrt((latent**2).mean((-3, -2, -1), keepdim=True))
            latent = latent / latent_sum

        if self.opt.SNR is not None:

            with torch.no_grad():
                self.sigma = 10**(-self.opt.SNR / 20)
                noise = self.sigma * torch.randn_like(latent)

                if self.opt.selection:
                    #noise[self.hard_mask == 0, :, :] = 0
                    if not self.opt.is_infer and self.opt.method == 'gumbel':
                        noise = noise * self.soft_mask.unsqueeze(-1).unsqueeze(-1)
                    else:
                        noise = noise * self.hard_mask.unsqueeze(-1).unsqueeze(-1)

            latent = latent + noise

        self.fake = self.netG(latent)

        if torch.isnan(self.fake.sum()):
            import pdb
            pdb.set_trace()  # breakpoint 833284f7 //

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""

        self.loss_G_L2 = self.criterionL2(self.fake, self.real_B) * self.opt.lambda_L2
        if self.opt.selection:
            self.loss_G_reward = torch.mean(self.hard_mask.sum(-1) / self.C) * self.opt.lambda_reward
            self.loss_G_soft_reward = torch.mean(self.prob.sum(-1) / self.C) * self.opt.lambda_soft_reward
        else:
            self.loss_G_reward = torch.zeros_like(self.loss_G_L2)
            self.loss_G_soft_reward = torch.zeros_like(self.loss_G_L2)
        
        self.loss_G_entropy = torch.mean(torch.sum(self.prob * self.prob.log(), -1))
        self.loss_G = self.loss_G_L2 + self.loss_G_soft_reward + self.loss_G_reward
        self.loss_G.backward()

    def optimize_parameters(self):

        self.forward()
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G.step()             # udpate G's weights

    def cal_weight_L2(self):

        sum_P = 0
        for param in self.netP.parameters():
            sum_P += torch.mean(param**2)

        sum_E = 0
        for param in self.netE.parameters():
            sum_E += torch.mean(param**2)

        sum_G = 0
        for param in self.netG.parameters():
            sum_G += torch.mean(param**2)

        return sum_P, sum_E, sum_G
