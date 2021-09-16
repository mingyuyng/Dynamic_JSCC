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
import math

class DynaAWGNModel(BaseModel):

    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_L2', 'G_reward']
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
                                          n_downsample=opt.n_downsample, C_channel=opt.N_input,
                                          norm=opt.norm_EG, init_type=opt.init_type,
                                          init_gain=opt.init_gain, gpu_ids=self.gpu_ids,
                                          image_W=opt.image_size, image_H=opt.image_size, method=opt.method, N_output=opt.N_options + 1)

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

        if self.opt.SNR is not None:
            self.snr = torch.ones(self.real_A.shape[0], 1).to(self.device) * self.opt.SNR
        else:
            self.snr = torch.rand(self.real_A.shape[0], 1).to(self.device) * 20

        # Generate latent vector
        latent, z = self.netE(self.real_A, self.snr)

        # Generate decision mask
        self.hard_mask, self.soft_mask, prob = self.netP(z, self.snr, self.temp)
        
        #self.hard_mask = torch.cat((torch.ones((self.hard_mask.shape[0], 4), device=self.device), torch.zeros((self.hard_mask.shape[0], 0), device=self.device)), -1)

        # Normalize each channel
        latent_sum = torch.sqrt((latent**2).mean((-2, -1), keepdim=True))
        latent = latent / latent_sum

        # Reshape the latent
        N, C, W, H = latent.shape
        pad = torch.ones((N, self.opt.N_options), device=self.device)
        self.hard_mask = torch.cat((pad, self.hard_mask), -1)
        self.soft_mask = torch.cat((pad, self.soft_mask), -1)

        self.count = self.hard_mask.sum(-1)
        latent_res = latent.view(N, 2 * self.opt.N_options, -1)

        if self.opt.selection:
            # zero out the redundant latents
            if not self.opt.is_test:
                latent_res = latent_res * self.soft_mask.unsqueeze(-1)
            else:
                latent_res = latent_res * self.hard_mask.unsqueeze(-1)

        if self.opt.is_noise is not None:
            with torch.no_grad():
                self.sigma = 10**(-self.snr / 20)
                noise = self.sigma.view(self.real_A.shape[0], 1, 1) * torch.randn_like(latent_res)
                if self.opt.selection:
                    if not self.opt.is_test:
                        noise = noise * self.soft_mask.unsqueeze(-1)
                    else:
                        noise = noise * self.hard_mask.unsqueeze(-1)
            latent_res = latent_res + noise

        self.fake = self.netG(latent_res.view(latent.shape), self.snr)

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""

        self.loss_G_L2 = self.criterionL2(self.fake, self.real_B) * self.opt.lambda_L2
        if self.opt.selection:
            self.loss_G_reward = torch.mean(self.count)
        else:
            self.loss_G_reward = torch.zeros_like(self.loss_G_L2)

        self.loss_reg = 0
        for param in self.netP.parameters():
            self.loss_reg += torch.sum(param**2)

        self.loss_G = self.loss_G_L2 + self.opt.lambda_reward * self.loss_G_reward
        self.loss_G.backward()

    def optimize_parameters(self):

        self.forward()
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G.step()             # udpate G's weights

    def update_temp(self):
        self.temp *= math.exp(-self.opt.eta)
        self.temp = max(self.temp, 0.005)
        if self.opt.constant:
            self.temp = 0.67
