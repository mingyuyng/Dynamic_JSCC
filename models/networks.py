# Copyright (C) 2017 NVIDIA Corporation. All rights reserved.
# Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
import numpy as np
from torch.nn import functional as F
from typing import List, Callable, Union, Any, TypeVar, Tuple
from math import exp
from torch.distributions.utils import broadcast_all, probs_to_logits, logits_to_probs, lazy_property, clamp_probs
from torch.distributions.normal import Normal
from torch.distributions import kl_divergence

# from torch import tensor as Tensor

Tensor = TypeVar('torch.tensor')
###############################################################################
# Functions
###############################################################################


class Identity(nn.Module):
    def forward(self, x):
        return x


def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        def norm_layer(x): return Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net


def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.n_epochs> epochs
    and linearly decay the rate to zero over the next <opt.n_epochs_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(opt.n_epochs_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


class Flatten(nn.Module):
    def forward(self, x):
        N, C, H, W = x.size()  # read in N, C, H, W
        return x.view(N, -1)  # "flatten" the C * H * W values into a single vector per image


class Normalize(nn.Module):
    def forward(self, x, power):
        N = x.shape[0]
        pwr = torch.mean(x**2, (1, 2, 3), True)
        return np.sqrt(power) * x / torch.sqrt(pwr)


def define_G(output_nc, ngf, max_ngf, n_downsample, C_channel, n_blocks, norm="instance", init_type='kaiming', init_gain=0.02, gpu_ids=[], first_kernel=7, activation='sigmoid'):
    net = None
    norm_layer = get_norm_layer(norm_type=norm)
    net = Generator(output_nc=output_nc, ngf=ngf, max_ngf=max_ngf, C_channel=C_channel, n_blocks=n_blocks, n_downsampling=n_downsample, norm_layer=norm_layer, padding_type="reflect", first_kernel=first_kernel, activation_=activation)
    return init_net(net, init_type, init_gain, gpu_ids)


def define_JSCC_G(C_channel, init_type='kaiming', init_gain=0.02, gpu_ids=[]):
    net = None
    net = JSCC_decoder(C_channel)
    return init_net(net, init_type, init_gain, gpu_ids)


def print_network(net):
    if isinstance(net, list):
        net = net[0]
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)

##############################################################################
# Losses
##############################################################################


class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp', 'none']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label.to(torch.float32)
        else:
            target_tensor = self.fake_label.to(torch.float32)
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss


def cal_gradient_penalty(netD, real_data, fake_data, device, type='mixed', constant=1.0, lambda_gp=10.0):
    """Calculate the gradient penalty loss, used in WGAN-GP paper https://arxiv.org/abs/1704.00028

    Arguments:
        netD (network)              -- discriminator network
        real_data (tensor array)    -- real images
        fake_data (tensor array)    -- generated images from the generator
        device (str)                -- GPU / CPU: from torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        type (str)                  -- if we mix real and fake data or not [real | fake | mixed].
        constant (float)            -- the constant used in formula ( | |gradient||_2 - constant)^2
        lambda_gp (float)           -- weight for this loss

    Returns the gradient penalty loss
    """
    if lambda_gp > 0.0:
        if type == 'real':   # either use real images, fake images, or a linear interpolation of two.
            interpolatesv = real_data
        elif type == 'fake':
            interpolatesv = fake_data
        elif type == 'mixed':
            alpha = torch.rand(real_data.shape[0], 1, device=device)
            alpha = alpha.expand(real_data.shape[0], real_data.nelement() // real_data.shape[0]).contiguous().view(*real_data.shape)
            interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)
        else:
            raise NotImplementedError('{} not implemented'.format(type))
        interpolatesv.requires_grad_(True)
        disc_interpolates = netD(interpolatesv)
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolatesv,
                                        grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                        create_graph=True, retain_graph=True, only_inputs=True)
        gradients = gradients[0].view(real_data.size(0), -1)  # flat the data
        gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp        # added eps
        return gradient_penalty, gradients
    else:
        return 0.0, None


##############################################################################
# Encoder
##############################################################################
class Encoder(nn.Module):

    def __init__(self, input_nc, ngf=64, max_ngf=512, C_channel=16, n_blocks=2, n_downsampling=2, norm_layer=nn.BatchNorm2d, padding_type="reflect", first_kernel=7):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            ngf (int)           -- the number of filters in the first conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert(n_downsampling >= 0)
        assert(n_blocks >= 0)
        super(Encoder, self).__init__()

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        activation = nn.ReLU(True)

        model = [nn.ReflectionPad2d((first_kernel - 1) // 2),
                 nn.Conv2d(input_nc, ngf, kernel_size=first_kernel, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 activation]

        # add downsampling layers
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(min(ngf * mult, max_ngf), min(ngf * mult * 2, max_ngf), kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(min(ngf * mult * 2, max_ngf)), activation]

        # add ResNet blocks
        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(min(ngf * mult, max_ngf), padding_type=padding_type, norm_layer=norm_layer, use_dropout=False, use_bias=use_bias)]

        self.model = nn.Sequential(*model)
        self.projection = nn.Conv2d(min(ngf * mult, max_ngf), C_channel, kernel_size=3, padding=1, stride=1, bias=use_bias)
        self.normalization = norm_layer(C_channel)

    def forward(self, input):
        z = self.model(input)
        return self.projection(z)


class Generator(nn.Module):
    def __init__(self, output_nc, ngf=64, max_ngf=512, C_channel=16, n_blocks=2, n_downsampling=2, norm_layer=nn.BatchNorm2d, padding_type="reflect", first_kernel=7, activation_='sigmoid'):
        assert (n_blocks >= 0)
        assert(n_downsampling >= 0)

        super(Generator, self).__init__()

        self.activation_ = activation_

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        activation = nn.ReLU(True)

        mult = 2 ** n_downsampling
        ngf_dim = min(ngf * mult, max_ngf)
        model = [nn.Conv2d(C_channel, ngf_dim, kernel_size=3, padding=1, stride=1, bias=use_bias)]

        for i in range(n_blocks):
            model += [ResnetBlock(ngf_dim, padding_type=padding_type, norm_layer=norm_layer, use_dropout=False, use_bias=use_bias)]

        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(min(ngf * mult, max_ngf), min(ngf * mult // 2, max_ngf),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(min(ngf * mult // 2, max_ngf)),
                      activation]

        model += [nn.ReflectionPad2d((first_kernel - 1) // 2), nn.Conv2d(ngf, output_nc, kernel_size=first_kernel, padding=0)]

        if activation_ == 'tanh':
            model += [nn.Tanh()]
        elif activation_ == 'sigmoid':
            model += [nn.Sigmoid()]

        self.model = nn.Sequential(*model)

    def forward(self, input):

        if self.activation_ == 'tanh':
            return self.model(input)
        elif self.activation_ == 'sigmoid':
            return 2 * self.model(input) - 1

#############################################################################################################


class bsc_channel(nn.Module):
    def __init__(self, opt):
        super(bsc_channel, self).__init__()
        self.opt = opt
        self.Temp = self.opt.temp

    def forward(self, x):

        # 1. Generating the probability for bernoulli distribution
        if self.opt.enc_type == 'prob':
            pass
        elif self.opt.enc_type == 'hard':
            index = torch.zeros_like(x)
            index[x > 0.5] = 1
            with torch.no_grad():
                bias = index - x
            x = x + bias
        elif self.opt.enc_type == 'soft':
            x = torch.sigmoid((x**2 - (x - 1)**2) / self.Temp)
        elif self.opt.enc_type == 'soft_hard':
            x = torch.sigmoid((x**2 - (x - 1)**2) / self.Temp)
            index = torch.zeros_like(x)
            index[x > 0.5] = 1
            with torch.no_grad():
                bias = index - x
            x = x + bias

        out_prob = self.opt.ber + x - 2 * self.opt.ber * x

        # 2. Sample the bernoulli distribution and generate decoder input
        if self.opt.sample_type == 'st':
            cha_out = torch.bernoulli(out_prob.detach())
            with torch.no_grad():
                bias = cha_out - out_prob
            dec_in = out_prob + bias

        elif self.opt.sample_type == 'gumbel_softmax':
            probs = clamp_probs(out_prob)
            uniforms = clamp_probs(torch.rand_like(out_prob))
            logits = (uniforms.log() - (-uniforms).log1p() + probs.log() - (-probs).log1p()) / self.Temp
            dec_in = torch.sigmoid(logits)
        elif self.opt.sample_type == 'gumbel_softmax_hard':
            probs = clamp_probs(out_prob)
            uniforms = clamp_probs(torch.rand_like(out_prob))
            logits = (uniforms.log() - (-uniforms).log1p() + probs.log() - (-probs).log1p()) / self.Temp
            dec_in = torch.sigmoid(logits)
            index = torch.zeros_like(x)
            index[dec_in > 0.5] = 1
            with torch.no_grad():
                bias = index - dec_in
            dec_in = dec_in + bias

        return dec_in

    def update_Temp(self, new_temp):
        self.Temp = new_temp


class awgn_channel(nn.Module):
    def __init__(self, opt):
        super(awgn_channel, self).__init__()
        self.opt = opt
        self.sigma = 10**(-opt.SNR / 20)

    def forward(self, x):

        noise = self.sigma * torch.randn_like(x)
        dec_in = x + noise
        return dec_in

##################################################################################################################################


class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()

        self.n_layers = n_layers

        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [[nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [[
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [[
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]]

        sequence += [[nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]]  # output 1 channel prediction map

        for n in range(len(sequence)):
            setattr(self, 'model' + str(n), nn.Sequential(*sequence[n]))

    def forward(self, input):
        """Standard forward."""
        res = [input]
        for n in range(self.n_layers + 1):
            model = getattr(self, 'model' + str(n))
            res.append(model(res[-1]))

        model = getattr(self, 'model' + str(self.n_layers + 1))
        out = model(res[-1])

        return res[1:], out


class PixelDiscriminator(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""

    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d):
        """Construct a 1x1 PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        """
        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        """Standard forward."""
        return self.net(input)


class MultiscaleDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d,
                 use_sigmoid=False, num_D=3, getIntermFeat=False, one_D_conv=False, one_D_conv_size=63):
        super(MultiscaleDiscriminator, self).__init__()

        self.num_D = num_D
        self.n_layers = n_layers
        self.getIntermFeat = getIntermFeat

        for i in range(num_D - 1):
            netD = NLayerDiscriminator(input_nc, ndf, n_layers, norm_layer, use_sigmoid, getIntermFeat)
            if getIntermFeat:
                for j in range(n_layers + 2):
                    setattr(self, 'scale' + str(i) + '_layer' + str(j), getattr(netD, 'model' + str(j)))
            else:
                setattr(self, 'layer' + str(i), netD.model)
        netD = NLayerDiscriminator(input_nc, ndf, n_layers, norm_layer, use_sigmoid, getIntermFeat, one_D_conv=one_D_conv, one_D_conv_size=one_D_conv_size)
        if getIntermFeat:
            for j in range(n_layers + 2):
                setattr(self, 'scale' + str(num_D - 1) + '_layer' + str(j), getattr(netD, 'model' + str(j)))
        else:
            setattr(self, 'layer' + str(num_D - 1), netD.model)

        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)
        self.one_D_conv = one_D_conv

    def singleD_forward(self, model, input):
        if self.getIntermFeat:
            result = [input]
            for i in range(len(model)):
                result.append(model[i](result[-1]))
            return result[1:]
        else:
            return [model(input)]

    def forward(self, input):
        num_D = self.num_D
        result = []
        input_downsampled = input

        for i in range(num_D - 1):
            if self.getIntermFeat:
                model = [getattr(self, 'scale' + str(i) + '_layer' + str(j)) for j in range(self.n_layers + 2)]
            else:
                model = getattr(self, 'layer' + str(i))
            result.append(self.singleD_forward(model, input_downsampled))
            if i != (num_D - 1):
                input_downsampled = self.downsample(input_downsampled)
        if self.getIntermFeat:
            model = [getattr(self, 'scale' + str(num_D - 1) + '_layer' + str(j)) for j in range(self.n_layers + 2)]
        else:
            model = getattr(self, 'layer' + str(num_D - 1))
        if self.one_D_conv:
            result.append(self.singleD_forward(model, input))
        else:
            result.append(self.singleD_forward(model, input_downsampled))
        return result

########################################################


def define_dynaE(input_nc, ngf, max_ngf, n_downsample, C_channel, n_blocks, norm='instance', init_type='kaiming', init_gain=0.02, gpu_ids=[], first_kernel=7):
    net = None
    norm_layer = get_norm_layer(norm_type=norm)
    net = Encoder_dyna(input_nc=input_nc, ngf=ngf, max_ngf=max_ngf, C_channel=C_channel, n_blocks=n_blocks, n_downsampling=n_downsample, norm_layer=norm_layer, padding_type="reflect", first_kernel=first_kernel)
    return init_net(net, init_type, init_gain, gpu_ids)


def define_dynaG(output_nc, ngf, max_ngf, n_downsample, C_channel, n_blocks, norm='instance', init_type='kaiming', init_gain=0.02, gpu_ids=[], first_kernel=7, activation='sigmoid'):
    net = None
    norm_layer = get_norm_layer(norm_type=norm)
    net = Generator_dyna(output_nc=output_nc, ngf=ngf, max_ngf=max_ngf, C_channel=C_channel, n_blocks=n_blocks, n_downsampling=n_downsample, norm_layer=norm_layer, padding_type="reflect", first_kernel=first_kernel, activation_=activation)
    return init_net(net, init_type, init_gain, gpu_ids)


def define_dynaP(ngf, max_ngf, C_channel, n_downsample, norm='instance', init_type='kaiming', init_gain=0.02, gpu_ids=[], image_W=32, image_H=32, method='gate', N_output=7):
    net = None
    norm_layer = get_norm_layer(norm_type=norm)
    net = Policy_dyna(ngf=ngf, max_ngf=max_ngf, C_channel=C_channel, n_downsampling=n_downsample, norm_layer=norm_layer, image_W=image_W, image_H=image_H, method=method, N_output=N_output)
    return init_net(net, 'normal', 0.002, gpu_ids)


class Encoder_dyna(nn.Module):

    def __init__(self, input_nc, ngf=64, max_ngf=512, C_channel=16, n_blocks=2, n_downsampling=2, norm_layer=nn.BatchNorm2d, padding_type="reflect", first_kernel=7):

        assert(n_downsampling >= 0)
        assert(n_blocks >= 0)
        super(Encoder_dyna, self).__init__()

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        activation = nn.ReLU(True)

        # Downscale network
        model = [nn.ReflectionPad2d((first_kernel - 1) // 2),
                 nn.Conv2d(input_nc, ngf, kernel_size=first_kernel, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 activation]

        # add downsampling layers
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(min(ngf * mult, max_ngf), min(ngf * mult * 2, max_ngf), kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(min(ngf * mult * 2, max_ngf)), activation]

        self.model_downsample = nn.Sequential(*model)

        # Resnet
        mult = 2 ** n_downsampling
        self.res1 = ResnetBlock(min(ngf * mult, max_ngf), padding_type=padding_type, norm_layer=norm_layer, use_dropout=False, use_bias=use_bias)
        self.res2 = ResnetBlock(min(ngf * mult, max_ngf), padding_type=padding_type, norm_layer=norm_layer, use_dropout=False, use_bias=use_bias)
        self.mod1 = modulation(min(ngf * mult, max_ngf))
        self.mod2 = modulation(min(ngf * mult, max_ngf))

        self.projection = nn.Conv2d(min(ngf * mult, max_ngf), C_channel, kernel_size=3, padding=1, stride=1, bias=use_bias)

    def forward(self, input, SNR):
        z = self.model_downsample(input)
        z = self.mod1(self.res1(z), SNR)
        z = self.mod2(self.res2(z), SNR)
        latent = self.projection(z)
        return latent, z


class Generator_dyna(nn.Module):
    def __init__(self, output_nc, ngf=64, max_ngf=512, C_channel=16, n_blocks=2, n_downsampling=2, norm_layer=nn.BatchNorm2d, padding_type="reflect", first_kernel=7, activation_='sigmoid'):
        assert (n_blocks >= 0)
        assert(n_downsampling >= 0)

        super(Generator_dyna, self).__init__()

        self.activation_ = activation_

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        activation = nn.ReLU(True)

        mult = 2 ** n_downsampling
        ngf_dim = min(ngf * mult, max_ngf)

        self.mask_conv = nn.Conv2d(C_channel, ngf_dim, kernel_size=3, padding=1, stride=1, bias=use_bias)

        model = []

        self.res1 = ResnetBlock(ngf_dim, padding_type=padding_type, norm_layer=norm_layer, use_dropout=False, use_bias=use_bias)
        self.res2 = ResnetBlock(ngf_dim, padding_type=padding_type, norm_layer=norm_layer, use_dropout=False, use_bias=use_bias)
        self.mod1 = modulation(ngf_dim)
        self.mod2 = modulation(ngf_dim)

        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(min(ngf * mult, max_ngf), min(ngf * mult // 2, max_ngf),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(min(ngf * mult // 2, max_ngf)),
                      activation]

        model += [nn.ReflectionPad2d((first_kernel - 1) // 2), nn.Conv2d(ngf, output_nc, kernel_size=first_kernel, padding=0)]

        if activation_ == 'tanh':
            model += [nn.Tanh()]
        elif activation_ == 'sigmoid':
            model += [nn.Sigmoid()]

        self.model = nn.Sequential(*model)

    def forward(self, input, SNR):
        z = self.mask_conv(input)
        z = self.mod1(self.res1(z), SNR)
        z = self.mod2(self.res2(z), SNR)

        if self.activation_ == 'tanh':
            return self.model(z)
        elif self.activation_ == 'sigmoid':
            return 2 * self.model(z) - 1


class Policy_dyna(nn.Module):
    def __init__(self, ngf=64, max_ngf=512, C_channel=16, N_output=7, n_downsampling=2, norm_layer=nn.BatchNorm2d, image_W=32, image_H=32, is_infer=False, method='gate', threshold=4):

        super(Policy_dyna, self).__init__()

        activation = nn.ReLU(True)

        mult = 2 ** n_downsampling
        image_W = image_W // mult
        image_H = image_H // mult

        # Policy network
        model = [nn.Linear(C_channel + 1, 64), activation, nn.BatchNorm1d(64),
                 nn.Linear(64, 64), activation, nn.BatchNorm1d(64),
                 nn.Linear(64, N_output)]
        self.model_gate = nn.Sequential(*model)

        self.method = method

    def forward(self, z, SNR, temp=5):

        # Policy/gate network
        N, C, W, H = z.shape
        z = self.model_gate(torch.cat((z.mean((-2, -1)), SNR), -1))      # N x C+1

        if self.method == 'gumbel':

            soft = nn.functional.gumbel_softmax(z, temp, dim=-1)

            with torch.no_grad():
                index = torch.zeros_like(soft)
                index[torch.arange(0, N), soft.argmax(-1)] = 1
                bias = index - soft

            hard = soft + bias
            soft_mask = one_hot_to_thermo(soft[:, 1:])
            hard_mask = one_hot_to_thermo(hard[:, 1:])

            return hard_mask, soft_mask, z


def one_hot_to_thermo(h):
    # 1. flip the order
    h = torch.flip(h, [-1])
    # 2. Accumulate sume
    s = torch.cumsum(h, -1)
    # 3. flip the result
    return torch.flip(s, [-1])


class modulation(nn.Module):
    def __init__(self, C_channel):

        super(modulation, self).__init__()

        activation = nn.ReLU(True)

        # Policy network
        model_multi = [nn.Linear(C_channel + 1, C_channel), activation,
                       nn.Linear(C_channel, C_channel), nn.Sigmoid()]

        model_add = [nn.Linear(C_channel + 1, C_channel), activation,
                     nn.Linear(C_channel, C_channel)]

        self.model_multi = nn.Sequential(*model_multi)
        self.model_add = nn.Sequential(*model_add)

    def forward(self, z, SNR):

        # Policy/gate network
        N, C, W, H = z.shape

        z_mean = torch.mean(z, (-2, -1))
        z_cat = torch.cat((z_mean, SNR), -1)

        factor = self.model_multi(z_cat).view(N, C, 1, 1)
        addition = self.model_add(z_cat).view(N, C, 1, 1)

        return z * factor + addition
