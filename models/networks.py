# Copyright (C) 2017 NVIDIA Corporation. All rights reserved.
# Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.nn import functional as F



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

class Flatten(nn.Module):
    def forward(self, x):
        N, C, H, W = x.size()  # read in N, C, H, W
        return x.view(N, -1)  # "flatten" the C * H * W values into a single vector per image


class Normalize(nn.Module):
    def forward(self, x, power):
        N = x.shape[0]
        pwr = torch.mean(x**2, (1, 2, 3), True)
        return np.sqrt(power) * x / torch.sqrt(pwr)

def print_network(net):
    if isinstance(net, list):
        net = net[0]
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)

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

########################################################

def define_dynaG(output_nc, ngf, max_ngf, n_downsample, C_channel, n_blocks, norm='instance', init_type='kaiming', init_gain=0.02, gpu_ids=[], activation='sigmoid'):
    net = None
    norm_layer = get_norm_layer(norm_type=norm)
    net = Generator_dyna(output_nc=output_nc, ngf=ngf, max_ngf=max_ngf, C_channel=C_channel, n_blocks=n_blocks, n_downsampling=n_downsample, norm_layer=norm_layer, padding_type="reflect")
    return init_net(net, init_type, init_gain, gpu_ids)

def define_dynaP(ngf, max_ngf, n_downsample, init_type='kaiming', init_gain=0.02, gpu_ids=[], N_output=7):
    net = None
    net = Policy_dyna(ngf=ngf, max_ngf=max_ngf, n_downsampling=n_downsample, N_output=N_output)
    return init_net(net, 'normal', 0.002, gpu_ids)

def define_SE(input_nc, ngf, max_ngf, n_downsample, norm='instance', init_type='normal', init_gain=0.02, gpu_ids=[]):
    net = None
    norm_layer = get_norm_layer(norm_type=norm)
    net = Source_Encoder(input_nc=input_nc, ngf=ngf, max_ngf=max_ngf, n_downsampling=n_downsample, norm_layer=norm_layer)
    return init_net(net, init_type, init_gain, gpu_ids)

def define_CE(ngf, max_ngf, n_downsample, C_channel,norm='instance', init_type='normal', init_gain=0.02, gpu_ids=[]):
    net = None
    norm_layer = get_norm_layer(norm_type=norm)
    net = Channel_Encoder(ngf=ngf, max_ngf=max_ngf, C_channel=C_channel, n_downsampling=n_downsample, norm_layer=norm_layer)
    return init_net(net, init_type, init_gain, gpu_ids)


class Source_Encoder(nn.Module):

    def __init__(self, input_nc, ngf=64, max_ngf=512, n_downsampling=2, norm_layer=nn.BatchNorm2d):

        assert(n_downsampling >= 0)
        super(Source_Encoder, self).__init__()

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        activation = nn.ReLU(True)

        # Downscale network
        model = [nn.ReflectionPad2d((7 - 1) // 2),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 activation]
        # add downsampling layers
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(min(ngf * mult, max_ngf), min(ngf * mult * 2, max_ngf), kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(min(ngf * mult * 2, max_ngf)), activation]
        self.net = nn.Sequential(*model)

    def forward(self, input):
        return self.net(input)


class Channel_Encoder(nn.Module):

    def __init__(self, ngf=64, max_ngf=512, C_channel=16, n_downsampling=2, norm_layer=nn.BatchNorm2d):

        assert(n_downsampling >= 0)
        super(Channel_Encoder, self).__init__()

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        activation = nn.ReLU(True)

        # Resnet
        mult = 2 ** n_downsampling
        self.res1 = ResnetBlock(min(ngf * mult, max_ngf), padding_type='zero', norm_layer=norm_layer, use_dropout=False, use_bias=use_bias)
        self.res2 = ResnetBlock(min(ngf * mult, max_ngf), padding_type='zero', norm_layer=norm_layer, use_dropout=False, use_bias=use_bias)
        self.mod1 = modulation(min(ngf * mult, max_ngf))
        self.mod2 = modulation(min(ngf * mult, max_ngf))
        self.projection = nn.Conv2d(min(ngf * mult, max_ngf), C_channel, kernel_size=3, padding=1, stride=1)

    def forward(self, z, SNR):
        z = self.mod1(self.res1(z), SNR)
        z = self.mod2(self.res2(z), SNR)
        latent = self.projection(z)
        return latent

class Generator_dyna(nn.Module):
    def __init__(self, output_nc, ngf=64, max_ngf=512, C_channel=16, n_blocks=2, n_downsampling=2, norm_layer=nn.BatchNorm2d, padding_type="reflect"):
        assert (n_blocks >= 0)
        assert(n_downsampling >= 0)

        super(Generator_dyna, self).__init__()

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

        model += [nn.ReflectionPad2d((5 - 1) // 2), nn.Conv2d(ngf, output_nc, kernel_size=5, padding=0)]
        model += [nn.Sigmoid()]

        self.model = nn.Sequential(*model)

    def forward(self, input, SNR):
        z = self.mask_conv(input)
        z = self.mod1(self.res1(z), SNR)
        z = self.mod2(self.res2(z), SNR)
        return 2 * self.model(z) - 1


class Policy_dyna(nn.Module):
    def __init__(self, ngf=64, max_ngf=256, N_output=7, n_downsampling=2):

        super(Policy_dyna, self).__init__()

        activation = nn.ReLU(True)

        mult = 2 ** n_downsampling
        ngf_dim = min(ngf * mult, max_ngf)

        # Policy network
        model = [nn.Linear(ngf_dim + 1, 64), activation, nn.BatchNorm1d(64),
                 nn.Linear(64, 64), activation, nn.BatchNorm1d(64),
                 nn.Linear(64, N_output)]
        self.model_gate = nn.Sequential(*model)

    def forward(self, z, SNR, temp=5):

        # Policy/gate network
        N, C, W, H = z.shape
        z = self.model_gate(torch.cat((z.mean((-2, -1)), SNR), -1))      # N x C+1

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
    # 2. Accumulate sum
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

    