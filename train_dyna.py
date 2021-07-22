# Copyright (C) 2017 NVIDIA Corporation. All rights reserved.
# Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import time
import math
from models import create_model
from data import create_dataset
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
import util.util as util
from util.visualizer import Visualizer
import os
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import scipy.io as sio

# Set random seed
torch.manual_seed(2)
np.random.seed(0)

# Extract the options
opt = TrainOptions().parse()


# Set the input dataset
opt.dataset_mode = 'CIFAR10'   # Current dataset:  CIFAR10, CelebA


if opt.dataset_mode in ['CIFAR10', 'CIFAR100']:
    opt.n_layers_D = 3
    opt.label_smooth = 1          # Label smoothing factor (for lsgan and vanilla gan only)
    opt.n_downsample = 2          # Downsample times
    opt.n_blocks = 2              # Numebr of residual blocks
    opt.first_kernel = 5          # The filter size of the first convolutional layer in encoder
    opt.batchsize = 128
    opt.n_epochs = 200            # # of epochs without lr decay
    opt.n_epochs_decay = 200      # # of epochs with lr decay
    opt.lr_policy = 'linear'      # decay policy.  Availability:  see options/train_options.py
    opt.beta1 = 0.5               # parameter for ADAM
    opt.lr = 5e-4                 # Initial learning rate
    opt.image_size = 32

elif opt.dataset_mode == 'CelebA':
    opt.n_layers_D = 3
    opt.label_smooth = 1          # Label smoothing factor (for lsgan and vanilla gan only)
    opt.n_downsample = 3          # Downsample times
    opt.n_blocks = 2              # Numebr of residual blocks
    opt.first_kernel = 5          # The filter size of the first convolutional layer in encoder
    opt.batch_size = 64
    opt.n_epochs = 25             # # of epochs without lr decay
    opt.n_epochs_decay = 25       # # of epochs with lr decay
    opt.lr_policy = 'linear'      # decay policy.  Availability:  see options/train_options.py
    opt.beta1 = 0.5               # parameter for ADAM
    opt.lr = 5e-4
    opt.save_latest_freq = 20000
    opt.image_size = 64

elif opt.dataset_mode == 'OpenImage':
    opt.n_layers_D = 4
    opt.label_smooth = 1          # Label smoothing factor (for lsgan and vanilla gan only)
    opt.n_downsample = 3          # Downsample times
    opt.n_blocks = 4              # Numebr of residual blocks
    opt.first_kernel = 5          # The filter size of the first convolutional layer in encoder
    opt.batch_size = 16
    opt.n_epochs = 20             # # of epochs without lr decay
    opt.n_epochs_decay = 20       # # of epochs with lr decay
    opt.lr_policy = 'linear'      # decay policy.  Availability:  see options/train_options.py
    opt.beta1 = 0.5               # parameter for ADAM
    opt.lr = 5e-4


############################ Things recommanded to be changed ##########################################
# Set up the training procedure
opt.C_channel = 32
opt.SNR = 0
opt.is_infer = False
opt.method = 'st'
opt.temp = 3
opt.lambda_reward = 1
opt.lambda_soft_reward = 0
opt.lambda_L2 = 256       # The weight for L2 loss
opt.selection = False

##############################################################################################################

opt.activation = 'sigmoid'    # The output activation function at the last layer in the decoder
opt.norm_EG = 'batch'


if opt.dataset_mode == 'CIFAR10':
    opt.dataroot = './data'
    opt.size = 32
    transform = transforms.Compose(
        [transforms.RandomHorizontalFlip(p=0.5),
         transforms.RandomCrop(opt.size, padding=5, pad_if_needed=True, fill=0, padding_mode='reflect'),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    dataset = torch.utils.data.DataLoader(trainset, batch_size=opt.batchsize,
                                          shuffle=True, num_workers=2, drop_last=True)
    dataset_size = len(dataset)
    print('#training images = %d' % dataset_size)

elif opt.dataset_mode == 'CIFAR100':
    opt.dataroot = './data'
    opt.size = 32
    transform = transforms.Compose(
        [transforms.RandomHorizontalFlip(p=0.5),
         transforms.RandomCrop(opt.size, padding=5, pad_if_needed=True, fill=0, padding_mode='reflect'),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                             download=True, transform=transform)
    dataset = torch.utils.data.DataLoader(trainset, batch_size=opt.batchsize,
                                          shuffle=True, num_workers=2, drop_last=True)
    dataset_size = len(dataset)
    print('#training images = %d' % dataset_size)

elif opt.dataset_mode == 'CelebA':
    opt.dataroot = './data/celeba/CelebA_train'
    opt.load_size = 80
    opt.crop_size = 64
    opt.size = 64
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)
    print('#training images = %d' % dataset_size)

elif opt.dataset_mode == 'OpenImage':
    opt.dataroot = './data/opv6'
    opt.size = 256
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)
    print('#training images = %d' % dataset_size)

else:
    raise Exception('Not implemented yet')


########################################  OFDM setting  ###########################################
# Display setting
opt.checkpoints_dir = './Checkpoints/' + opt.dataset_mode + '_dynamic'

if opt.selection:
    opt.name = 'C' + str(opt.C_channel) + '_SNR_' + str(opt.SNR) + '_method_' + opt.method + '_L2_' + str(opt.lambda_L2)
else:
    opt.name = 'C' + str(opt.C_channel) + '_SNR_' + str(opt.SNR)

opt.display_env = opt.dataset_mode + opt.name

# Choose the neural network model
opt.model = 'DynaAWGN'

model = create_model(opt)      # create a model given opt.model and other options
model.setup(opt)               # regular setup: load and print networks; create schedulers
visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
total_iters = 0                # the total number of training iterations

# Train with the Discriminator
loss_D_list = []
loss_G_list = []
count = 0
temp = opt.temp
eps = 1e-3

eta = 0.03
n_epochs_warmup = 50
n_epochs_joint = 150
n_epochs_fine = 100

lr_warmup = 1e-3
lr_joint = 1e-4
lr_fine = 1e-5

total_epoch = n_epochs_warmup + n_epochs_joint + n_epochs_fine
gap = (opt.temp - eps) / opt.n_epochs


# Setupt the warmup stage
print('Warm up stage begins!')
model.optimizer_G.param_groups[0]['lr'] = lr_warmup
print(f'Learning rate changed to {lr_warmup}')
# Dsiable the update for the policy network
for param in model.netP.parameters():
    param.requires_grad = False
print('Policy Network Disabled!')

for epoch in range(opt.epoch_count, total_epoch + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
    epoch_start_time = time.time()  # timer for entire epoch
    iter_data_time = time.time()    # timer for data loading per iteration
    epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
    visualizer.reset()              # reset the visualizer: make sure it saves the results to HTML at least once every epoch

    # Update temperature
    model.temp = max(opt.temp * math.exp(-eta * (epoch - 1)), 0.01)
    print(f'Update temperature to {model.temp}')

    # Setup the joint training stage
    if epoch == n_epochs_warmup + 1:
        print('Joint learning stage begins!')
        model.optimizer_G.param_groups[0]['lr'] = lr_joint
        print(f'Learning rate changed to {lr_joint}')
        for param in model.netP.parameters():
            param.requires_grad = True
        print('Policy Network Enabled!')

    if epoch == n_epochs_warmup + n_epochs_joint + 1:
        print('Fine-tuning stage begins!')
        model.optimizer_G.param_groups[0]['lr'] = lr_fine
        print(f'Learning rate changed to {lr_fine}')
        for param in model.netP.parameters():
            param.requires_grad = False
        print('Policy Network Disabled!')
        #opt.is_infer = True

    for i, data in enumerate(dataset):  # inner loop within one epoch
        iter_start_time = time.time()  # timer for computation per iteration
        if total_iters % opt.print_freq == 0:
            t_data = iter_start_time - iter_data_time

        total_iters += opt.batch_size
        epoch_iter += opt.batch_size

        if opt.dataset_mode in ['CIFAR10', 'CIFAR100']:
            input = data[0]
        elif opt.dataset_mode == 'CelebA':
            input = data['data']
        elif opt.dataset_mode == 'OpenImage':
            input = data['data']

        model.set_input(input)         # unpack data from dataset and apply preprocessing
        model.optimize_parameters()   # calculate loss functions, get gradients, update network weights
        #count += 1
        # model.update_temp(count)

        if total_iters % opt.display_freq == 0:   # display images on visdom and save images to a HTML file
            save_result = total_iters % opt.update_html_freq == 0
            model.compute_visuals()
            visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

        if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
            losses = model.get_current_losses()
            t_comp = (time.time() - iter_start_time) / opt.batch_size
            visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
            if opt.display_id > 0:
                visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)
            #sum_P, sum_E, sum_G = model.cal_weight_L2()
            #print(f'P:{sum_P.item():.6f}, E:{sum_E.item():.6f}, G:{sum_G.item():.6f}')
        if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
            print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
            save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
            model.save_networks(save_suffix)
        iter_data_time = time.time()

    if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
        print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
        model.save_networks('latest')
        model.save_networks(epoch)

    print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
    # model.update_learning_rate()
