# Copyright (C) 2017 NVIDIA Corporation. All rights reserved.
# Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import time
import math
from models import create_model
from data import create_dataset
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
import util.util as util
from util.visualizer import Visualizer
import os
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import scipy.io as sio
import shutil
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
import math


# Set random seed
torch.manual_seed(0)
np.random.seed(0)

# Extract the options
opt = TestOptions().parse()


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
opt.C_channel = 16
opt.method = 'gumbel'
opt.temp = 3
opt.eta = 0.05
opt.lambda_reward = 0.5
opt.lambda_L2 = 200       # The weight for L2 loss
opt.selection = True
opt.N_input = 256
opt.N_options = 4
opt.is_noise = True
opt.constant = False
opt.how_many_channel = 1
opt.num_test = 10000
opt.is_test = True
opt.SNR = 20
opt.force_length = 0
opt.select = 'hard'
##############################################################################################################

opt.activation = 'sigmoid'    # The output activation function at the last layer in the decoder
opt.norm_EG = 'batch'


if opt.dataset_mode == 'CIFAR10':
    opt.dataroot = './data'
    opt.size = 32
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    dataset = torch.utils.data.DataLoader(testset, batch_size=1,
                                          shuffle=False, num_workers=2, drop_last=True)
    dataset_size = len(dataset)
    print('#training images = %d' % dataset_size)
else:
    raise Exception('Not implemented yet')

########################################  OFDM setting  ###########################################
# Display setting
opt.checkpoints_dir = './Checkpoints/' + opt.dataset_mode + '_dynamic'

if opt.selection:
    opt.name = 'C' + str(opt.C_channel) + '_method_' + opt.method + '_L2_' + str(opt.lambda_L2) + '_re_' + str(opt.lambda_reward) + '_noise_' + str(opt.is_noise) + '_' + str(opt.constant) + '_' + opt.select
else:
    opt.name = 'C' + str(opt.C_channel) + '_noise_' + str(opt.is_noise)

opt.display_env = opt.dataset_mode + opt.name

# Choose the neural network model
opt.model = 'DynaAWGN'

model = create_model(opt)      # create a model given opt.model and other options
model.setup(opt)               # regular setup: load and print networks; create schedulers
model.eval()

total_iters = 0                # the total number of training iterations

output_path = './Images/' + opt.dataset_mode + '_dyna/' + opt.name
if os.path.exists(output_path) == False:
    os.makedirs(output_path)
else:
    shutil.rmtree(output_path)
    os.makedirs(output_path)

# Train with the Discriminator
PSNR_list = []
SSIM_list = []
N_channel_list = []
count_list = [[], [], [], [], [], [], [], [], [], []]
PSNR_class_list = [[], [], [], [], [], [], [], [], [], []]
#usage = []

#index = np.load(f'SNR_{opt.SNR}_alpha_{opt.lambda_reward}.npy')
#PSNR_output = []
#SSIM_output = []

for i, data in enumerate(dataset):
    if i >= opt.num_test:  # only apply our model to opt.num_test images.
        break

    start_time = time.time()

    if opt.dataset_mode in ['CIFAR10', 'CIFAR100']:
        input = data[0]
    elif opt.dataset_mode == 'CelebA':
        input = data['data']

    model.set_input(input.repeat(opt.how_many_channel, 1, 1, 1))
    model.temp = 0.005
    model.forward()
    fake = model.fake
    hard_mask = model.hard_mask

    N_channel_list.append(hard_mask[0].sum().item())
    count_list[data[1].item()].append(hard_mask[0].sum().item())

    # Get the int8 generated images
    img_gen_numpy = fake.detach().cpu().float().numpy()
    img_gen_numpy = (np.transpose(img_gen_numpy, (0, 2, 3, 1)) + 1) / 2.0 * 255.0
    img_gen_int8 = img_gen_numpy.astype(np.uint8)

    origin_numpy = input.detach().cpu().float().numpy()
    origin_numpy = (np.transpose(origin_numpy, (0, 2, 3, 1)) + 1) / 2.0 * 255.0
    origin_int8 = origin_numpy.astype(np.uint8)

    diff = np.mean((np.float64(img_gen_int8) - np.float64(origin_int8))**2, (1, 2, 3))

    PSNR = 10 * np.log10((255**2) / diff)
    PSNR_list.append(np.mean(PSNR))

    PSNR_class_list[data[1].item()].append(PSNR)

    img_gen_tensor = torch.from_numpy(np.transpose(img_gen_int8, (0, 3, 1, 2))).float()
    origin_tensor = torch.from_numpy(np.transpose(origin_int8, (0, 3, 1, 2))).float()

    ssim_val = ssim(img_gen_tensor, origin_tensor.repeat(opt.how_many_channel, 1, 1, 1), data_range=255, size_average=False)  # return (N,)
    # ms_ssim_val = ms_ssim(img_gen_tensor,origin_tensor.repeat(opt.how_many_channel,1,1,1), data_range=255, size_average=False ) #(N,)
    SSIM_list.append(torch.mean(ssim_val).item())

    for j in range(opt.how_many_channel):
        # Save the first sampled image
        save_path = f'{output_path}/{i}_PSNR_{PSNR[j]:.3f}_SSIM_{ssim_val[j]:.3f}_C_{hard_mask[j].sum().item():.1f}_SNR_{model.snr[j].item()}dB.png'
        util.save_image(util.tensor2im(fake[j].unsqueeze(0)), save_path, aspect_ratio=1)

        save_path = f'{output_path}/{i}.png'
        util.save_image(util.tensor2im(input), save_path, aspect_ratio=1)

    if i % 100 == 0:
        print(i)

counts = [np.mean(count_list[i]) for i in range(10)]
PSNRs = [np.mean(np.hstack(PSNR_class_list[i])) for i in range(10)]

np.save(f'SNR_{opt.SNR}_alpha_{opt.lambda_reward}.npy', np.array(N_channel_list))

print(f'Mean PSNR: {np.mean(PSNR_list):.3f}')
print(f'Mean SSIM: {np.mean(SSIM_list):.3f}')
print(f'Mean Channel: {np.mean(N_channel_list):.3f}')
print(f"Counts: {*counts,}")
print(f"PSNRs: {*PSNRs,}")
