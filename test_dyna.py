# Copyright (C) 2017 NVIDIA Corporation. All rights reserved.
# Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import time
from models import create_model
from options.test_options import TestOptions
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms


# Extract the options
opt = TestOptions().parse()

# Prepare the dataset   
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
trainset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
dataset = torch.utils.data.DataLoader(trainset, batch_size=1,
                                        shuffle=False, num_workers=2, drop_last=True)
dataset_size = len(dataset)
print('#test images = %d' % dataset_size)

opt.name = 'C' + str(opt.C_channel) + '_L2_' + str(opt.lambda_L2) + '_re_' + str(opt.lambda_reward) + '_' + opt.select 

model = create_model(opt)      # create a model given opt.model and other options
model.setup(opt)               # regular setup: load and print networks; create schedulers
model.eval()

PSNR_list = []
SSIM_list = []
N_channel_list = []
count_list = [[]]*10
PSNR_class_list = [[]]*10

for i, data in enumerate(dataset):
    if i >= opt.num_test:  # only apply our model to opt.num_test images.
        break
    start_time = time.time()
    input = data[0]
    model.set_input(input.repeat(opt.num_test_channel, 1, 1, 1))
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

    if i % 100 == 0:
        print(i)

counts = [np.mean(count_list[i]) for i in range(10)]
PSNRs = [np.mean(np.hstack(PSNR_class_list[i])) for i in range(10)]

print(f'Mean PSNR: {np.mean(PSNR_list):.3f}')
print(f'Mean SSIM: {np.mean(SSIM_list):.3f}')
print(f'Mean Channel: {np.mean(N_channel_list):.3f}')
print(f"Counts: {*counts,}")
print(f"PSNRs: {*PSNRs,}")
