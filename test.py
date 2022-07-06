import argparse
import pickle

from pkg_resources import require
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils import data
from torchvision.datasets import ImageFolder

# from data_set import FDCDataset

import numpy as np
import random
import time
import os
from tqdm import tqdm
from copy import deepcopy
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

from training.model import Generator, LocalPathway, Encoder

random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

def image_grid(array, ncols=8):
    index, height, width, channels = array.shape
    nrows = index//ncols    
    img_grid = (array.reshape(nrows, ncols, height, width, channels)
              .swapaxes(1,2)
              .reshape(height*nrows, width*ncols))
        
    return img_grid

def tensor2image(im_tensor):
    '''
    Input tensor is a detached tensor
    '''
    im_np = im_tensor.data[0].cpu().numpy()
    channel, _, _ = im_np.shape

    if channel == 1:
        new_im_np = im_np[0,:,:]
    elif channel == 3:
        new_im_np = np.transpose(im_np, (1,2,0))
    elif channel == 64:
        new_im_np = im_np[..., np.newaxis]
        new_im_np = image_grid(new_im_np)
    else:
        raise NotImplementedError(f'Not yet supported for channel {channel}')
    
    return new_im_np

if __name__ == "__main__":
    device = 'cuda'
    parser = argparse.ArgumentParser()

    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--img_path", type=str, required=True)
    parser.add_argument("--save_name", type=str, required=True)

    parser.add_argument("--test_suffix", type=str, default='test_global_rgb')
    parser.add_argument("--num_workers", type=int, default=10)
    parser.add_argument("--save_image_dir", type=str, default='expr')
    parser.add_argument("--mode", type=str, default='test')
    parser.add_argument("--input_name", type=str, default='overlaid.png')
    parser.add_argument("--gt_name", type=str, default='gt.png')
    parser.add_argument("--image_size", type=int, default=256)
    args = parser.parse_args()

    ngpus = torch.cuda.device_count()
    print('{} GPUs!'.format(ngpus))

    is_ae = True
    is_gray = False
    is_global = True
    
    inference_model = None

    ckpt = torch.load(args.ckpt)

    if is_ae:
        train_args = ckpt['train_args']
        train_cfg = ckpt['cfg']
        if is_global:
            if is_gray:
                global_gray_ae = LocalPathway(is_gray, use_batchnorm=train_cfg.TRAIN.GRAY2RGB_USE_BATCHNORM)
                global_gray_ae.eval()
                global_gray_ae.to(device)
            else:
                global_rgb_ae = LocalPathway(is_gray, use_batchnorm=train_cfg.TRAIN.GRAY2RGB_USE_BATCHNORM)
                global_rgb_ae.eval()
                global_rgb_ae.to(device)
        else:
            pass
    else:
        train_args = ckpt['train_args']
        generator = Generator(
            train_args.size,
            train_args.mapping_layer_num,
            train_args.latent_channel_size,
            train_args.latent_spatial_size,
            lr_mul = train_args.lr_mul,
            channel_multiplier=train_args.channel_multiplier,
            normalize_mode=train_args.normalize_mode,
            small_generator=train_args.small_generator,
        )

        encoder = Encoder(
            train_args.size,
            train_args.latent_channel_size,
            train_args.latent_spatial_size,
            channel_multiplier=train_args.channel_multiplier,
        )

        generator.eval()
        encoder.eval()
        generator = generator.to(device)
        encoder = encoder.to(device)
    
    if is_ae:
        if is_gray:
            global_gray_ae.load_state_dict(ckpt["global_gray_ae"])
        else:
            global_rgb_ae.load_state_dict(ckpt["global_rgb_ae"])
    else:
        generator.load_state_dict(ckpt["generator"])
        encoder.load_state_dict(ckpt["encoder"])
    
    input_img = Image.open(os.path.join(args.img_path, args.input_name))
    input_img = input_img.convert('RGB')
    input_img = input_img.resize((args.image_size, args.image_size), Image.BICUBIC)
    input_img = np.asarray(input_img, np.float32)

    gt_img = Image.open(os.path.join(args.img_path, args.gt_name))
    gt_img = gt_img.convert('RGB')
    gt_img = gt_img.resize((args.image_size, args.image_size), Image.BICUBIC)
    gt_img = np.asarray(gt_img, np.float32)

    mean_rgb = (128, 128, 128)
    input_img -= mean_rgb
    input_img = input_img/128.
    input_img = input_img.transpose((2,0,1))

    gt_img -= mean_rgb
    gt_img = gt_img/128.
    gt_img = gt_img.transpose((2,0,1))

    input_img = torch.tensor(input_img)
    input_img = input_img.unsqueeze(0)
    input_img = input_img.to(device)

    gt_img = torch.tensor(gt_img)
    gt_img = gt_img.unsqueeze(0)
    gt_img = gt_img.to(device)

    if is_ae:
        if is_global:
            if is_gray:
                fake_gray_global_img, fake_gray_global_feature = global_gray_ae(input_img)
                fake_img = fake_gray_global_img
            else:
                fake_rgb_global_img, fake_rgb_global_feature = global_rgb_ae(input_img)
                fake_img = fake_rgb_global_img
        else:
            pass
    else:
        fake_stylecode = encoder(input_img)
        fake_img, _ = generator(fake_stylecode, input_is_stylecode=True)

    temp_input = input_img.detach()
    vis_input = tensor2image(temp_input)
    vis_input = (vis_input+1)/2.

    temp_fake = fake_img.detach()
    vis_fake = tensor2image(temp_fake)
    vis_fake = (vis_fake+1)/2.

    temp_gt = gt_img.detach()
    vis_gt = tensor2image(temp_gt)
    vis_gt = (vis_gt+1)/2.

    nrow = 1
    ncol = 3
    fig = plt.figure(figsize=(10, 10))
    gs = gridspec.GridSpec(nrow, ncol,
                                    wspace=0.1, hspace=0.0, 
                                    top=1.-0.5/(nrow+1), bottom=0.5/(nrow+1), 
                                    left=0.5/(ncol+1), right=1-0.5/(ncol+1))
    ax1 = plt.subplot(gs[0, 0])
    ax1.imshow(vis_input)
    ax1.title.set_text('Input Image')
    ax1.set_xticklabels([])
    ax1.set_yticklabels([])
    ax1.axis('off')

    ax2 = plt.subplot(gs[0, 1])
    ax2.imshow(vis_fake)
    ax2.title.set_text('Fake Image')
    ax2.set_xticklabels([])
    ax2.set_yticklabels([])
    ax2.axis('off')

    ax3 = plt.subplot(gs[0, 2])
    ax3.imshow(vis_gt)
    ax3.title.set_text('GT Image')
    ax3.set_xticklabels([])
    ax3.set_yticklabels([])
    ax3.axis('off')

    test_save_path = os.path.join(args.save_image_dir, args.test_suffix)
    if not os.path.exists(test_save_path):
        os.makedirs(test_save_path)
    img_save_path = os.path.join(test_save_path, args.save_name)

    print('Saving results to ... ', )
    fig.savefig(img_save_path)
    print('Image is saved.')

