import argparse
import pickle
from selectors import EpollSelector

from pkg_resources import require
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils import data
from torchvision.datasets import ImageFolder
from skimage.metrics import structural_similarity

from data_set.fdc import FDCDataset

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

from training.model import Generator, GlobalPathway_2, LocalPathway, Encoder, GlobalPathway_1, LocalFuser, GlobalPathway
from training.model import global_fuser

random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

random_seed = 1234

def init_fn(worker_id):
    np.random.seed(random_seed + worker_id)

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
    if len(im_tensor.shape) == 4:
        im_np = im_tensor.data[0].cpu().numpy()
    else:
        im_np = im_tensor.data.cpu().numpy()

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
    parser.add_argument("--ckpt_global", type=str) # Provide the trained global model
    # parser.add_argument("--img_path", type=str, required=True)
    # parser.add_argument("--local_img_path", type=str) # Provide different local patches
    # parser.add_argument("--save_name", type=str, required=True)
    parser.add_argument("--list_path", type=str, required=True)
    parser.add_argument("--suffix", type=str, default='debugging')
    # parser.add_argument("--num_workers", type=int, default=10)
    parser.add_argument("--test_mode", type=str, default='weak')

    parser.add_argument("--save_image_dir", type=str, default='/nfs/STG/Audio2FacePose/MEAD/Experiments/StyleMapGAN/expr')
    parser.add_argument("--mode", type=str, default='test')
    parser.add_argument("--input_name", type=str, default='overlaid.png')
    parser.add_argument("--gt_name", type=str, default='gt.png')
    parser.add_argument("--ref_name", type=str, default='reference.png')
    parser.add_argument("--local_left_name", type=str, default='left_patch_gray_angle.png')
    parser.add_argument("--local_right_name", type=str, default='right_patch_gray_angle.png')
    parser.add_argument("--local_mouth_name", type=str, default='mouth_patch_gray_angle.png')
    parser.add_argument("--mask_name", type=str, default='mask.npy')
    parser.add_argument("--left_gt_name", type=str, default='left_patch_rgb_gt.png')
    parser.add_argument("--right_gt_name", type=str, default='right_patch_rgb_gt.png')
    parser.add_argument("--mouth_gt_name", type=str, default='mouth_patch_rgb_gt.png')
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=1)
    args = parser.parse_args()

    # args.local_img_path = args.img_path
    
    args.suffix += ('_'+args.test_mode)

    ngpus = torch.cuda.device_count()
    print('{} GPUs!'.format(ngpus))

    is_ae = True
    is_gray = False # Can be deprecated
    is_global = True # True only for entangled method
    is_single_channel = True # Only for local method
    is_equal = True # Only for global method; True: GlobalPathway; False: GlobalPathway_2
    is_temp = False

    inference_model = None

    ckpt = torch.load(args.ckpt)

    if args.ckpt_global:
        ckpt_global = torch.load(args.ckpt_global)

    if is_ae:
        train_args = ckpt['train_args']
        train_cfg = ckpt['cfg']
        if is_global:
            if is_gray:
                global_gray_ae = LocalPathway(is_gray, use_batchnorm=train_cfg.TRAIN.GRAY2RGB_USE_BATCHNORM)
                global_gray_ae.eval()
                global_gray_ae.to(device)
            else:
                if is_equal:
                    global_rgb_ae = GlobalPathway(use_batchnorm=train_cfg.TRAIN.GRAY2RGB_USE_BATCHNORM)
                else:
                    global_rgb_ae = GlobalPathway_2(use_batchnorm=train_cfg.TRAIN.GRAY2RGB_USE_BATCHNORM)
                global_rgb_ae.eval()
                global_rgb_ae.to(device)
        else:
            local_pathway_left_eye = LocalPathway(is_gray=True, use_batchnorm=train_cfg.TRAIN.GRAY2RGB_USE_BATCHNORM)
            local_pathway_left_eye.eval()
            local_pathway_left_eye.to(device)
            local_pathway_right_eye = LocalPathway(is_gray=True, use_batchnorm=train_cfg.TRAIN.GRAY2RGB_USE_BATCHNORM)
            local_pathway_right_eye.eval()
            local_pathway_right_eye.to(device)
            local_pathway_mouth = LocalPathway(is_gray=True, use_batchnorm=train_cfg.TRAIN.GRAY2RGB_USE_BATCHNORM)
            local_pathway_mouth.eval()
            local_pathway_mouth.to(device)

            # Use the global method
            global_rgb_ae = GlobalPathway(use_batchnorm=train_cfg.TRAIN.GRAY2RGB_USE_BATCHNORM)
            global_rgb_ae.eval()
            global_rgb_ae.to(device)

            # # Original global fusion for local method
            # local_fuser = LocalFuser(is_single_channel=is_single_channel)
            # local_fuser.eval()
            # local_fuser.to(device)
            # global_ae = GlobalPathway_1(is_gray=True, is_single_channel=is_single_channel, use_batchnorm=train_cfg.TRAIN.GRAY2RGB_USE_BATCHNORM)
            # global_ae.eval()
            # global_ae.to(device)
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
        if is_global:
            if is_gray:
                global_gray_ae.load_state_dict(ckpt["global_gray_ae"])
            else:
                global_rgb_ae.load_state_dict(ckpt["global_rgb_ae"])
        else:
            local_pathway_left_eye.load_state_dict(ckpt["local_pathway_left_eye"])
            local_pathway_right_eye.load_state_dict(ckpt["local_pathway_right_eye"])
            local_pathway_mouth.load_state_dict(ckpt["local_pathway_mouth"])

            # # Original global model for local method
            # global_ae.load_state_dict(ckpt["global_ae"])

            global_rgb_ae.load_state_dict(ckpt_global["global_rgb_ae"])
    else:
        generator.load_state_dict(ckpt["generator"])
        encoder.load_state_dict(ckpt["encoder"])
    
    test_save_path = os.path.join(args.save_image_dir, args.suffix)
    if not os.path.exists(test_save_path):
        os.makedirs(test_save_path)

    if args.test_mode == 'weak':
        test_dataset = FDCDataset(is_global, is_equal, is_temp, list_path=args.list_path, local_transforms=None, gray_ref_transforms=None, set='weak_test_small')
    elif args.test_mode == 'strong':
        test_dataset = FDCDataset(is_global, is_equal, is_temp, list_path=args.list_path, local_transforms=None, gray_ref_transforms=None, set='strong_test_small')

    test_loader = data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        worker_init_fn=init_fn
    )
    test_iter = enumerate(test_loader)

    mse_in_list = []
    mse_out_list = []
    mse_list = []
    ssim_list = []
    psnr_list = []
    lpips_list = []

    for i_iter, batch in tqdm(test_iter):
        if is_global:
            if is_temp:
                raise NotImplementedError('No temp data for test')
            else:
                ref_rgb, overlaid_rgb, gt_rgb, mask = batch
                ref_rgb = ref_rgb.to(device)
                overlaid_rgb = overlaid_rgb.to(device)
                gt_rgb = gt_rgb.to(device)
                if overlaid_rgb.shape[0] != args.batch_size:
                    print(f'Data batch wrong---{overlaid_rgb.shape[0]}, continue...')
                    continue
        else:
            overlaid_rgb, ref_rgb, ref_gray, gt_rgb, gt_gray, left_gt, left_raw, right_gt, right_raw, mouth_gt, mouth_raw, mask = batch
            overlaid_rgb = overlaid_rgb.to(device)
            ref_rgb = ref_rgb.to(device)
            ref_gray = ref_gray.to(device)
            gt_rgb = gt_rgb.to(device)
            gt_gray = gt_gray.to(device)
            left_gt = left_gt.to(device)
            left_raw = left_raw.to(device)
            right_gt = right_gt.to(device)
            right_raw = right_raw.to(device)
            mouth_gt = mouth_gt.to(device)
            mouth_raw = mouth_raw.to(device)
            mask = mask.to(device)
            if ref_rgb.shape[0] != args.batch_size:
                print(f'Data batch wrong---{ref_rgb.shape[0]}, continue...')
                continue

    # input_img = Image.open(os.path.join(args.img_path, args.input_name))
    # input_img = input_img.convert('RGB')
    # input_img = input_img.resize((args.image_size, args.image_size), Image.BICUBIC)
    # input_img = np.asarray(input_img, np.float32)

    # ref_img = Image.open(os.path.join(args.img_path, args.ref_name))
    # ref_gray_img = ref_img.convert('L')

    # ref_img = ref_img.convert('RGB')
    # ref_img = ref_img.resize((args.image_size, args.image_size), Image.BICUBIC)
    # ref_img = np.asarray(ref_img, np.float32)

    # ref_gray_img = ref_gray_img.resize((args.image_size, args.image_size), Image.BICUBIC)
    # ref_gray_np = np.asarray(ref_gray_img, np.float32)
    # ref_gray_np = ref_gray_np[..., np.newaxis]

    # gt_img = Image.open(os.path.join(args.img_path, args.gt_name))
    # gt_img = gt_img.convert('RGB')
    # gt_img = gt_img.resize((args.image_size, args.image_size), Image.BICUBIC)
    # gt_img = np.asarray(gt_img, np.float32)
    
    # local_left_img = Image.open(os.path.join(args.local_img_path, args.local_left_name))
    # local_left_img = local_left_img.convert('L')
    # local_left_img = local_left_img.resize((args.image_size, args.image_size), Image.BICUBIC)
    # local_left_np = np.asarray(local_left_img, np.float32)
    # local_left_np = local_left_np[..., np.newaxis]

    # left_gt_img = Image.open(os.path.join(args.local_img_path, args.left_gt_name))
    # left_gt_img = left_gt_img.convert('L')
    # left_gt_img = left_gt_img.resize((args.image_size, args.image_size), Image.BICUBIC)
    # left_gt_np = np.asarray(left_gt_img, np.float32)
    # left_gt_np = left_gt_np[..., np.newaxis]

    # local_right_img = Image.open(os.path.join(args.local_img_path, args.local_right_name))
    # local_right_img = local_right_img.convert('L')
    # local_right_img = local_right_img.resize((args.image_size, args.image_size), Image.BICUBIC)
    # local_right_np = np.asarray(local_right_img, np.float32)
    # local_right_np = local_right_np[..., np.newaxis]

    # right_gt_img = Image.open(os.path.join(args.local_img_path, args.right_gt_name))
    # right_gt_img = right_gt_img.convert('L')
    # right_gt_img = right_gt_img.resize((args.image_size, args.image_size), Image.BICUBIC)
    # right_gt_np = np.asarray(right_gt_img, np.float32)
    # right_gt_np = right_gt_np[..., np.newaxis]

    # local_mouth_img = Image.open(os.path.join(args.local_img_path, args.local_mouth_name))
    # local_mouth_img = local_mouth_img.convert('L')
    # local_mouth_img = local_mouth_img.resize((args.image_size, args.image_size), Image.BICUBIC)
    # local_mouth_np = np.asarray(local_mouth_img, np.float32)
    # local_mouth_np = local_mouth_np[..., np.newaxis]
    
    # mouth_gt_img = Image.open(os.path.join(args.local_img_path, args.mouth_gt_name))
    # mouth_gt_img = mouth_gt_img.convert('L')
    # mouth_gt_img = mouth_gt_img.resize((args.image_size, args.image_size), Image.BICUBIC)
    # mouth_gt_np = np.asarray(mouth_gt_img, np.float32)
    # mouth_gt_np = mouth_gt_np[..., np.newaxis]

    # mask = np.load(os.path.join(args.img_path, args.mask_name))
    # mask_im = Image.fromarray(np.uint8(mask))
    # resized_mask_im = mask_im.resize((args.image_size, args.image_size), Image.NEAREST)
    # mask  = np.asarray(resized_mask_im)

    # mean_rgb = (128, 128, 128)
    # mean_gray = 128

    # input_img -= mean_rgb
    # input_img = input_img/128.
    # input_img = input_img.transpose((2,0,1))

    # ref_img -= mean_rgb
    # ref_img = ref_img/128.
    # ref_img = ref_img.transpose((2,0,1))

    # ref_gray_np -= mean_gray
    # ref_gray_np /= 128.
    # ref_gray_np = ref_gray_np.transpose((2,0,1))

    # gt_img -= mean_rgb
    # gt_img = gt_img/128.
    # gt_img = gt_img.transpose((2,0,1))
    
    # local_left_np -= mean_gray
    # local_left_np /= 128.
    # local_left_np = local_left_np.transpose((2,0,1))

    # left_gt_np -= mean_gray
    # left_gt_np /= 128.
    # left_gt_np = left_gt_np.transpose((2,0,1))

    # local_right_np -= mean_gray
    # local_right_np /= 128.
    # local_right_np = local_right_np.transpose((2,0,1))

    # right_gt_np -= mean_gray
    # right_gt_np /= 128.
    # right_gt_np = right_gt_np.transpose((2,0,1))

    # local_mouth_np -= mean_gray
    # local_mouth_np /= 128.
    # local_mouth_np = local_mouth_np.transpose((2,0,1))

    # mouth_gt_np -= mean_gray
    # mouth_gt_np /= 128.
    # mouth_gt_np = mouth_gt_np.transpose((2,0,1))

    # input_img = torch.tensor(input_img)
    # input_img = input_img.unsqueeze(0)
    # input_img = input_img.to(device)

    # ref_img = torch.tensor(ref_img)
    # ref_img = ref_img.unsqueeze(0)
    # ref_img = ref_img.to(device)
    
    # ref_gray = torch.tensor(ref_gray_np)
    # ref_gray = ref_gray.unsqueeze(0)
    # ref_gray = ref_gray.to(device)

    # gt_img = torch.tensor(gt_img)
    # gt_img = gt_img.unsqueeze(0)
    # gt_img = gt_img.to(device)

    # local_left = torch.tensor(local_left_np)
    # local_left = local_left.unsqueeze(0)
    # local_left = local_left.to(device)

    # left_gt = torch.tensor(left_gt_np)
    # left_gt = left_gt.unsqueeze(0)
    # left_gt = left_gt.to(device)

    # local_right = torch.tensor(local_right_np)
    # local_right = local_right.unsqueeze(0)
    # local_right = local_right.to(device)

    # right_gt = torch.tensor(right_gt_np)
    # right_gt = right_gt.unsqueeze(0)
    # right_gt = right_gt.to(device)

    # local_mouth = torch.tensor(local_mouth_np)
    # local_mouth = local_mouth.unsqueeze(0)
    # local_mouth = local_mouth.to(device)

    # mouth_gt = torch.tensor(mouth_gt_np)
    # mouth_gt = mouth_gt.unsqueeze(0)
    # mouth_gt = mouth_gt.to(device)

    # mask = torch.tensor(mask)
    # mask = mask.unsqueeze(0)
    # mask = mask.to(device)

        if is_ae:
            if is_global:
                if is_gray:
                    raise NotImplementedError('Gray is not supported any more.')
                else:
                    if is_equal:
                        fake_rgb_global_img, fake_rgb_global_feature = global_rgb_ae(overlaid_rgb)
                        fake_img = fake_rgb_global_img
                    else:
                        fake_rgb_global_img, fake_rgb_global_feature = global_rgb_ae(overlaid_rgb, ref_rgb)
                        fake_img = fake_rgb_global_img
                mask = mask.detach()
                mask = tensor2image(mask)
            else:
                local_left_pred, _ = local_pathway_left_eye(left_raw)
                local_right_pred, _ = local_pathway_right_eye(right_raw)
                local_mouth_pred, _ = local_pathway_mouth(mouth_raw)
                
                temp_left_pred = local_left_pred.detach()
                vis_left_pred = tensor2image(temp_left_pred)
                vis_left_pred = (vis_left_pred+1)/2.

                temp_right_pred = local_right_pred.detach()
                vis_right_pred = tensor2image(temp_right_pred)
                vis_right_pred = (vis_right_pred+1)/2.

                temp_mouth_pred = local_mouth_pred.detach()
                vis_mouth_pred = tensor2image(temp_mouth_pred)
                vis_mouth_pred = (vis_mouth_pred+1)/2.
                
                temp_gt = gt_rgb.detach()
                vis_gt = tensor2image(temp_gt)

                mask = mask.detach()
                mask = tensor2image(mask)

                # global fusion from global method
                global_input = global_fuser(vis_left_pred, vis_right_pred, vis_mouth_pred, vis_gt, mask)
                
                global_input_np = deepcopy(global_input)
                global_input_np = np.transpose(global_input_np, (1,2,0)) # convert from torch format to numpy format

                global_input = torch.tensor(global_input)
                global_input = global_input.unsqueeze(0)
                global_input = global_input.to(device)

                fake_rgb_global_img, fake_rgb_global_feature = global_rgb_ae(global_input)
                fake_img = fake_rgb_global_img

                # # Original global fusion for local method
                # local_fused = local_fuser(local_left_pred, local_right_pred, local_mouth_pred, mask)
                # # local_fused = local_fuser(left_gt, right_gt, mouth_gt, ref_gray, mask)

                # # global_input = torch.cat((local_fused, ref_gray), dim=1)
                # print('Max value in local fused: ', torch.max(local_fused))
                # print('Min value in local fused: ', torch.min(local_fused))
                # fake_pred, _ = global_ae(ref_gray, local_fused)
                # fake_img = fake_pred
        else:
            fake_stylecode = encoder(overlaid_rgb)
            fake_img, _ = generator(fake_stylecode, input_is_stylecode=True)

        temp_input = overlaid_rgb.detach()

        vis_input_np = tensor2image(temp_input)
        # if not is_global:
        #     vis_input_np = global_input_np # Visualize the new overlaid image, only when local method is used
        vis_input = (vis_input_np+1)/2.

        temp_fake = fake_img.detach()
        vis_fake = tensor2image(temp_fake)
        vis_fake = (vis_fake+1)/2.

        temp_gt = gt_rgb.detach()
        vis_gt = tensor2image(temp_gt)
        vis_gt = (vis_gt+1)/2.

        assert vis_fake.shape == vis_gt.shape
        binary_mask = np.zeros(mask.shape).astype(int)
        binary_mask[mask!=255] = 1
        num_pixel_in = np.count_nonzero(binary_mask==1)
        num_pixel_out = np.count_nonzero(binary_mask==0)
        error = vis_gt - vis_fake
        SE = error**2
        SE = np.sum(SE, axis=2)
        
        diff_image = (SE>1e-2)
        se_image = deepcopy(SE)

        SE_in = SE*binary_mask
        SE_out = SE*(1-binary_mask)
        MSE = np.mean(SE)
        MSE_in = np.sum(SE_in) / num_pixel_in
        MSE_out = np.sum(SE_out) / num_pixel_out
        mse_list.append(MSE)
        mse_in_list.append(MSE_in)
        mse_out_list.append(MSE_out)
        
        # print('MSE: ', MSE)
        # print('MSE_in: ', MSE_in)
        # print('MSE_out: ', MSE_out)

        # SSIM
        fake_gray = cv2.cvtColor(vis_fake, cv2.COLOR_BGR2GRAY)
        gt_gray = cv2.cvtColor(vis_gt, cv2.COLOR_BGR2GRAY)
        (score, diff) = structural_similarity(fake_gray, gt_gray, full=True)
        # print('SSIM: ', score)
        ssim_list.append(score)

        # PSNR
        new_fake = deepcopy(vis_fake)
        new_gt = deepcopy(vis_gt)
        new_fake = (255*new_fake).astype(np.uint8)
        new_gt = (255*new_gt).astype(np.uint8)
        psnr = cv2.PSNR(new_fake, new_gt)
        # print('PSNR: ', psnr)
        psnr_list.append(psnr)

        # LPIPS
        from training import lpips
        lpips_loss = lpips.exportPerceptualLoss(
                model="net-lin", net="vgg", use_gpu=device.startswith("cuda")
            )
        lpips_loss.to(device)
        d = lpips_loss(fake_img, gt_rgb).mean().item()
        lpips_list.append(d)
        # print('LPIPS: ', d)
        # input()

        nrow = 2
        ncol = 4
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

        ax7 = plt.subplot(gs[0, 3])
        ax7.imshow(diff_image)
        ax7.title.set_text('Difference Mask')
        ax7.set_xticklabels([])
        ax7.set_yticklabels([])
        ax7.axis('off')

        ax8 = plt.subplot(gs[1, 3])
        ax8.imshow(se_image)
        ax8.title.set_text('Squared Error')
        ax8.set_xticklabels([])
        ax8.set_yticklabels([])
        ax8.axis('off')

        if not is_global:
            ax4 = plt.subplot(gs[1, 0])
            ax4.imshow(vis_left_pred)
            ax4.title.set_text('Left pred')
            ax4.set_xticklabels([])
            ax4.set_yticklabels([])
            ax4.axis('off')

            ax5 = plt.subplot(gs[1, 1])
            ax5.imshow(vis_right_pred)
            ax5.title.set_text('Right pred')
            ax5.set_xticklabels([])
            ax5.set_yticklabels([])
            ax5.axis('off')

            ax6 = plt.subplot(gs[1, 2])
            ax6.imshow(vis_mouth_pred)
            ax6.title.set_text('Mouth pred')
            ax6.set_xticklabels([])
            ax6.set_yticklabels([])
            ax6.axis('off')

        save_name = str(i_iter)+'.png'
        image_save_path = os.path.join(test_save_path, save_name)
        fig.savefig(image_save_path)

    mean_MSE = sum(mse_list) / len(mse_list)
    mean_MSE_in = sum(mse_in_list) / len(mse_in_list)
    mean_MSE_out = sum(mse_out_list) / len(mse_out_list)
    mean_ssim = sum(ssim_list) / len(ssim_list)
    mean_psnr = sum(psnr_list) / len(psnr_list)
    mean_lpips = sum(lpips_list) / len(lpips_list)

    print('Mean of MSE: ', mean_MSE)
    print('Mean of MSE_in: ', mean_MSE_in)
    print('Mean of MSE_out: ', mean_MSE_out)
    print('Mean of SSIM: ', mean_ssim)
    print('Mean of PSNR: ', mean_psnr)
    print('Mean of LPIPS: ', mean_lpips)

        # test_save_path = os.path.join(args.save_image_dir, args.test_suffix)
        # if not os.path.exists(test_save_path):
        #     os.makedirs(test_save_path)
        # img_save_path = os.path.join(test_save_path, args.save_name)

        # print('Saving results to ... ', )
        # fig.savefig(img_save_path)
        # print('Image is saved.')

