"""
StyleMapGAN
Copyright (c) 2021-present NAVER Corp.

This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""

import argparse
from operator import truediv
import os
from tkinter import N
import torch
from torch import nn, autograd, optim
from torch.nn import functional as F
from torch.utils import data
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import transforms
from training import lpips
from training.model import Generator, Discriminator, Encoder, LocalPathway
from training.dataset_ddp import MultiResolutionDataset
from tqdm import tqdm
from tensorboardX import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from copy import deepcopy
from data_set.fdc import FDCDataset
from training.config import cfg

torch.backends.cudnn.benchmark = True

random_seed = 1234

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

def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def gather_grad(params, world_size):
    for param in params:
        if param.grad is not None:
            dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
            param.grad.data.div_(world_size)


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def accumulate(model1, model2, decay=0.999):
    with torch.no_grad():
        par1 = dict(model1.named_parameters())
        par2 = dict(model2.named_parameters())

        for k in par1.keys():
            par1[k].data.mul_(decay).add_(1 - decay, par2[k].data)


def copy_norm_params(model_tgt, model_src):
    with torch.no_grad():
        src_state_dict = model_src.state_dict()
        tgt_state_dict = model_tgt.state_dict()
        names = [name for name, _ in model_tgt.named_parameters()]

        for n in names:
            del src_state_dict[n]

        tgt_state_dict.update(src_state_dict)
        model_tgt.load_state_dict(tgt_state_dict)


def sample_data(loader):
    while True:
        for batch in loader:
            yield batch


def d_logistic_loss(real_pred, fake_pred):
    real_loss = F.softplus(-real_pred)
    fake_loss = F.softplus(fake_pred)

    return real_loss.mean() + fake_loss.mean()


def d_r1_loss(real_pred, real_img):
    (grad_real,) = autograd.grad(
        outputs=real_pred.sum(), inputs=real_img, create_graph=True
    )
    grad_penalty = grad_real.pow(2).view(grad_real.shape[0], -1).sum(1).mean()

    return grad_penalty


def g_nonsaturating_loss(fake_pred):
    loss = F.softplus(-fake_pred).mean()

    return loss


def make_noise(batch, latent_channel_size, device):
    return torch.randn(batch, latent_channel_size, device=device)

def filter_mse_values(mask, scale):
    """
    mask: a torch tensor
    """
    img_size = mask.shape[2]
    margin = int(img_size*scale)
    left_area = torch.where(mask==1) # left eye
    # todo
    

class DDPModel(nn.Module):
    def __init__(self, is_global, is_ae, is_gray, device, args):
        super(DDPModel, self).__init__()
        self.l1_loss = nn.L1Loss(size_average=True)
        self.mse_loss = nn.MSELoss(size_average=True)
        self.percept = lpips.exportPerceptualLoss(
            model="net-lin", net="vgg", use_gpu=device.startswith("cuda")
        )
        self.device = device
        self.args = args
        self.is_global = is_global
        self.is_ae = is_ae
        self.is_gray = is_gray
        
        if self.is_global: # Using global image
            if not self.is_ae: # Using GAN framework
                self.generator = Generator(
                    args.size,
                    args.mapping_layer_num,
                    args.latent_channel_size,
                    args.latent_spatial_size,
                    lr_mul=args.lr_mul,
                    channel_multiplier=args.channel_multiplier,
                    normalize_mode=args.normalize_mode,
                    small_generator=args.small_generator,
                )
                self.g_ema = Generator(
                    args.size,
                    args.mapping_layer_num,
                    args.latent_channel_size,
                    args.latent_spatial_size,
                    lr_mul=args.lr_mul,
                    channel_multiplier=args.channel_multiplier,
                    normalize_mode=args.normalize_mode,
                    small_generator=args.small_generator,
                )

                self.discriminator = Discriminator(
                    args.size, channel_multiplier=args.channel_multiplier
                )
                self.encoder = Encoder(
                    args.size,
                    args.latent_channel_size,
                    args.latent_spatial_size,
                    channel_multiplier=args.channel_multiplier,
                )
                self.e_ema = Encoder(
                    args.size,
                    args.latent_channel_size,
                    args.latent_spatial_size,
                    channel_multiplier=args.channel_multiplier,
                )
            else: # Using AE framework
                if self.is_gray: # Using gray global image
                    self.global_gray_ae = LocalPathway(self.is_gray, use_batchnorm=cfg.TRAIN.GRAY2RGB_USE_BATCHNORM)
                else: # Using rgb global image
                    self.global_rgb_ae = LocalPathway(self.is_gray, use_batchnorm=cfg.TRAIN.GRAY2RGB_USE_BATCHNORM)

        self.local_pathway_left_eye = LocalPathway(use_batchnorm=cfg.TRAIN.GRAY2RGB_USE_BATCHNORM)
        self.local_pathway_right_eye = LocalPathway(use_batchnorm=cfg.TRAIN.GRAY2RGB_USE_BATCHNORM)
        self.local_pathway_mouth = LocalPathway(use_batchnorm=cfg.TRAIN.GRAY2RGB_USE_BATCHNORM)

        self.global_side_pathway = LocalPathway(use_batchnorm=cfg.TRAIN.GRAY2RGB_USE_BATCHNORM)
        self.gloabl_frontal_pathway = LocalPathway(use_batchnorm=cfg.TRAIN.GRAY2RGB_USE_BATCHNORM)

    def forward(self, real_img, mode):
        if mode == "G":
            z = make_noise(
                self.args.batch_per_gpu,
                self.args.latent_channel_size,
                self.device,
            )

            fake_img, stylecode = self.generator(z, return_stylecode=True)
            fake_pred = self.discriminator(fake_img)
            adv_loss = g_nonsaturating_loss(fake_pred)
            # fake_img = fake_img.detach()
            # stylecode = stylecode.detach()
            # fake_stylecode = self.encoder(fake_img)
            # w_rec_loss = self.mse_loss(stylecode, fake_stylecode)

            # return adv_loss, w_rec_loss, stylecode, fake_stylecode, fake_img
            return adv_loss, stylecode, fake_img

        elif mode == "D":
            with torch.no_grad():
                z = make_noise(
                    self.args.batch_per_gpu,
                    self.args.latent_channel_size,
                    self.device,
                )
                fake_img, _ = self.generator(z)
                fake_stylecode = self.encoder(real_img)
                fake_img_from_E, _ = self.generator(
                    fake_stylecode, input_is_stylecode=True
                )
            
            real_pred = self.discriminator(real_img)
            fake_pred = self.discriminator(fake_img)
            d_loss = d_logistic_loss(real_pred, fake_pred)
            fake_pred_from_E = self.discriminator(fake_img_from_E)
            indomainGAN_D_loss = F.softplus(fake_pred_from_E).mean()

            return (
                d_loss,
                indomainGAN_D_loss,
                real_pred.mean(),
                fake_pred.mean(),
            )

        elif mode == "D_reg":
            real_img.requires_grad = True
            real_pred = self.discriminator(real_img)
            r1_loss = d_r1_loss(real_pred, real_img)
            d_reg_loss = (
                self.args.r1 / 2 * r1_loss * self.args.d_reg_every + 0 * real_pred[0]
            )

            return d_reg_loss, r1_loss

        elif mode == "E_x_rec":
            # Here the real_img is a list of image batches, the first one is the overlaid image
            # the second one is the ground-truth rgb image
            real_img = real_img[0] # Read the input image
            gt = real_img[1] # Read the gt image
            fake_stylecode = self.encoder(real_img)
            fake_img, _ = self.generator(fake_stylecode, input_is_stylecode=True)
            x_rec_loss = self.mse_loss(gt, fake_img)
            perceptual_loss = self.percept(gt, fake_img).mean()
            fake_pred_from_E = self.discriminator(fake_img)
            indomainGAN_E_loss = F.softplus(-fake_pred_from_E).mean()

            real_stylecode = fake_stylecode
            return x_rec_loss, perceptual_loss, indomainGAN_E_loss, real_stylecode

        elif mode == "cal_mse_lpips":
            fake_stylecode = self.e_ema(real_img)
            fake_img, _ = self.g_ema(fake_stylecode, input_is_stylecode=True)
            x_rec_loss = self.mse_loss(real_img, fake_img)
            perceptual_loss = self.percept(real_img, fake_img).mean()

            return x_rec_loss, perceptual_loss
        elif mode == 'left':
            left_gray = real_img[0]
            left_gt = real_img[1]
            # Note the input channels
            left_fake, left_feature = self.local_pathway_left_eye(left_gray)
            left_rec_loss = self.mse_loss(left_gt, left_fake)
            left_perceptual_loss = self.percept(left_gt, left_fake).mean()
            return left_fake, left_feature, left_rec_loss, left_perceptual_loss
        elif mode == 'right':
            right_gray = real_img[0]
            right_gt = real_img[1]
            # Note the input channels
            right_fake, right_feature = self.local_pathway_right_eye(right_gray)
            right_rec_loss = self.mse_loss(right_gt, right_fake)
            right_perceptual_loss = self.percept(right_gt, right_fake).mean()
            return right_fake, right_feature, right_rec_loss, right_perceptual_loss
        elif mode == 'mouth':
            mouth_gray = real_img[0]
            mouth_gt = real_img[1]
            # Note the input channels
            mouth_fake, mouth_feature = self.local_pathway_mouth(mouth_gray)
            mouth_rec_loss = self.mse_loss(mouth_gt, mouth_fake)
            mouth_perceptual_loss = self.percept(mouth_gt, mouth_fake).mean()
            return mouth_fake, mouth_feature, mouth_rec_loss, mouth_perceptual_loss
        elif mode == 'global_gray_ae':
            # Input is the overlaid image with the raw side patches
            global_side_gray = real_img[0]
            global_gray_gt = real_img[1]
            
            global_side_fake, global_side_feature = self.global_gray_ae(global_side_gray)
            global_side_rec_loss = self.mse_loss(global_gray_gt, global_side_fake)
            global_side_perceptual_loss = self.percept(global_gray_gt, global_side_fake).mean()
            return global_side_fake, global_side_feature, global_side_rec_loss, global_side_perceptual_loss
        elif mode == 'global_rgb_ae':
            # Input is the overlaid image with the raw side patches
            global_side_rgb = real_img[0]
            global_gray_gt = real_img[1]
            
            global_side_fake, global_side_feature = self.global_rgb_ae(global_side_gray)
            global_side_rec_loss = self.mse_loss(global_gray_gt, global_side_fake)
            global_side_perceptual_loss = self.percept(global_gray_gt, global_side_fake).mean()
            return global_side_fake, global_side_feature, global_side_rec_loss, global_side_perceptual_loss
            
def run(ddp_fn, world_size, args):
    print("world size", world_size)
    mp.spawn(ddp_fn, args=(world_size, args), nprocs=world_size, join=True)

def init_fn(worker_id):
        np.random.seed(random_seed + worker_id)

def ddp_main(rank, world_size, args):
    print(f"Running DDP model on rank {rank}.")    
    setup(rank, world_size)
    map_location = f"cuda:{rank}"

    torch.cuda.set_device(map_location)
    
    is_celeba = False
    is_debugging = False
    is_tensorboard = True
    # is_gray_2_rgb = True

    is_ae = True # Use autoencoder structure
    is_gray = False # Is the input image grayscale or rgb?
    is_global = True # Is the input small patch or a whole face

    global cfg

    # Those two variable does not exist if the resume iter is not proper
    r1_val = None
    d_reg_loss_val = None

    # This condition being true means the training is resumed
    if args.ckpt:  # ignore current arguments
        if is_ae:
            ckpt = torch.load(args.ckpt, map_location=map_location)
            train_args = ckpt["train_args"]
            cfg = ckpt["cfg"] # Note: Only read this when pathway net is used
            print("load model:", args.ckpt)
            train_args.start_iter = int(args.ckpt.split("/")[-1].replace(".pt", ""))
            print(f"continue training from {train_args.start_iter} iter")
            args = train_args
            args.ckpt = True
        else:
            ckpt = torch.load(args.ckpt, map_location=map_location)
            train_args = ckpt["train_args"]
            # cfg = ckpt["cfg"] # Note: Only read this when pathway net is used
            print("load model:", args.ckpt)
            train_args.start_iter = int(args.ckpt.split("/")[-1].replace(".pt", ""))
            print(f"continue training from {train_args.start_iter} iter")
            args = train_args
            args.ckpt = True
    else:
        args.start_iter = 0

    # create model and move it to GPU with id rank
    model = DDPModel(is_gray, is_global, device=map_location, args=args).to(map_location)
    model = DDP(model, device_ids=[rank], find_unused_parameters=True)
    model.train()

    g_module = model.module.generator
    g_ema_module = model.module.g_ema
    g_ema_module.eval()
    accumulate(g_ema_module, g_module, 0)

    e_module = model.module.encoder
    e_ema_module = model.module.e_ema
    e_ema_module.eval()
    accumulate(e_ema_module, e_module, 0)

    d_reg_ratio = args.d_reg_every / (args.d_reg_every + 1)

    g_optim = optim.Adam(
        g_module.parameters(),
        lr=args.lr,
        betas=(0, 0.99),
    )

    d_optim = optim.Adam(
        model.module.discriminator.parameters(),
        lr=args.lr * d_reg_ratio,
        betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio),
    )

    e_optim = optim.Adam(
        e_module.parameters(),
        lr=args.lr,
        betas=(0, 0.99),
    )

    accum = 0.999

    local_left_optim = optim.Adam(
        model.module.local_pathway_left_eye.parameters(),
        lr = cfg.TRAIN.GRAY2RGB_LR
    )

    local_right_optim = optim.Adam(
        model.module.local_pathway_right_eye.parameters(),
        lr = cfg.TRAIN.GRAY2RGB_LR
    )

    local_mouth_optim = optim.Adam(
        model.module.local_pathway_mouth.parameters(),
        lr = cfg.TRAIN.GRAY2RGB_LR
    )

    global_side_optim = optim.Adam(
        model.module.global_side_pathway.parameters(),
        lr = cfg.TRAIN.GRAY2RGB_LR
    )

    if args.ckpt:
        if is_ae:
            # model.module.local_pathway_left_eye.load_state_dict(ckpt["local_pathway_left_eye"])
            # model.module.local_pathway_right_eye.load_state_dict(ckpt["local_pathway_right_eye"])
            # model.module.local_pathway_mouth.load_state_dict(ckpt["local_pathway_mouth"])
            if is_gray:
                model.module.global_side_pathway.load_state_dict(ckpt["gray_ae"])
            else:
                model.module.global_side_pathway.load_state_dict(ckpt["rgb_ae"])
        else:
            model.module.generator.load_state_dict(ckpt["generator"])
            model.module.discriminator.load_state_dict(ckpt["discriminator"])
            model.module.g_ema.load_state_dict(ckpt["g_ema"])
            g_optim.load_state_dict(ckpt["g_optim"])
            d_optim.load_state_dict(ckpt["d_optim"])

            model.module.encoder.load_state_dict(ckpt["encoder"])
            e_optim.load_state_dict(ckpt["e_optim"])
            model.module.e_ema.load_state_dict(ckpt["e_ema"])

        del ckpt  # free GPU memory

    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
        ]
    )

    save_dir = "expr"
    os.makedirs(save_dir, 0o777, exist_ok=True)
    os.makedirs(save_dir + "/checkpoints/" + args.suffix, 0o777, exist_ok=True)

    if is_celeba:
        train_dataset = MultiResolutionDataset(args.train_lmdb, transform, args.size)
        val_dataset = MultiResolutionDataset(args.val_lmdb, transform, args.size)
    else:
        train_dataset = FDCDataset(list_path=args.list_path, set='train')

    if is_celeba:
        print(f"train_dataset: {len(train_dataset)}, val_dataset: {len(val_dataset)}")
    else:
        print(f"train_dataset: {len(train_dataset)}")

    if is_celeba:
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_dataset, num_replicas=world_size, rank=rank, shuffle=True
        )
        val_loader = data.DataLoader(
            val_dataset,
            batch_size=args.batch_per_gpu,
            drop_last=True,
            sampler=val_sampler,
            num_workers=args.num_workers,
            pin_memory=True,
        )

        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, num_replicas=world_size, rank=rank, shuffle=True
        )

        train_loader = data.DataLoader(
            train_dataset,
            batch_size=args.batch_per_gpu,
            drop_last=True,
            sampler=train_sampler,
            num_workers=args.num_workers,
            pin_memory=True,
        )

        train_loader = sample_data(train_loader)
    else:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, num_replicas=world_size, rank=rank, shuffle=True
        )
        train_loader = data.DataLoader(train_dataset,
                                        batch_size=args.batch,
                                        drop_last=True,
                                        sampler=train_sampler,
                                        num_workers=args.num_workers,
                                        pin_memory=True,
                                        worker_init_fn=init_fn)
        train_iter = enumerate(train_loader)

    pbar = range(args.start_iter, args.iter)
    pbar = tqdm(pbar, initial=args.start_iter, mininterval=1)

    requires_grad(model.module.discriminator, False)
    epoch = -1
    gpu_group = dist.new_group(list(range(args.ngpus)))

    os.makedirs(args.tensorboard_logdir, exist_ok=True) # To avoid the file exists error
    writer = SummaryWriter(log_dir=args.tensorboard_logdir)
    torch.autograd.set_detect_anomaly(True)

    for i in pbar:
        if i > args.iter:
            print("Done!")
            break
        elif i % (len(train_dataset) // args.batch) == 0:
            epoch += 1
            if is_celeba:
                val_sampler.set_epoch(epoch)
                train_sampler.set_epoch(epoch)
            else:
                train_sampler.set_epoch(epoch)
                train_loader = data.DataLoader(train_dataset,
                                        batch_size=args.batch,
                                        drop_last=True,
                                        sampler=train_sampler,
                                        num_workers=args.num_workers,
                                        pin_memory=True,
                                        worker_init_fn=init_fn)
                train_iter = enumerate(train_loader)

            print("epoch: ", epoch)

        if is_celeba:
            real_img = next(train_loader)
            real_img = real_img.to(map_location) # Using celeba dataset
        else:
            batch_list = train_iter.__next__()
            temp_batch = batch_list[1]
            ## ======== Read inputs for original and gray2rgb ========
            # real_img, gt, left_gray, right_gray, mouth_gray, left_gt, right_gt, mouth_gt, _, _ = temp_batch

            # real_img = real_img.to(map_location)
            # gt = gt.to(map_location)
            # left_gray = left_gray.to(map_location)
            # right_gray = right_gray.to(map_location)
            # mouth_gray = mouth_gray.to(map_location)
            # left_gt = left_gt.to(map_location)
            # right_gt = right_gt.to(map_location)
            # mouth_gt = mouth_gt.to(map_location)

            # ======== Read inputs for new model ========
            # ref_rgb, ref_gray, overlaid_gray, gt_rgb, gt_gray, mask, left_gray, right_gray, mouth_gray, _, _ = temp_batch
            # ref_rgb = ref_rgb.to(map_location)
            # ref_gray = ref_gray.to(map_location)
            # overlaid_gray = overlaid_gray.to(map_location)
            # gt_rgb = gt_rgb.to(map_location)
            # gt_gray = gt_gray.to(map_location)
            # mask = mask.to(map_location)
            # left_gray = left_gray.to(map_location)
            # right_gray = right_gray.to(map_location)
            # mouth_gray = mouth_gray.to(map_location)
            # if ref_rgb.shape[0] != args.batch:
            #     print(f'Data batch wrong---{real_img.shape[0]}, continue...')
            #     continue

            # overlaid_gray, gt_gray, _, _ = temp_batch
            overlaid_rgb, overlaid_gray, gt_rgb, gt_gray, _, _ = temp_batch

            overlaid_rgb = overlaid_rgb.to(map_location)
            overlaid_gray = overlaid_gray.to(map_location)
            gt_rgb = gt_rgb.to(map_location)
            gt_gray = gt_gray.to(map_location)
            if overlaid_gray.shape[0] != args.batch:
                print(f'Data batch wrong---{real_img.shape[0]}, continue...')
                continue
        
        # if real_img.shape[0] != args.batch:
        #     print(f'Data batch wrong---{real_img.shape[0]}, continue...')
        #     continue

        if is_gray_2_rgb:
            global_side_input = [overlaid_gray, gt_gray]
            global_side_fake, global_side_feature, global_side_rec_loss, global_side_perceptual_loss = model(global_side_input, 'global_side')
            global_side_optim.zero_grad()
            global_side_loss = (global_side_rec_loss*args.lambda_x_rec_loss + 
                                global_side_perceptual_loss*args.lambda_perceptual_loss)
            global_side_loss.backward()
            global_side_optim.step()

            print('Writing losses to tensorboard...')
            writer.add_scalar('train/global_side_rec_loss', global_side_rec_loss, i)
            writer.add_scalar('train/global_side_percept_loss', global_side_perceptual_loss, i)

            log_imgs_every = 5
            if i%log_imgs_every == 0:
                temp_global_side_gray = overlaid_gray.detach()
                vis_global_side_gray = tensor2image(temp_global_side_gray)
                vis_global_side_gray = (vis_global_side_gray+1)/2.

                temp_global_side_fake = global_side_fake.detach()
                vis_global_side_fake = tensor2image(temp_global_side_fake)
                vis_global_side_fake = (vis_global_side_fake+1)/2.

                temp_gt_gray = gt_gray.detach()
                vis_gt_gray = tensor2image(temp_gt_gray)
                vis_gt_gray = (vis_gt_gray+1)/2.

                print('Writing images to tensorboard')
                nrow = 1
                ncol = 3
                fig = plt.figure(figsize=(10, 10))
                gs = gridspec.GridSpec(nrow, ncol,
                                                wspace=0.1, hspace=0.0, 
                                                top=1.-0.5/(nrow+1), bottom=0.5/(nrow+1), 
                                                left=0.5/(ncol+1), right=1-0.5/(ncol+1))
                ax1 = plt.subplot(gs[0, 0])
                ax1.imshow(vis_global_side_gray)
                # ax1.title.set_text('Real Image')
                ax1.set_xticklabels([])
                ax1.set_yticklabels([])
                ax1.axis('off')

                ax2 = plt.subplot(gs[0, 1])
                ax2.imshow(vis_global_side_fake)
                # ax2.title.set_text('Fake Image')
                ax2.set_xticklabels([])
                ax2.set_yticklabels([])
                ax2.axis('off')

                ax3 = plt.subplot(gs[0, 2])
                ax3.imshow(vis_gt_gray)
                # ax2.title.set_text('Fake Image')
                ax3.set_xticklabels([])
                ax3.set_yticklabels([])
                ax3.axis('off')
                writer.add_figure('train_figs'+str(i), fig)
            if i%args.save_network_interval == 0:
                torch.save(
                    {
                        "global_side_pathway": model.module.global_side_pathway.state_dict(),
                        "train_args": args,
                        "cfg": cfg,
                    },
                    f"{save_dir}/checkpoints/{str(i).zfill(6)}.pt",
                )

            # left_model_input = [left_gray, left_gt]
            # left_fake, left_feature, left_rec_loss, left_perceptual_loss = model(left_model_input, 'left')
            # local_left_optim.zero_grad()
            # left_local_loss = (left_rec_loss*args.lambda_x_rec_loss+
            #                     left_perceptual_loss*args.lambda_perceptual_loss)            
            # left_local_loss.backward()
            # local_left_optim.step()

            # right_model_input = [right_gray, right_gt]
            # right_fake, right_feature, right_rec_loss, right_perceptual_loss = model(right_model_input, 'right')
            # local_right_optim.zero_grad()
            # right_local_loss = (right_rec_loss*args.lambda_x_rec_loss+
            #                     right_perceptual_loss*args.lambda_perceptual_loss)
            # right_local_loss.backward()
            # local_right_optim.step()

            # mouth_model_input = [mouth_gray, mouth_gt]
            # mouth_fake, mouth_feature, mouth_rec_loss, mouth_perceptual_loss = model(mouth_model_input, 'mouth')
            # local_mouth_optim.zero_grad()
            # mouth_local_loss = (mouth_rec_loss*args.lambda_x_rec_loss+
            #                     mouth_perceptual_loss*args.lambda_perceptual_loss)
            # mouth_local_loss.backward()
            # local_mouth_optim.step()

            # if is_tensorboard:
            #     # Left gray image
            #     temp_left_gray = left_gray.detach()
            #     vis_left_gray = tensor2image(temp_left_gray)
            #     vis_left_gray = (vis_left_gray+1)/2.

            #     # Left gt image
            #     temp_left_gt = left_gt.detach()
            #     vis_left_gt = tensor2image(temp_left_gt)
            #     vis_left_gt = (vis_left_gt+1)/2.

            #     # Left fake image
            #     temp_left_fake = left_fake.detach()
            #     vis_left_fake = tensor2image(temp_left_fake)
            #     vis_left_fake = (vis_left_fake+1)/2.

            #     # Right gray image
            #     temp_right_gray = right_gray.detach()
            #     vis_right_gray = tensor2image(temp_right_gray)
            #     vis_right_gray = (vis_right_gray+1)/2.

            #     # Right gt image
            #     temp_right_gt = right_gt.detach()
            #     vis_right_gt = tensor2image(temp_right_gt)
            #     vis_right_gt = (vis_right_gt+1)/2.
                
            #     # Right fake image
            #     temp_right_fake = right_fake.detach()
            #     vis_right_fake = tensor2image(temp_right_fake)
            #     vis_right_fake = (vis_right_fake+1)/2.

            #     # Mouth gray image
            #     temp_mouth_gray = mouth_gray.detach()
            #     vis_mouth_gray = tensor2image(temp_mouth_gray)
            #     vis_mouth_gray = (vis_mouth_gray+1)/2.
                
            #     # Mouth gt image
            #     temp_mouth_gt = mouth_gt.detach()
            #     vis_mouth_gt = tensor2image(temp_mouth_gt)
            #     vis_mouth_gt = (vis_mouth_gt+1)/2.

            #     # Mouth fake image
            #     temp_mouth_fake = mouth_fake.detach()
            #     vis_mouth_fake = tensor2image(temp_mouth_fake)
            #     vis_mouth_fake = (vis_mouth_fake+1)/2.

            #     print('Writing losses to tensorboard...')
            #     writer.add_scalar('train/left_rec_loss', left_rec_loss, i)
            #     writer.add_scalar('train/left_percept_loss', left_perceptual_loss, i)
            #     writer.add_scalar('train/right_rec_loss', right_rec_loss, i)
            #     writer.add_scalar('train/right_percept_loss', right_perceptual_loss, i)
            #     writer.add_scalar('train/mouth_rec_loss', mouth_rec_loss, i)
            #     writer.add_scalar('train/mouth_percept_loss', mouth_perceptual_loss, i)

            #     log_imgs_every = 5
            #     if i%log_imgs_every == 0:
            #         nrow = 3
            #         ncol = 3
            #         fig = plt.figure(figsize=(10, 10))
            #         gs = gridspec.GridSpec(nrow, ncol,
            #                                         wspace=0.1, hspace=0.0, 
            #                                         top=1.-0.5/(nrow+1), bottom=0.5/(nrow+1), 
            #                                         left=0.5/(ncol+1), right=1-0.5/(ncol+1))
            #         ax1 = plt.subplot(gs[0, 0])
            #         ax1.imshow(vis_left_gray)
            #         # ax1.title.set_text('Real Image')
            #         ax1.set_xticklabels([])
            #         ax1.set_yticklabels([])
            #         ax1.axis('off')

            #         ax2 = plt.subplot(gs[0, 1])
            #         ax2.imshow(vis_right_gray)
            #         # ax2.title.set_text('Fake Image')
            #         ax2.set_xticklabels([])
            #         ax2.set_yticklabels([])
            #         ax2.axis('off')

            #         ax3 = plt.subplot(gs[0, 2])
            #         ax3.imshow(vis_mouth_gray)
            #         # ax3.title.set_text('Fake Image')
            #         ax3.set_xticklabels([])
            #         ax3.set_yticklabels([])
            #         ax3.axis('off')

            #         ax4 = plt.subplot(gs[1, 0])
            #         ax4.imshow(vis_left_fake)
            #         # ax2.title.set_text('Fake Image')
            #         ax4.set_xticklabels([])
            #         ax4.set_yticklabels([])
            #         ax4.axis('off')

            #         ax5 = plt.subplot(gs[1, 1])
            #         ax5.imshow(vis_right_fake)
            #         # ax2.title.set_text('Fake Image')
            #         ax5.set_xticklabels([])
            #         ax5.set_yticklabels([])
            #         ax5.axis('off')

            #         ax6 = plt.subplot(gs[1, 2])
            #         ax6.imshow(vis_mouth_fake)
            #         # ax2.title.set_text('Fake Image')
            #         ax6.set_xticklabels([])
            #         ax6.set_yticklabels([])
            #         ax6.axis('off')

            #         ax7 = plt.subplot(gs[2, 0])
            #         ax7.imshow(vis_left_gt)
            #         # ax7.title.set_text('Fake Image')
            #         ax7.set_xticklabels([])
            #         ax7.set_yticklabels([])
            #         ax7.axis('off')

            #         ax8 = plt.subplot(gs[2, 1])
            #         ax8.imshow(vis_right_gt)
            #         # ax8.title.set_text('Fake Image')
            #         ax8.set_xticklabels([])
            #         ax8.set_yticklabels([])
            #         ax8.axis('off')

            #         ax9 = plt.subplot(gs[2, 2])
            #         ax9.imshow(vis_mouth_gt)
            #         # ax2.title.set_text('Fake Image')
            #         ax9.set_xticklabels([])
            #         ax9.set_yticklabels([])
            #         ax9.axis('off')
            #         writer.add_figure('train_figs'+str(i), fig)
            
            # if i%args.save_network_interval == 0:
            #     torch.save(
            #         {
            #             "local_pathway_left_eye": model.module.local_pathway_left_eye.state_dict(),
            #             "local_pathway_right_eye": model.module.local_pathway_right_eye.state_dict(),
            #             "local_pathway_mouth": model.module.local_pathway_mouth.state_dict(),
            #             "train_args": args,
            #             "cfg": cfg,
            #         },
            #         f"{save_dir}/checkpoints/{str(i).zfill(6)}.pt",
            #     )
        else:
            # Here stylecode is from noise z; fake_stylecode is from encoder
            # adv_loss, w_rec_loss, stylecode, fake_stylecode, fake_img = model(None, "G")
            adv_loss, stylecode, fake_img = model(None, "G")
            real_img = overlaid_rgb
            gt = gt_rgb
            if is_debugging:
                print('real img: ', real_img.shape)
                print('fake_img: ', fake_img.shape)
                # print('fake stylecode: ', fake_stylecode.shape)
                print('z stylecode: ', stylecode.shape)
            
            if is_tensorboard:
                # temp_real_img = deepcopy(real_img)
                temp_real_img = real_img.detach()
                vis_real_img = tensor2image(temp_real_img)
                vis_real_img = (vis_real_img+1)/2.

                temp_gt = gt.detach()
                vis_gt_img = tensor2image(temp_gt)
                vis_gt_img = (vis_gt_img+1)/2.

                # temp_fake_img = deepcopy(fake_img)
                temp_fake_img = fake_img.detach()
                vis_fake_img = tensor2image(temp_fake_img)
                vis_fake_img = (vis_fake_img+1)/2.

                # temp_fake_stylecode = deepcopy(fake_stylecode)
                # temp_fake_stylecode = fake_stylecode.detach()
                # vis_fake_stylecode = tensor2image(temp_fake_stylecode)

                # temp_z_stylecode = deepcopy(stylecode)
                temp_z_stylecode = stylecode.detach()
                vis_z_stylecode = tensor2image(temp_z_stylecode)

            adv_loss = adv_loss.mean()

            with torch.no_grad():
                latent_std = stylecode.std().mean().item()
                latent_channel_std = stylecode.std(dim=1).mean().item()
                latent_spatial_std = stylecode.std(dim=(2, 3)).mean().item()

            g_loss = adv_loss * args.lambda_adv_loss
            g_loss_val = g_loss.item()
            adv_loss_val = adv_loss.item()

            if is_debugging:
                print('adv_loss_val: ', adv_loss_val)

            g_optim.zero_grad()
            g_loss.backward()
            gather_grad(
                g_module.parameters(), world_size
            )  # Explicitly synchronize Generator parameters. There is a gradient sync bug in G.
            g_optim.step()

            # w_rec_loss = w_rec_loss.mean()
            # w_rec_loss_val = w_rec_loss.item()

            # if is_debugging:
            #     print('w_rec_loss_val: ', w_rec_loss_val)

            # e_optim.zero_grad()
            # (w_rec_loss * args.lambda_w_rec_loss).backward()
            # e_optim.step()

            requires_grad(model.module.discriminator, True)

            # D adv
            d_loss, indomainGAN_D_loss, real_score, fake_score = model(gt, "D") # Replace real_img with gt
            d_loss = d_loss.mean()
            indomainGAN_D_loss = indomainGAN_D_loss.mean()
            indomainGAN_D_loss_val = indomainGAN_D_loss.item()

            if is_debugging:
                print('indomainGAN_D_loss_val: ', indomainGAN_D_loss_val)

            d_loss_val = d_loss.item()

            if is_debugging:
                print('d_loss_val: ', d_loss_val)

            d_optim.zero_grad()

            (
                d_loss * args.lambda_d_loss
                + indomainGAN_D_loss * args.lambda_indomainGAN_D_loss
            ).backward()
            d_optim.step()

            real_score_val = real_score.mean().item()
            fake_score_val = fake_score.mean().item()

            if is_debugging:
                print('real_score_val: ', real_score_val)
                print('fake_score_val: ', fake_score_val)

            # D reg
            d_regularize = i % args.d_reg_every == 0
            if d_regularize:
                d_reg_loss, r1_loss = model(gt, "D_reg") # Replace the real_img with gt
                d_reg_loss = d_reg_loss.mean()
                d_reg_loss_val = d_reg_loss.item()
                
                d_optim.zero_grad()
                d_reg_loss.backward()
                d_optim.step()
                r1_val = r1_loss.mean().item()
                if is_debugging:
                    print('d_reg_loss_val: ', d_reg_loss_val)
                    print('r1_val: ', r1_val)

            requires_grad(model.module.discriminator, False)

            # E_x_rec
            model_input = [real_img, gt]
            x_rec_loss, perceptual_loss, indomainGAN_E_loss, real_stylecode = model(model_input, "E_x_rec")
            x_rec_loss = x_rec_loss.mean()
            perceptual_loss = perceptual_loss.mean()

            if indomainGAN_E_loss is not None:
                indomainGAN_E_loss = indomainGAN_E_loss.mean()
                indomainGAN_E_loss_val = indomainGAN_E_loss.item()
            else:
                indomainGAN_E_loss = 0
                indomainGAN_E_loss_val = 0

            if is_debugging:
                print('real_stylecode: ', real_stylecode.shape)
                print('indomainGAN_E_loss_val: ', indomainGAN_E_loss_val)

            if is_tensorboard:
                # temp_real_stylecode = deepcopy(real_stylecode)
                temp_real_stylecode = real_stylecode.detach()
                vis_real_stylecode = tensor2image(temp_real_stylecode)

            e_optim.zero_grad()
            g_optim.zero_grad()

            encoder_loss = 100*(
                x_rec_loss * args.lambda_x_rec_loss
                + perceptual_loss * args.lambda_perceptual_loss
                + indomainGAN_E_loss * args.lambda_indomainGAN_E_loss
            )

            encoder_loss.backward()
            e_optim.step()
            g_optim.step()

            x_rec_loss_val = x_rec_loss.item()
            perceptual_loss_val = perceptual_loss.item()
            
            if is_debugging:
                print('x_rec_loss_val: ', x_rec_loss_val)
                print('perceptual_loss_val: ', perceptual_loss_val)

            if r1_val is None:
                pbar.set_description(
                    (f"g: {g_loss_val:.4f}; d: {d_loss_val:.4f};")
                )
            else:
                pbar.set_description(
                    (f"g: {g_loss_val:.4f}; d: {d_loss_val:.4f}; r1: {r1_val:.4f};")
                )
            
            if is_tensorboard:
                # Log losses and images to tensorboard
                print('Writing losses to tensorboard...')
                writer.add_scalar('train/adv_loss', adv_loss_val, i)
                # writer.add_scalar('train/w_rec_loss', w_rec_loss_val, i)
                writer.add_scalar('train/indomaingan_d_loss', indomainGAN_D_loss_val, i)
                writer.add_scalar('train/d_loss', d_loss_val, i)
                writer.add_scalar('train/real_score', real_score_val, i)
                writer.add_scalar('train/fake_score', fake_score_val, i)
                if d_reg_loss_val is not None and r1_val is not None:
                    writer.add_scalar('train/d_reg_loss', d_reg_loss_val, i)
                    writer.add_scalar('train/r1', r1_val, i)
                writer.add_scalar('train/indomaingan_e_loss', indomainGAN_E_loss_val, i)
                writer.add_scalar('train/x_rec_loss', x_rec_loss_val, i)
                writer.add_scalar('train/perceptual_loss', perceptual_loss_val, i)
                
                # Log images with a smaller frequency
                log_imgs_every = 5
                if i%log_imgs_every == 0:
                    print('max of vis_fake_img: ', np.max(vis_fake_img))
                    print('min of vis_fake_img: ', np.min(vis_fake_img))

                    nrow = 2
                    ncol = 3
                    fig = plt.figure(figsize=(12, 10))
                    gs = gridspec.GridSpec(nrow, ncol,
                                                    wspace=0.1, hspace=0.0, 
                                                    top=1.-0.5/(nrow+1), bottom=0.5/(nrow+1), 
                                                    left=0.5/(ncol+1), right=1-0.5/(ncol+1))
                    ax1 = plt.subplot(gs[0, 0])
                    ax1.imshow(vis_real_img)
                    ax1.title.set_text('Real Image')
                    ax1.set_xticklabels([])
                    ax1.set_yticklabels([])
                    ax1.axis('off')

                    ax2 = plt.subplot(gs[0, 1])
                    ax2.imshow(vis_fake_img)
                    ax2.title.set_text('Fake Image')
                    ax2.set_xticklabels([])
                    ax2.set_yticklabels([])
                    ax2.axis('off')

                    ax3 = plt.subplot(gs[0, 2])
                    ax3.imshow(vis_gt_img)
                    ax3.title.set_text('GT Image')
                    ax3.set_xticklabels([])
                    ax3.set_yticklabels([])
                    ax3.axis('off')

                    ax4 = plt.subplot(gs[1, 0])
                    im4 = ax4.imshow(vis_z_stylecode)
                    divider = make_axes_locatable(ax4)
                    cax = divider.append_axes('right', size='5%', pad=0.05)
                    fig.colorbar(im4, cax=cax, orientation='vertical')
                    ax4.title.set_text('z Stylecode (from F)')
                    ax4.set_xticklabels([])
                    ax4.set_yticklabels([])
                    ax4.axis('off')

                    ax5 = plt.subplot(gs[1, 1])
                    im5 = ax5.imshow(vis_real_stylecode)
                    divider = make_axes_locatable(ax5)
                    cax = divider.append_axes('right', size='5%', pad=0.05)
                    fig.colorbar(im5, cax=cax, orientation='vertical')
                    ax5.title.set_text('Real Stylecode (from E)')
                    ax5.set_xticklabels([])
                    ax5.set_yticklabels([])
                    ax5.axis('off')

                    # ax6 = plt.subplot(gs[1, 2])
                    # im6 = ax6.imshow(vis_fake_stylecode)
                    # divider = make_axes_locatable(ax6)
                    # cax = divider.append_axes('right', size='5%', pad=0.05)
                    # fig.colorbar(im6, cax=cax, orientation='vertical')
                    # ax6.title.set_text('Fake Stylecode (from E)')
                    # ax6.set_xticklabels([])
                    # ax6.set_yticklabels([])
                    # ax6.axis('off')
                    
                    print('Writing images to tensorboard...')
                    writer.add_figure('train_figs'+str(i), fig)
            
            if i%args.save_network_interval == 0:
                print('Saving network parameters...')
                torch.save(
                    {
                        "generator": model.module.generator.state_dict(),
                        "discriminator": model.module.discriminator.state_dict(),
                        "encoder": model.module.encoder.state_dict(),
                        "g_ema": g_ema_module.state_dict(),
                        "e_ema": e_ema_module.state_dict(),
                        "train_args": args,
                        "e_optim": e_optim.state_dict(),
                        "g_optim": g_optim.state_dict(),
                        "d_optim": d_optim.state_dict(),
                    },
                    f"{save_dir}/checkpoints/{args.suffix}/{str(i).zfill(6)}.pt",
                )
        
        # Original
        if is_celeba:
            # Validation
            with torch.no_grad():
                accumulate(g_ema_module, g_module, accum)
                accumulate(e_ema_module, e_module, accum)

                if i % args.save_network_interval == 0:
                    copy_norm_params(g_ema_module, g_module)
                    copy_norm_params(e_ema_module, e_module)
                    x_rec_loss_avg, perceptual_loss_avg = 0, 0
                    iter_num = 0

                    for test_image in tqdm(val_loader):
                        test_image = test_image.to(map_location)
                        x_rec_loss, perceptual_loss = model(test_image, "cal_mse_lpips")
                        x_rec_loss_avg += x_rec_loss.mean()
                        perceptual_loss_avg += perceptual_loss.mean()
                        iter_num += 1

                    x_rec_loss_avg /= iter_num
                    perceptual_loss_avg /= iter_num

                    dist.reduce(
                        x_rec_loss_avg, dst=0, op=dist.ReduceOp.SUM, group=gpu_group
                    )
                    dist.reduce(
                        perceptual_loss_avg,
                        dst=0,
                        op=dist.ReduceOp.SUM,
                        group=gpu_group,
                    )

                    if rank == 0:
                        x_rec_loss_avg = x_rec_loss_avg / args.ngpus
                        perceptual_loss_avg = perceptual_loss_avg / args.ngpus
                        x_rec_loss_avg_val = x_rec_loss_avg.item()
                        perceptual_loss_avg_val = perceptual_loss_avg.item()

                        print(
                            f"x_rec_loss_avg: {x_rec_loss_avg_val}, perceptual_loss_avg: {perceptual_loss_avg_val}"
                        )

                        print(
                            f"step={i}, epoch={epoch}, x_rec_loss_avg_val={x_rec_loss_avg_val}, perceptual_loss_avg_val={perceptual_loss_avg_val}, d_loss_val={d_loss_val}, indomainGAN_D_loss_val={indomainGAN_D_loss_val}, indomainGAN_E_loss_val={indomainGAN_E_loss_val}, x_rec_loss_val={x_rec_loss_val}, perceptual_loss_val={perceptual_loss_val}, g_loss_val={g_loss_val}, adv_loss_val={adv_loss_val}, w_rec_loss_val={w_rec_loss_val}, r1_val={r1_val}, real_score_val={real_score_val}, fake_score_val={fake_score_val}, latent_std={latent_std}, latent_channel_std={latent_channel_std}, latent_spatial_std={latent_spatial_std}"
                        )

                        torch.save(
                            {
                                "generator": model.module.generator.state_dict(),
                                "discriminator": model.module.discriminator.state_dict(),
                                "encoder": model.module.encoder.state_dict(),
                                "g_ema": g_ema_module.state_dict(),
                                "e_ema": e_ema_module.state_dict(),
                                "train_args": args,
                                "e_optim": e_optim.state_dict(),
                                "g_optim": g_optim.state_dict(),
                                "d_optim": d_optim.state_dict(),
                            },
                            f"{save_dir}/checkpoints/{str(i).zfill(6)}.pt",
                        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_lmdb", type=str)
    parser.add_argument("--val_lmdb", type=str)
    parser.add_argument("--ckpt", type=str)
    parser.add_argument("--suffix", type=str)
    parser.add_argument("--tensorboard_logdir", type=str, default='/home/uss00067/Experiments/StyleMapGAN/logs/')
    parser.add_argument("--list_path", type=str)

    parser.add_argument(
        "--dataset",
        type=str,
        default="celeba_hq",
        choices=[
            "celeba_hq",
            "afhq",
            "ffhq",
            "lsun/church_outdoor",
            "lsun/car",
            "lsun/bedroom",
        ],
    )
    parser.add_argument("--iter", type=int, default=5000) # 5000 training iters in total
    parser.add_argument("--save_network_interval", type=int, default=1000) # Save the checkpoint every 50 iters
    parser.add_argument("--small_generator", action="store_true")
    parser.add_argument("--batch", type=int, default=8, help="total batch sizes")
    parser.add_argument("--size", type=int, choices=[128, 256, 512, 1024], default=256)
    parser.add_argument("--r1", type=float, default=10)
    parser.add_argument("--d_reg_every", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--lr_mul", type=float, default=0.01)
    parser.add_argument("--channel_multiplier", type=int, default=2)
    parser.add_argument("--latent_channel_size", type=int, default=64)
    parser.add_argument("--latent_spatial_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument(
        "--normalize_mode",
        type=str,
        choices=["LayerNorm", "InstanceNorm2d", "BatchNorm2d", "GroupNorm"],
        default="LayerNorm",
    )
    parser.add_argument("--mapping_layer_num", type=int, default=8)

    parser.add_argument("--lambda_x_rec_loss", type=float, default=1)
    parser.add_argument("--lambda_adv_loss", type=float, default=1)
    parser.add_argument("--lambda_w_rec_loss", type=float, default=1)
    parser.add_argument("--lambda_d_loss", type=float, default=1)
    parser.add_argument("--lambda_perceptual_loss", type=float, default=0.1)
    parser.add_argument("--lambda_indomainGAN_D_loss", type=float, default=1)
    parser.add_argument("--lambda_indomainGAN_E_loss", type=float, default=0.01)

    input_args = parser.parse_args()

    input_args.tensorboard_logdir = input_args.tensorboard_logdir + input_args.suffix

    ngpus = torch.cuda.device_count()
    print("{} GPUS!".format(ngpus))

    assert input_args.batch % ngpus == 0
    input_args.batch_per_gpu = input_args.batch // ngpus
    input_args.ngpus = ngpus
    print("{} batch per gpu!".format(input_args.batch_per_gpu))

    run(ddp_main, ngpus, input_args)
