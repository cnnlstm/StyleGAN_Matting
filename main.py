import argparse
import math
import random
import os

import numpy as np
import torch
from torch import nn, autograd, optim
from torch.nn import functional as F
from torch.utils import data
from torchvision import transforms, utils, datasets
from tqdm import tqdm
from PIL import Image
import lpips
import tensorboardX
from collections import OrderedDict
import itertools

from model import Generator, Discriminator

from func import *
from evaluate import *
    
def mk_dir(paths):
    for _path in paths:
        if not os.path.exists(_path):
            os.makedirs(_path)


def cal_unknow_area(vector):
    return (vector==1).sum()
    
def min_entropy_loss(vector,loss_type='entropy'):
    if loss_type == "entropy":
        prob = F.softmax(vector,dim=1)
        return -(prob.log().mul(prob).sum(1).mean())
    else:
        return 0.4714-(F.softmax(vector,dim=1).std().mean())

def regression_loss(logit, target, weight=None, loss_type='l1',):
    if weight is None:
            if loss_type == 'l1':
                return F.l1_loss(logit, target)
            elif loss_type == 'l2':
                return F.mse_loss(logit, target)
            else:
                raise NotImplementedError("NotImplemented loss type {}".format(loss_type))
    else:
            if loss_type == 'l1':
                return F.l1_loss(logit * weight, target * weight, reduction='sum') / (torch.sum(weight) + 1e-8)
            elif loss_type == 'l2':
                return F.mse_loss(logit * weight, target * weight, reduction='sum') / (torch.sum(weight) + 1e-8)
            else:
                raise NotImplementedError("NotImplemented loss type {}".format(loss_type))

def synthesis(fore_image, bg_img, alpha):
    return  alpha * fore_image + (1. - alpha) * bg_img


def load_matting_model(args, model):
    if model == "gca":
        from gca_matting.generators import get_generator
        gca_model = get_generator(encoder='resnet_gca_encoder_29', decoder='res_gca_decoder_22')
        gca_ckpt = torch.load("gca_matting/checkpoints_finetune/gca-dist-all-data.pth")
        gca_model.load_state_dict(remove_prefix_state_dict(gca_ckpt['state_dict']), strict=True)
        gca_model.eval()
        return gca_model

    elif model == "indexnet":
        from indexnet_matting.hlmobilenetv2 import hlmobilenetv2
        indexnet_model = hlmobilenetv2(
            pretrained=False,
            freeze_bn=True, 
            output_stride=32,
            apply_aspp=True,
            conv_operator='std_conv',
            decoder='indexnet',
            decoder_kernel_size=5,
            indexnet='depthwise',
            index_mode='m2o',
            use_nonlinear=True,
            use_context=True
        )
        indexnet_ckpt = torch.load("indexnet_matting/ckpt/model_best.pth.tar")
        pretrained_dict = OrderedDict()
        for key, value in indexnet_ckpt['state_dict'].items():
            if 'module' in key:
                key = key[7:]
            pretrained_dict[key] = value
        indexnet_model.load_state_dict(pretrained_dict)
        indexnet_model.cuda()
        indexnet_model.eval()
        return indexnet_model

    elif model == "deepmatting":
        from deepmatting_matting.models import DIMModel
        ckpt = torch.load('deepmatting_matting/BEST_checkpoint.tar')
        deepmatting_model = DIMModel()
        deepmatting_model.load_state_dict(ckpt)
        deepmatting_model.cuda()
        deepmatting_model.eval()        
        return deepmatting_model

def load_trimap_model(args, model):
    if model == "deeplab":
        from deeplab_trimap.deeplab import DeepLab
        _nclass = 3
        _backbone = 'resnet'
        _out_stride = 16
        _sync_bn = False
        _freeze_bn = True
        _deeplab_ckpt_path = 'deeplab_trimap/checkpoint/deeplab_model_best.pth.tar' #'deeplab_trimap/deeplab_ckpt_best.pt'

        deeplab_model = DeepLab(num_classes=_nclass,
                            backbone=_backbone,
                            output_stride=_out_stride,
                            sync_bn=_sync_bn,
                            freeze_bn=_freeze_bn)

        deeplab_ckpt = torch.load(_deeplab_ckpt_path)
        deeplab_model.load_state_dict(deeplab_ckpt["state_dict"])
        deeplab_model.eval()
        return deeplab_model

                            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--iter', type=int, default=500)
    parser.add_argument('--size', type=int, default=1024)
    parser.add_argument('--ckpt', type=str, default='stylegan2-ffhq-config-f.pt')
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--channel_multiplier', type=int, default=2)
    parser.add_argument('--start_iter', type=int, default=0)
    parser.add_argument('--coco_root', type=str, default="dataset/coco_dataset")
    parser.add_argument('--result_dir', type=str, default="results")
    parser.add_argument('--name', type=str, default="default", help="sub dir in result_dir")
    parser.add_argument('--range', nargs=2, type=int, default=[0,-1], help="for multi-card inference")
    parser.add_argument('--mse', type=int, default=1)
    parser.add_argument('--e', type=int, default=1)
    parser.add_argument('--g', type=int, default=1)
    parser.add_argument('--optimize_choices', type=str,choices=['noise','latent','both'], default='both')
    parser.add_argument('--backbone', type=str,choices=['indexnet','gca','deepmatting'], default='indexnet')
    
    args = parser.parse_args()

    args.latent = 512
    args.n_mlp = 8
    args.device = 'cuda'
    
    torch.backends.cudnn.enabled = True
    
    # load stylegan2 model
    discriminator = Discriminator(args.size, channel_multiplier=args.channel_multiplier).to(args.device)
    g_ema = Generator(args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier).to(args.device)
    g_ema.eval()

    ckpt = torch.load(args.ckpt)
    discriminator.load_state_dict(ckpt['d'], strict=False)
    g_ema.load_state_dict(ckpt['g_ema'], strict=False)
    
    percept = lpips.PerceptualLoss(
            model='net-lin', net='vgg', use_gpu="cuda:0"
    )
    # load muiltiple models
    trimap_model = load_trimap_model(args, "deeplab")
    if args.backbone == 'indexnet':
        matting_model = load_matting_model(args, "indexnet")
    elif args.backbone == 'gca':
        matting_model = load_matting_model(args, "gca")
    elif args.backbone == 'deepmatting':
        matting_model = load_matting_model(args, "deepmatting")
    
    trimap_model.cuda()
    matting_model.cuda()
    
    # freeze parameter
    requires_grad(g_ema, False)
    requires_grad(discriminator, False)
    requires_grad(trimap_model, False)
    requires_grad(matting_model, False)
    
    print("models loaded...")
    
    transform = transforms.Compose(
                [
                    transforms.Resize((1024,1024),3),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                ]
    )

    # define coco dataset as background image
    coco_dataset = datasets.ImageFolder(root=args.coco_root, transform=transform)
    coco_loader = torch.utils.data.DataLoader(coco_dataset,batch_size=1,shuffle=True)
    coco_loader = iter(coco_loader)

    param_dir_path = "dataset/testing/param"

    # sort for muilt-card inference
    param_paths = []
    for param_path in os.listdir(param_dir_path):
        if 'pt' in param_path:
            param_paths.append(param_path)
    param_paths.sort()
    if args.range[1] == -1:
        param_paths = param_paths[args.range[0]:]
    else:
        param_paths = param_paths[args.range[0]:args.range[1]]
    
    for param_path in param_paths:
        param_file_name = os.path.join(param_dir_path, param_path)
        print("param_file_path : {}".format(param_file_name))
        
        gt_matte_path = param_file_name.replace("param","matte").replace("pt","png")
        print("gt_matte_path : {}".format(gt_matte_path))
        
        gt_img_path = param_file_name.replace("param","image").replace("pt","png")
        print("gt_img_path : {}".format(gt_img_path))


        fg_img_path = param_file_name.replace("param","foreground").replace("pt","png")
        print("fg_img_path : {}".format(fg_img_path))
        
        # get file name
        file_name = os.path.basename(param_file_name).split(".")[0]
        print("file_name : {}".format(file_name))
        
        # create dir to save result
        result_dir = os.path.join(args.result_dir, args.name)
        
        output_evaluate_path = os.path.join(result_dir, 'evaluate/{}.txt'.format(file_name))
        print("output_evaluate_path : {}".format(output_evaluate_path))

        output_matte_path = os.path.join(result_dir, 'matte/{}.png'.format(file_name))
        print("output_matte_path : {}".format(output_matte_path))

        output_img_path = os.path.join(result_dir, 'img/{}.png'.format(file_name))
        print("output_img_path : {}".format(output_img_path))
        
        output_synthesis_path = os.path.join(result_dir, 'synthesis/{}.png'.format(file_name))
        print("output_synthesis_path : {}".format(output_synthesis_path))
        
        output_trimap_path = os.path.join(result_dir, 'trimap/{}.png'.format(file_name))
        print("output_trimap_path : {}".format(output_trimap_path))

        mk_dir([os.path.dirname(output_evaluate_path), os.path.dirname(output_matte_path), os.path.dirname(output_img_path), 
                os.path.dirname(output_synthesis_path), os.path.dirname(output_trimap_path)])
        
        matte_list = []
        img_list = []
        synthesis_list = []
        trimap_list = []
        sad = []
        mse = []
        grad = []
        conn = []
        
        # get ground-truth image to compare
        gt_img = (transforms.ToTensor()(Image.open(gt_img_path)) - 0.5) * 2
        gt_img_resized = torch.nn.functional.interpolate(gt_img.unsqueeze(0), size=(1024, 1024), mode='nearest')
        gt_matte = transforms.ToTensor()(Image.open(gt_matte_path))
        

        fg_img = (transforms.ToTensor()(Image.open(fg_img_path)) - 0.5) * 2
        fg_img_resized = torch.nn.functional.interpolate(fg_img.unsqueeze(0), size=(1024, 1024), mode='nearest')


        img_list.append(gt_img)
        matte_list.append(gt_matte)

        # get latent code and noise
        param_file = torch.load(param_file_name)
        latent_in, noises = param_file['latent_in'], param_file['noises']

        # optim
        if args.optimize_choices == 'noise':
            latent_in.requires_grad = False
            for noise in noises:
                noise.require_grad = True
            optimizer_n = optim.Adam(noises, lr=args.lr)
                
        if args.optimize_choices == 'latent':
            latent_in.requires_grad = True
            for noise in noises:
                noise.require_grad = False
            optimizer_n = optim.Adam([latent_in], lr=args.lr)
        
        if args.optimize_choices == 'both':
            latent_in.requires_grad = True
            for noise in noises:
                noise.require_grad = True
            optimizer_n = optim.Adam([latent_in] + noises, lr=args.lr)
            
        optimizer_d = optim.Adam(discriminator.parameters(), lr=0.002)
        
        background_img = next(coco_loader)[0]

        # move to cuda
        gt_img = gt_img.cuda()
        gt_img_resized = gt_img_resized.cuda()

        fg_img = fg_img.cuda()
        fg_img_resized = fg_img_resized.cuda()
        
        gt_matte = gt_matte.cuda()
        background_img = background_img.cuda()
        latent_in = latent_in.cuda()
        for n in noises:
            n = n.cuda()
            
        # tensorboardx log
        logs_path = os.path.join(result_dir, 'logs')
        log_path = os.path.join(logs_path, file_name)
        
        mk_dir([log_path])
        train_writer = tensorboardX.SummaryWriter(log_path)
        
        # progress bar
        pbar = tqdm(range(1, args.iter+1), initial=args.start_iter, dynamic_ncols=True, smoothing=0.01)
        loss_dict = {}
        
        for idx in pbar:
            i = idx + args.start_iter
            if i > args.iter:
                print('Done!')
                break
            if idx == 1:
                
                gt_img_resized_ = torch.nn.functional.interpolate(gt_img_resized, size=(800, 608), mode='bicubic', )

                trimap_adaption, t_argmax = trimap_model(gt_img_resized_)
                trimap_onehot = F.one_hot(t_argmax.detach(), num_classes=3).permute(0, 3, 1, 2).float()
                t_argmax_resized = torch.nn.functional.interpolate(t_argmax.float().unsqueeze(0), size=(1024, 1024), mode='nearest', )
                
                gt_input = (gt_img_resized_ + 1) /2
                t = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                renormal_gt_img = t(gt_input.squeeze(0)).unsqueeze(0)
                renormal_trimap = t_argmax.float() / 2
                renormal_trimap = renormal_trimap.unsqueeze(0).cuda()
                
                if args.backbone == "gca":
                    alpha_adaption, _ = matting_model(renormal_gt_img, trimap_onehot.detach())
                else:
                    _in = torch.cat((renormal_gt_img, renormal_trimap.detach()),1)
                    alpha_adaption = matting_model(_in)
                
                if args.backbone == "deepmatting":
                    alpha_adaption_resize = torch.nn.functional.interpolate(alpha_adaption, size=(1024, 1024), mode='bicubic')
                    alpha_adaption_resize_3c = alpha_adaption_resize.repeat(1,3,1,1)
                    t_argmax_resized_3c = t_argmax_resized.repeat(1,3,1,1)
                    fg_img_resized[t_argmax_resized_3c == [2,2,2]] = fg_img_resized / (alpha_adaption_resize_3c + 0.000001)
                    fg_img_resized[t_argmax_resized_3c == [0,0,0]] = 0
                    background_img[t_argmax_resized_3c == [0,0,0]] = background_img / (1 - alpha_adaption_resize_3c + 0.000001)
                    background_img[t_argmax_resized_3c == [2,2,2]] = 0
                else:
                    alpha_adaption[t_argmax.unsqueeze(0) == 0] = 0
                    alpha_adaption[t_argmax.unsqueeze(0) == 2] = 1

                alpha_adaption_resize = torch.nn.functional.interpolate(alpha_adaption, size=(1024, 1024), mode='bicubic')

            else:

                # generate fake image
                fake_img, _ = g_ema([latent_in], input_is_latent=True, noise=noises)
                fake_img_resized = torch.nn.functional.interpolate(fake_img, size=(800, 608), mode='bicubic', )

                trimap_adaption, t_argmax = trimap_model(fake_img_resized)
                trimap_onehot = F.one_hot(t_argmax.detach(), num_classes=3).permute(0, 3, 1, 2).float()
                t_argmax_resized = torch.nn.functional.interpolate(t_argmax.float().unsqueeze(0), size=(1024, 1024), mode='nearest', )
                
                fake_input = (fake_img_resized + 1) /2
                t = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                renormal_fake_img = t(fake_input.squeeze(0)).unsqueeze(0)
                renormal_trimap = t_argmax.float() / 2
                renormal_trimap = renormal_trimap.unsqueeze(0).cuda()
                
                if args.backbone == "gca":
                    alpha_adaption, _ = matting_model(renormal_fake_img, trimap_onehot.detach())
                else:
                    _in = torch.cat((renormal_fake_img, renormal_trimap.detach()),1)
                    alpha_adaption = matting_model(_in)
                
                if args.backbone == "deepmatting":
                    alpha_adaption_resize = torch.nn.functional.interpolate(alpha_adaption, size=(1024, 1024), mode='bicubic')
                    alpha_adaption_resize_3c = alpha_adaption_resize.repeat(1,3,1,1)
                    t_argmax_resized_3c = t_argmax_resized.repeat(1,3,1,1)
                    fg_img_resized[t_argmax_resized_3c == [2,2,2]] = fg_img_resized / (alpha_adaption_resize_3c + 0.000001)
                    fg_img_resized[t_argmax_resized_3c == [0,0,0]] = 0
                    background_img[t_argmax_resized_3c == [0,0,0]] = background_img / (1 - alpha_adaption_resize_3c + 0.000001)
                    background_img[t_argmax_resized_3c == [2,2,2]] = 0
                else:
                    alpha_adaption[t_argmax.unsqueeze(0) == 0] = 0
                    alpha_adaption[t_argmax.unsqueeze(0) == 2] = 1

                alpha_adaption_resize = torch.nn.functional.interpolate(alpha_adaption, size=(1024, 1024), mode='bicubic')
                synthesis_img = alpha_adaption_resize * fg_img_resized.detach() + (1. - alpha_adaption_resize) * background_img.detach()

                # Train generator
                g_fake_pred = discriminator(synthesis_img)
                g_loss = F.softplus(-g_fake_pred).mean() * args.g
                
                # minimum entropy
                e_loss = min_entropy_loss(trimap_adaption) * args.e

                mse_loss = F.mse_loss(fake_img , gt_img_resized) * args.mse
                
                batch, channel, height, width = fake_img.shape
                if height > 256:
                    factor = height // 256
                    p_fake_img = fake_img.reshape(
                        batch, channel, height // factor, factor, width // factor, factor
                    )
                    p_fake_img = p_fake_img.mean([3, 5])
                    p_gt_img_resized = gt_img_resized.reshape(
                        batch, channel, height // factor, factor, width // factor, factor
                    )
                    p_gt_img_resized = p_gt_img_resized.mean([3, 5])


                p_loss = percept(p_fake_img, p_gt_img_resized).sum()
                
                n_loss = e_loss + mse_loss + p_loss + g_loss

                optimizer_n.zero_grad()
                n_loss.backward()
                optimizer_n.step()
                
                train_writer.add_scalar('p_loss', p_loss, idx)
                train_writer.add_scalar('g_loss', g_loss, idx)
                train_writer.add_scalar('e_loss', e_loss, idx)
                train_writer.add_scalar('mse_loss', mse_loss, idx)

                pbar.set_description(('Running'))
            
            if args.backbone == 'deepmatting':
                alpha_adaption[t_argmax.unsqueeze(0) == 0] = 0
                alpha_adaption[t_argmax.unsqueeze(0) == 2] = 1

            if idx == 1:
                gt_alpha = np.array(Image.open(gt_matte_path))
                h, w = gt_alpha.shape
                p_alpha = alpha_adaption.detach().permute(0, 2, 3, 1).squeeze(0).squeeze(2).to('cpu').numpy()
                p_alpha = cv.resize(p_alpha, dsize=(w,h), interpolation=cv.INTER_CUBIC)
                p_alpha = (np.clip(p_alpha, 0, 1))*255
                p_alpha = p_alpha.astype(np.uint8)

                # p_trimap = t_argmax.detach().clamp_(min=0, max=2).div_(2).mul(255).type(torch.uint8).permute(1, 2, 0).to('cpu').numpy().squeeze(2) 
                # p_trimap = cv.resize(p_trimap, (w, h), interpolation=cv.INTER_NEAREST)      
                mask = np.ones_like(p_alpha).astype(np.float32)

                sad. append(compute_sad_loss(p_alpha, gt_alpha, mask))
                mse. append(compute_mse_loss(p_alpha, gt_alpha, mask))
                grad.append(compute_gradient_loss(p_alpha, gt_alpha, mask))
                conn.append(compute_connectivity_loss(p_alpha, gt_alpha, mask))

                # Save evaluate result
                with open(output_evaluate_path, 'a') as f:
                    f.writelines("iter:{} ".format(idx))
                    f.writelines("sad:{} ".format(sad))
                    f.writelines("mse:{} ".format(mse))
                    f.writelines("grad:{} ".format(grad))
                    f.writelines("conn:{}\n".format(conn))
                    
                # clear list
                sad = []
                mse = []
                grad = []
                conn = []



            if idx % 50 == 1 and idx!=1 :
                _a = alpha_adaption.detach().cpu()
                _f = fake_img_resized.detach().cpu()
                _s = synthesis_img.detach().cpu()
                _t = t_argmax.type_as(alpha_adaption).detach().cpu().unsqueeze(0)
                
                _a = torch.nn.functional.interpolate(_a, size=(800, 600), mode='bicubic', )
                _f = torch.nn.functional.interpolate(_f, size=(800, 600), mode='bicubic', )
                _s = torch.nn.functional.interpolate(_s, size=(800, 600), mode='bicubic', )
                _t = torch.nn.functional.interpolate(_t, size=(800, 600), mode='nearest', )
                
                matte_list.append(_a.squeeze(0))
                img_list.append(_f.squeeze(0))
                synthesis_list.append(_s.squeeze(0))
                trimap_list.append(_t.squeeze(0))
                
                # evaluate
                gt_alpha = np.array(Image.open(gt_matte_path))
                h, w = gt_alpha.shape
                p_alpha = alpha_adaption.detach().permute(0, 2, 3, 1).squeeze(0).squeeze(2).to('cpu').numpy()
                p_alpha = cv.resize(p_alpha, dsize=(w,h), interpolation=cv.INTER_CUBIC)
                p_alpha = (np.clip(p_alpha, 0, 1))*255
                p_alpha = p_alpha.astype(np.uint8)

                # p_trimap = t_argmax.detach().clamp_(min=0, max=2).div_(2).mul(255).type(torch.uint8).permute(1, 2, 0).to('cpu').numpy().squeeze(2) 
                # p_trimap = cv.resize(p_trimap, (w, h), interpolation=cv.INTER_NEAREST)      
                mask = np.ones_like(p_alpha).astype(np.float32)

                sad. append(compute_sad_loss(p_alpha, gt_alpha, mask))
                mse. append(compute_mse_loss(p_alpha, gt_alpha, mask))
                grad.append(compute_gradient_loss(p_alpha, gt_alpha, mask))
                conn.append(compute_connectivity_loss(p_alpha, gt_alpha, mask))

                # Save evaluate result
                with open(output_evaluate_path, 'a') as f:
                    f.writelines("iter:{} ".format(idx))
                    f.writelines("sad:{} ".format(sad))
                    f.writelines("mse:{} ".format(mse))
                    f.writelines("grad:{} ".format(grad))
                    f.writelines("conn:{}\n".format(conn))
                    
                # clear list
                sad = []
                mse = []
                grad = []
                conn = []
                
        # Save image
        s_matte     = torch.cat((matte_list), 2)
        s_img       = torch.cat((img_list), 2)
        s_synthesis = torch.cat((synthesis_list), 2)
        s_trimap    = torch.cat((trimap_list), 2)
        
        utils.save_image(
            s_matte,
            output_matte_path,
            nrow=5,
            normalize=False,
            range=(0, 1),
        )
        utils.save_image(
            s_img,
            output_img_path,
            nrow=5,
            normalize=True,
            range=(-1, 1),
        )
        utils.save_image(
            s_synthesis,
            output_synthesis_path,
            nrow=5,
            normalize=True,
            range=(-1, 1),
        )
        utils.save_image(
            s_trimap,
            output_trimap_path,
            nrow=5,
            normalize=True,
            range=(0, 2),
        )