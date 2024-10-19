import warnings
warnings.filterwarnings('ignore')
import argparse
import logging
import math
import os
import random
import time
import sys
from copy import deepcopy
from pathlib import Path
from threading import Thread

import numpy as np
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
import yaml
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import train, test  # import test.py to get mAP after each epoch
from models.experimental import attempt_load
from models.yolo import Model, TSCODE_Detect, Yolov7_Tiny_E_ELAN_Attention, Yolov7_Tiny_E_ELAN_DCN, IDetect, IDetect_AFPN_P345, Yolov7_Tiny_E_ELAN_DBB, DiverseBranchBlock, \
    IDetect_Decoupled_DyHead, IDetect_Decoupled, ImplicitA, ImplicitM, IDetect_DyHead_Prune, DyHeadBlock_Prune
from timm.models.rexnet import LinearBottleneck
from models.fasternet import MLPBlock
from models.efficientViT import CascadedGroupAttention, LocalWindowAttention, EfficientViT
from models.convnextv2 import LayerNorm
from utils.autoanchor import check_anchors
from utils.datasets import create_dataloader
from utils.general import labels_to_class_weights, increment_path, labels_to_image_weights, init_seeds, \
    fitness, strip_optimizer, get_latest_run, check_dataset, check_file, check_git_status, check_img_size, \
    check_requirements, print_mutation, set_logging, one_cycle, colorstr
from utils.google_utils import attempt_download
from utils.loss import ComputeLoss, ComputeLossOTA
from utils.plots import plot_images, plot_labels, plot_results, plot_evolution
from utils.torch_utils import ModelEMA, select_device, intersect_dicts, torch_distributed_zero_first, is_parallel
from utils.wandb_logging.wandb_utils import WandbLogger, check_wandb_resume

logger = logging.getLogger(__name__)

import copy
import torch_pruning as tp
from functools import partial
from thop import clever_format
import matplotlib.pylab as plt
class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

from MyPrune import DiverseBranchBlockPruner, LayerNormPruner, DyHeadBlockPruner
def get_pruner(opt, model, example_inputs):
    sparsity_learning = False
    if opt.prune_method == "random":
        imp = tp.importance.RandomImportance()
        pruner_entry = partial(tp.pruner.MagnitudePruner, global_pruning=opt.global_pruning)
    elif opt.prune_method == "l1":
        # https://arxiv.org/abs/1608.08710
        imp = tp.importance.MagnitudeImportance(p=1)
        pruner_entry = partial(tp.pruner.MagnitudePruner, global_pruning=opt.global_pruning)
    elif opt.prune_method == "lamp":
        # https://arxiv.org/abs/2010.07611
        imp = tp.importance.LAMPImportance(p=2)
        pruner_entry = partial(tp.pruner.MagnitudePruner, global_pruning=opt.global_pruning)
    elif opt.prune_method == "slim":
        # https://arxiv.org/abs/1708.06519
        sparsity_learning = True
        imp = tp.importance.BNScaleImportance()
        pruner_entry = partial(tp.pruner.BNScalePruner, reg=opt.reg, global_pruning=opt.global_pruning)
    elif opt.prune_method == "group_slim":
        # https://tibshirani.su.domains/ftp/sparse-grlasso.pdf
        sparsity_learning = True
        imp = tp.importance.BNScaleImportance()
        pruner_entry = partial(tp.pruner.BNScalePruner, reg=opt.reg, global_pruning=opt.global_pruning, group_lasso=True)
    elif opt.prune_method == "group_norm":
        # https://openaccess.thecvf.com/content/CVPR2023/html/Fang_DepGraph_Towards_Any_Structural_Pruning_CVPR_2023_paper.html
        imp = tp.importance.GroupNormImportance(p=2)
        pruner_entry = partial(tp.pruner.GroupNormPruner, global_pruning=opt.global_pruning)
    elif opt.prune_method == "group_sl":
        # https://openaccess.thecvf.com/content/CVPR2023/html/Fang_DepGraph_Towards_Any_Structural_Pruning_CVPR_2023_paper.html
        sparsity_learning = True
        imp = tp.importance.GroupNormImportance(p=2)
        pruner_entry = partial(tp.pruner.GroupNormPruner, reg=opt.reg, global_pruning=opt.global_pruning)
    elif opt.prune_method == "growing_reg":
        # https://arxiv.org/abs/2012.09243
        sparsity_learning = True
        imp = tp.importance.GroupNormImportance(p=2)
        pruner_entry = partial(tp.pruner.GrowingRegPruner, reg=opt.reg, delta_reg=opt.delta_reg, global_pruning=opt.global_pruning)
    elif opt.prune_method == "group_hessian":
        # https://proceedings.neurips.cc/paper/1989/hash/6c9882bbac1c7093bd25041881277658-Abstract.html
        imp = tp.importance.HessianImportance(group_reduction='mean')
        pruner_entry = partial(tp.pruner.MagnitudePruner, global_pruning=opt.global_pruning)
    elif opt.prune_method == "group_taylor":
        # https://openaccess.thecvf.com/content_CVPR_2019/papers/Molchanov_Importance_Estimation_for_Neural_Network_Pruning_CVPR_2019_paper.pdf
        imp = tp.importance.TaylorImportance(group_reduction='mean')
        pruner_entry = partial(tp.pruner.MagnitudePruner, global_pruning=opt.global_pruning)
    else:
        raise NotImplementedError
    
    #args.is_accum_importance = is_accum_importance
    unwrapped_parameters = []
    ignored_layers = []
    pruning_ratio_dict = {}
    customized_pruners = {}
    round_to = None
    # ignore output layers
    
    # for cfg/training/yolov7-tiny.yaml
    ignored_layers.append(model.model[77])
    
    # for cfg/training/yolov7-tiny-prune.yaml
    # for k, m in model.named_modules():
    #     if isinstance(m, TSCODE_Detect):
    #         ignored_layers.append(m.m_cls)
    #         ignored_layers.append(m.m_reg)
    #         ignored_layers.append(m.m_conf)
    #     if isinstance(m, Yolov7_Tiny_E_ELAN_Attention):
    #         ignored_layers.append(m.att)
    
    # for cfg/training/yolov7-tiny-prune-dcn.yaml
    # for k, m in model.named_modules():
    #     if isinstance(m, Yolov7_Tiny_E_ELAN_DCN):
    #         ignored_layers.append(m.conv3)
    #         ignored_layers.append(m.conv4)
    #     if isinstance(m, IDetect_AFPN_P345):
    #         ignored_layers.append(m.afpn.body[0].asff_scalezero1)
    #         ignored_layers.append(m.afpn.body[0].asff_scaleone1)
    #         ignored_layers.append(m.afpn.body[0].asff_scalezero2)
    #         ignored_layers.append(m.afpn.body[0].asff_scaleone2)
    #         ignored_layers.append(m.afpn.body[0].asff_scaletwo2)
    #         ignored_layers.append(m.m)
    #         ignored_layers.append(m.ia)
    #         ignored_layers.append(m.im)
    
    # for cfg/training/yolov7-tiny-fasternet.yaml
    # customized_pruners[DiverseBranchBlock] = DiverseBranchBlockPruner()
    # for k, m in model.named_modules():
    #     if isinstance(m, IDetect):
    #         ignored_layers.append(m)
    #     if isinstance(m, MLPBlock):
    #         ignored_layers.append(m.mlp[-1])
    
    # for cfg/training/yolov7-tiny-rexnet.yaml
    # for k, m in model.named_modules():
    #     if isinstance(m, IDetect_Decoupled_DyHead):
    #         ignored_layers.append(m.dyhead)
    #         ignored_layers.append(m.ia_cls)
    #         ignored_layers.append(m.ia_reg)
    #         ignored_layers.append(m.ia_conf)
    #         ignored_layers.append(m.im_cls)
    #         ignored_layers.append(m.im_reg)
    #         ignored_layers.append(m.im_conf)
    #         ignored_layers.append(m.m_reg)
    #         ignored_layers.append(m.m_conf)
    #         ignored_layers.append(m.m_cls)
    #         # ignored_layers.append(m.m_reg_conf)
    #         ignored_layers.append(m.m_stem)
    #     if isinstance(m, LinearBottleneck):
    #         ignored_layers.append(m.conv_pwl)

    #     if isinstance(m, (ImplicitA, ImplicitM)):
    #         unwrapped_parameters.append((m.implicit, 1)) # pruning 1st dimension of implicit matrix
    
    # for cfg/training/yolov7-tiny-rexnet-decoupled.yaml
    # for k, m in model.named_modules():
    #     if isinstance(m, IDetect_Decoupled):
    #         ignored_layers.append(m.ia_cls)
    #         ignored_layers.append(m.ia_reg)
    #         ignored_layers.append(m.ia_conf)
    #         ignored_layers.append(m.im_cls)
    #         ignored_layers.append(m.im_reg)
    #         ignored_layers.append(m.im_conf)
    #         ignored_layers.append(m.m_reg)
    #         ignored_layers.append(m.m_conf)
    #         ignored_layers.append(m.m_cls)
    #         # ignored_layers.append(m.m_reg_conf)
    #         ignored_layers.append(m.m_stem)
    #     if isinstance(m, LinearBottleneck):
    #         ignored_layers.append(m.conv_pwl)

    #     if isinstance(m, (ImplicitA, ImplicitM)):
    #         unwrapped_parameters.append((m.implicit, 1)) # pruning 1st dimension of implicit matrix
    
    # for cfg/training/yolov7-tiny-dyhead-prune.yaml
    # customized_pruners[DyHeadBlock_Prune] = DyHeadBlockPruner()
    # for k, m in model.named_modules():
    #     if isinstance(m, IDetect_DyHead_Prune):
    #         ignored_layers.append(m.conv)
    #         ignored_layers.append(m.m)
    #         ignored_layers.append(m.ia)
    #         ignored_layers.append(m.im)
    
    print(ignored_layers)
    # Here we fix iterative_steps=200 to prune the model progressively with small steps 
    # until the required speed up is achieved.
    pruner = pruner_entry(
        model,
        example_inputs,
        importance=imp,
        iterative_steps=opt.iterative_steps,
        pruning_ratio=1.0,
        pruning_ratio_dict=pruning_ratio_dict,
        max_pruning_ratio=opt.max_sparsity,
        ignored_layers=ignored_layers,
        unwrapped_parameters=unwrapped_parameters,
        customized_pruners=customized_pruners,
        round_to=round_to,
        root_module_types=[nn.Conv2d, nn.Linear]
    )
    return sparsity_learning, imp, pruner

linear_trans = lambda epoch, epochs, reg, reg_ratio: (1 - epoch / (epochs - 1)) * (reg - reg_ratio) + reg_ratio

def model_prune(opt, model, imp, prune, example_inputs, testloader, imgsz_test, trainloader):
    N_batchs = 10
    # model.eval()
    base_model = copy.deepcopy(model)
    with HiddenPrints():
        ori_flops, ori_params = tp.utils.count_ops_and_params(base_model, example_inputs)
    ori_flops = ori_flops * 2.0
    ori_flops_f, ori_params_f = clever_format([ori_flops, ori_params], "%.3f")
    ori_result, _, _ = test.test(opt.data, None, batch_size=opt.batch_size * 2,
                                 imgsz=imgsz_test, plots=False, model=base_model, dataloader=testloader)
    ori_map50, ori_map = ori_result[2], ori_result[3]
    iter_idx, prune_flops = 0, ori_flops
    speed_up = 1.0
    logger.info('begin pruning...')
    while speed_up < opt.speed_up:
        model.train()
        loss_func = ComputeLoss(model)
        if isinstance(imp, tp.importance.HessianImportance):
            for k, (imgs, targets, paths, _) in enumerate(trainloader):
                if k >= N_batchs: break
                imgs = imgs.to(device, non_blocking=True).float() / 255
                output = model(imgs) 
                # compute loss for each sample
                loss = loss_func(output, targets.to(device))[0]
                imp.zero_grad() # clear accumulated gradients
                model.zero_grad() # clear gradients
                loss.backward(retain_graph=True) # simgle-sample gradient
                imp.accumulate_grad(model) # accumulate g^2
        elif isinstance(imp, tp.importance.TaylorImportance):
            for k, (imgs, targets, paths, _) in enumerate(trainloader):
                if k >= N_batchs: break
                imgs = imgs.to(device, non_blocking=True).float() / 255
                output = model(imgs)
                loss = loss_func(output, targets.to(device))[0]
                loss.backward()
        
        iter_idx += 1
        prune.step(interactive=False)
        prune_result, _, _ = test.test(opt.data, None, batch_size=opt.batch_size * 2,
                                 imgsz=imgsz_test, plots=False, model=model, dataloader=testloader)
        prune_map50, prune_map = prune_result[2], prune_result[3]
        with HiddenPrints():
            prune_flops, prune_params = tp.utils.count_ops_and_params(model, example_inputs)
        prune_flops = prune_flops * 2.0
        prune_flops_f, prune_params_f = clever_format([prune_flops, prune_params], "%.3f")
        speed_up = ori_flops / prune_flops # ori_model_GFLOPs / prune_model_GFLOPs
        logger.info(f'pruning... iter:{iter_idx} ori model flops:{ori_flops_f} => {prune_flops_f}({prune_flops / ori_flops * 100:.2f}%) params:{ori_params_f} => {prune_params_f}({prune_params / ori_params * 100:.2f}%) map@50:{ori_map50:.3f} => {prune_map50:.3f}({prune_map50 - ori_map50:.3f}) map@50:95:{ori_map:.3f} => {prune_map:.3f}({prune_map - ori_map:.3f}) Speed Up:{ori_flops / prune_flops:.2f}')
        
        if prune.current_step == prune.iterative_steps:
            break
    
    if isinstance(imp, tp.importance.HessianImportance):
        imp.zero_grad()
    model.zero_grad()
    torch.cuda.empty_cache()
     
    logger.info('pruning done...')
    logger.info(f'model flops:{ori_flops_f} => {prune_flops_f}({prune_flops / ori_flops * 100:.2f}%) Speed Up:{ori_flops / prune_flops:.2f}')
    logger.info(f'model params:{ori_params_f} => {prune_params_f}({prune_params / ori_params * 100:.2f}%)')
    logger.info(f'model map@50:{ori_map50:.3f} => {prune_map50:.3f}({prune_map50 - ori_map50:.3f})')
    logger.info(f'model map@50:95:{ori_map:.3f} => {prune_map:.3f}({prune_map - ori_map:.3f})')
    return model

def sparsity_learning_train(opt, model, prune, dataloader, testloader, device):
    save_dir, epochs, rank, batch_size, total_batch_size = increment_path(Path(opt.project) / f'{opt.name}_sl', exist_ok=opt.exist_ok | opt.evolve), opt.sl_epochs, opt.global_rank, opt.batch_size // 2, opt.total_batch_size // 2
    cuda = device.type != 'cpu'
    plots = not opt.evolve
    
    if type(save_dir) is str:
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(f'{save_dir}/visual', exist_ok=True)
        save_dir = Path(save_dir)
    
    # Directories
    wdir = save_dir / 'weights'
    wdir.mkdir(parents=True, exist_ok=True)  # make dir
    last = wdir / 'last.pt'
    best = wdir / 'best.pt'
    results_file = save_dir / 'results.txt'
    
    # Sparsity Learning Hyperparameters
    with open(opt.sl_hyp) as f:
        hyp = yaml.load(f, Loader=yaml.SafeLoader)  # load hyps
    
    # Optimizer
    nbs = 64  # nominal batch size
    accumulate = max(round(nbs / total_batch_size), 1)  # accumulate loss before optimizing
    hyp['weight_decay'] *= total_batch_size * accumulate / nbs  # scale weight_decay
    logger.info(f"Scaled weight_decay = {hyp['weight_decay']}")

    pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
    for k, v in model.named_modules():
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
            pg2.append(v.bias)  # biases
        if isinstance(v, nn.BatchNorm2d):
            pg0.append(v.weight)  # no decay
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
            if 'dbb_avg.bn' in k or 'dbb_1x1.bn' in k or 'dbb_1x1_kxk.bn1' in k:
                continue
            pg1.append(v.weight)  # apply decay
        if hasattr(v, 'im'):
            if hasattr(v.im, 'implicit'):           
                pg0.append(v.im.implicit)
            else:
                for iv in v.im:
                    pg0.append(iv.implicit)
        if hasattr(v, 'imc'):
            if hasattr(v.imc, 'implicit'):           
                pg0.append(v.imc.implicit)
            else:
                for iv in v.imc:
                    pg0.append(iv.implicit)
        if hasattr(v, 'imb'):
            if hasattr(v.imb, 'implicit'):           
                pg0.append(v.imb.implicit)
            else:
                for iv in v.imb:
                    pg0.append(iv.implicit)
        if hasattr(v, 'imo'):
            if hasattr(v.imo, 'implicit'):           
                pg0.append(v.imo.implicit)
            else:
                for iv in v.imo:
                    pg0.append(iv.implicit)
        if hasattr(v, 'ia'):
            if hasattr(v.ia, 'implicit'):           
                pg0.append(v.ia.implicit)
            else:
                for iv in v.ia:
                    pg0.append(iv.implicit)
        if hasattr(v, 'attn'):
            if hasattr(v.attn, 'logit_scale'):   
                pg0.append(v.attn.logit_scale)
            if hasattr(v.attn, 'q_bias'):   
                pg0.append(v.attn.q_bias)
            if hasattr(v.attn, 'v_bias'):  
                pg0.append(v.attn.v_bias)
            if hasattr(v.attn, 'relative_position_bias_table'):  
                pg0.append(v.attn.relative_position_bias_table)
        if hasattr(v, 'rbr_dense'):
            if hasattr(v.rbr_dense, 'weight_rbr_origin'):  
                pg0.append(v.rbr_dense.weight_rbr_origin)
            if hasattr(v.rbr_dense, 'weight_rbr_avg_conv'): 
                pg0.append(v.rbr_dense.weight_rbr_avg_conv)
            if hasattr(v.rbr_dense, 'weight_rbr_pfir_conv'):  
                pg0.append(v.rbr_dense.weight_rbr_pfir_conv)
            if hasattr(v.rbr_dense, 'weight_rbr_1x1_kxk_idconv1'): 
                pg0.append(v.rbr_dense.weight_rbr_1x1_kxk_idconv1)
            if hasattr(v.rbr_dense, 'weight_rbr_1x1_kxk_conv2'):   
                pg0.append(v.rbr_dense.weight_rbr_1x1_kxk_conv2)
            if hasattr(v.rbr_dense, 'weight_rbr_gconv_dw'):   
                pg0.append(v.rbr_dense.weight_rbr_gconv_dw)
            if hasattr(v.rbr_dense, 'weight_rbr_gconv_pw'):   
                pg0.append(v.rbr_dense.weight_rbr_gconv_pw)
            if hasattr(v.rbr_dense, 'vector'):   
                pg0.append(v.rbr_dense.vector)

    if opt.adam:
        optimizer = optim.Adam(pg0, lr=hyp['lr0'], betas=(hyp['momentum'], 0.999))  # adjust beta1 to momentum
    else:
        optimizer = optim.SGD(pg0, lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)

    optimizer.add_param_group({'params': pg1, 'weight_decay': hyp['weight_decay']})  # add pg1 with weight_decay
    optimizer.add_param_group({'params': pg2})  # add pg2 (biases)
    logger.info('Optimizer groups: %g .bias, %g conv.weight, %g other' % (len(pg2), len(pg1), len(pg0)))
    del pg0, pg1, pg2

    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    # https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html#OneCycleLR
    if opt.linear_lr:
        lf = lambda x: (1 - x / (epochs - 1)) * (1.0 - hyp['lrf']) + hyp['lrf']  # linear
    else:
        lf = one_cycle(1, hyp['lrf'], epochs)  # cosine 1->hyp['lrf']
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    # plot_lr_scheduler(optimizer, scheduler, epochs)

    # EMA
    ema = ModelEMA(model) if rank in [-1, 0] else None
    
    start_epoch, best_fitness, best_sl = 0, 0.0, {}
    
    # Image sizes
    gs = max(int(model.stride.max()), 32)  # grid size (max stride)
    nl = model.model[-1].nl  # number of detection layers (used for scaling hyp['obj'])
    imgsz, imgsz_test = [check_img_size(x, gs) for x in opt.img_size]  # verify imgsz are gs-multiples

    # DP mode
    if cuda and rank == -1 and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    
    # SyncBatchNorm
    if opt.sync_bn and cuda and rank != -1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
        logger.info('Using SyncBatchNorm()')
    
    # DDP mode
    if cuda and rank != -1:
        model = DDP(model, device_ids=[opt.local_rank], output_device=opt.local_rank,
                    # nn.MultiheadAttention incompatibility with DDP https://github.com/pytorch/pytorch/issues/26698
                    find_unused_parameters=any(isinstance(layer, nn.MultiheadAttention) for layer in model.modules()))

    # Model parameters
    hyp['box'] *= 3. / nl  # scale to layers
    hyp['cls'] *= nc / 80. * 3. / nl  # scale to classes and layers
    hyp['obj'] *= (imgsz / 640) ** 2 * 3. / nl  # scale to image size and layers
    hyp['label_smoothing'] = opt.label_smoothing
    model.nc = nc  # attach number of classes to model
    model.hyp = hyp  # attach hyperparameters to model
    model.gr = 1.0  # iou loss ratio (obj_loss = 1.0 or iou)
    model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc  # attach class weights
    model.names = names
    
    # Start training
    t0 = time.time()
    nw = max(round(hyp['warmup_epochs'] * nb), 1000)  # number of warmup iterations, max(3 epochs, 1k iterations)
    # nw = min(nw, (epochs - start_epoch) / 2 * nb)  # limit warmup to < 1/2 of training
    maps = np.zeros(nc)  # mAP per class
    results = (0, 0, 0, 0, 0, 0, 0)  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
    scheduler.last_epoch = start_epoch - 1  # do not move
    scaler = amp.GradScaler(enabled=cuda)
    compute_loss_ota = ComputeLossOTA(model)  # init loss class
    compute_loss = ComputeLoss(model)  # init loss class
    logger.info(f'Image sizes {imgsz} train, {imgsz_test} test\n'
                f'Using {dataloader.num_workers} dataloader workers\n'
                f'Logging results to {save_dir}\n'
                f'Starting training for {epochs} epochs...')

    for epoch in range(start_epoch, epochs):  # epoch ------------------------------------------------------------------
        model.train()

        # Update image weights (optional)
        if opt.image_weights:
            # Generate indices
            if rank in [-1, 0]:
                cw = model.class_weights.cpu().numpy() * (1 - maps) ** 2 / nc  # class weights
                iw = labels_to_image_weights(dataset.labels, nc=nc, class_weights=cw)  # image weights
                dataset.indices = random.choices(range(dataset.n), weights=iw, k=dataset.n)  # rand weighted idx
            # Broadcast if DDP
            if rank != -1:
                indices = (torch.tensor(dataset.indices) if rank == 0 else torch.zeros(dataset.n)).int()
                dist.broadcast(indices, 0)
                if rank != 0:
                    dataset.indices = indices.cpu().numpy()

        # Update mosaic border
        # b = int(random.uniform(0.25 * imgsz, 0.75 * imgsz + gs) // gs * gs)
        # dataset.mosaic_border = [b - imgsz, -b]  # height, width borders

        mloss = torch.zeros(4, device=device)  # mean losses
        if rank != -1:
            dataloader.sampler.set_epoch(epoch)
        pbar = enumerate(dataloader)
        logger.info(('\n' + '%10s' * 8) % ('Epoch', 'gpu_mem', 'box', 'obj', 'cls', 'total', 'labels', 'img_size'))
        if rank in [-1, 0]:
            pbar = tqdm(pbar, total=nb)  # progress bar
        optimizer.zero_grad()
        for i, (imgs, targets, paths, _) in pbar:  # batch -------------------------------------------------------------
            ni = i + nb * epoch  # number integrated batches (since train start)
            imgs = imgs.to(device, non_blocking=True).float() / 255.0  # uint8 to float32, 0-255 to 0.0-1.0

            # Warmup
            if ni <= nw:
                xi = [0, nw]  # x interp
                # model.gr = np.interp(ni, xi, [0.0, 1.0])  # iou loss ratio (obj_loss = 1.0 or iou)
                accumulate = max(1, np.interp(ni, xi, [1, nbs / total_batch_size]).round())
                for j, x in enumerate(optimizer.param_groups):
                    # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                    x['lr'] = np.interp(ni, xi, [hyp['warmup_bias_lr'] if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(ni, xi, [hyp['warmup_momentum'], hyp['momentum']])

            # Multi-scale
            if opt.multi_scale:
                sz = random.randrange(imgsz * 0.5, imgsz * 1.5 + gs) // gs * gs  # size
                sf = sz / max(imgs.shape[2:])  # scale factor
                if sf != 1:
                    ns = [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]]  # new shape (stretched to gs-multiple)
                    imgs = F.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)

            # Forward
            # with amp.autocast(enabled=cuda):
            pred = model(imgs)  # forward
            if 'loss_ota' not in hyp or hyp['loss_ota'] == 1:
                loss, loss_items = compute_loss_ota(pred, targets.to(device), imgs)  # loss scaled by batch_size
            else:
                loss, loss_items = compute_loss(pred, targets.to(device))  # loss scaled by batch_size
            if rank != -1:
                loss *= opt.world_size  # gradient averaged between devices in DDP mode
            if opt.quad:
                loss *= 4.

            # Backward
            # scaler.scale(loss).backward()
            loss.backward()
            
            # with amp.autocast(enabled=cuda):
            if isinstance(prune, (tp.pruner.BNScalePruner,)):
                if opt.reg_decay_type == 'linear':
                    reg = linear_trans(epoch, epochs, opt.reg, opt.reg * opt.reg_decay)
                elif opt.reg_decay_type == 'step':
                    reg = opt.reg * (opt.reg_decay ** (epoch // opt.reg_decay_step))
                elif opt.opt.reg_decay_type == 'constant':
                    reg = opt.reg
                prune.regularize(model, reg=reg)
            elif isinstance(prune, (tp.pruner.GroupNormPruner, tp.pruner.GrowingRegPruner)):
                reg = opt.reg
                prune.regularize(model)

            # Optimize
            if ni % accumulate == 0:
                # scaler.step(optimizer)  # optimizer.step
                # scaler.update()
                optimizer.step()
                optimizer.zero_grad()
                if ema:
                    ema.update(model)

            # Print
            if rank in [-1, 0]:
                mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
                mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
                s = ('%10s' * 2 + '%10.4g' * 6) % (
                    '%g/%g' % (epoch, epochs - 1), mem, *mloss, targets.shape[0], imgs.shape[-1])
                pbar.set_description(s)

                # Plot
                if plots and ni < 10:
                    f = save_dir / f'train_batch{ni}.jpg'  # filename
                    Thread(target=plot_images, args=(imgs, targets, paths, f), daemon=True).start()
                    # if tb_writer:
                    #     tb_writer.add_image(f, result, dataformats='HWC', global_step=epoch)
                    #     tb_writer.add_graph(torch.jit.trace(model, imgs, strict=False), [])  # add model graph

            # end batch ------------------------------------------------------------------------------------------------
        # end epoch ----------------------------------------------------------------------------------------------------
        
        if isinstance(prune, (tp.pruner.GrowingRegPruner,)):
            prune.update_reg()
        
        if ema:
            model_sl = ema.ema.state_dict()
        else:
            model_sl = model.state_dict()
        
        tb = SummaryWriter(save_dir)
        
        bn_weight = []
        for name in model_sl:
            if 'weight' in name and len(model_sl[name].size()) == 1:
                weight = model_sl[name].data.cpu().abs().clone().numpy().reshape((-1))
                bn_weight.append(weight)
        bn_weight = np.concatenate(bn_weight)
        bn_weight = np.sort(bn_weight)
        bn_weight_percent = np.percentile(bn_weight, [1, 5, 10, 25, 50, 75])
        sparsity_ratio = np.sum(bn_weight < 1e-6) / bn_weight.shape[0]
        if f'{sparsity_ratio:.3f}' not in best_sl:
            best_sl[f'{sparsity_ratio:.3f}'] = 0.0
        if tb:
            tb.add_histogram('hist', bn_weight, epoch, bins='doane')
        
        del model_sl
            
        plt.figure(figsize=(15, 5), clear=True)
        plt.plot(bn_weight)
        plt.title(f'sparsity_ratio:{sparsity_ratio:.3f}\n')
        plt.tight_layout()
        plt.savefig(f'{save_dir}/visual/{epoch}_sl_{sparsity_ratio:.3f}.png')
        plt.close()
        logger.info(f'epoch:{epoch} reg:{reg:.5f} sparsity_ratio:{sparsity_ratio:.5f} bn_weight_1:{bn_weight_percent[0]:.10f} bn_weight_5:{bn_weight_percent[1]:.8f} bn_weight_10:{bn_weight_percent[2]:.8f}\nbn_weight_25:{bn_weight_percent[3]:.5f} bn_weight_50:{bn_weight_percent[4]:.5f} bn_weight_75:{bn_weight_percent[5]:.5f}')
        
        # Scheduler
        lr = [x['lr'] for x in optimizer.param_groups]  # for tensorboard
        scheduler.step()

        # DDP process 0 or single-GPU
        if rank in [-1, 0]:
            # mAP
            ema.update_attr(model, include=['yaml', 'nc', 'hyp', 'gr', 'names', 'stride', 'class_weights'])
            final_epoch = epoch + 1 == epochs
            if not opt.notest or final_epoch:  # Calculate mAP
                results, maps, times = test.test(data_dict,
                                                 batch_size=batch_size * 2,
                                                 imgsz=imgsz_test,
                                                 model=ema.ema,
                                                 single_cls=opt.single_cls,
                                                 dataloader=testloader,
                                                 save_dir=save_dir,
                                                 verbose=nc < 50 and final_epoch,
                                                 plots=plots and final_epoch,
                                                 compute_loss=compute_loss,
                                                 is_coco=is_coco,
                                                 v5_metric=opt.v5_metric)

            # Write
            with open(results_file, 'a') as f:
                f.write(s + '%10.4g' * 7 % results + '\n')  # append metrics, val_loss
            if len(opt.name) and opt.bucket:
                os.system('gsutil cp %s gs://%s/results/results%s.txt' % (results_file, opt.bucket, opt.name))

            # Log
            tags = ['train/box_loss', 'train/obj_loss', 'train/cls_loss',  # train loss
                    'metrics/precision', 'metrics/recall', 'metrics/mAP_0.5', 'metrics/mAP_0.5:0.95',
                    'val/box_loss', 'val/obj_loss', 'val/cls_loss',  # val loss
                    'x/lr0', 'x/lr1', 'x/lr2']  # params

            # Update best mAP
            fi = fitness(np.array(results).reshape(1, -1))  # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
            if fi > best_fitness:
                best_fitness = fi
            if fi > best_sl[f'{sparsity_ratio:.3f}']:
                best_sl[f'{sparsity_ratio:.3f}'] = fi

            # Save model
            if (not opt.nosave) or (final_epoch and not opt.evolve):  # if save
                ckpt = {'epoch': epoch,
                        'best_fitness': best_fitness,
                        'training_results': results_file.read_text(),
                        'model': deepcopy(model.module if is_parallel(model) else model).half(),
                        'ema': deepcopy(ema.ema).half(),
                        'updates': ema.updates,
                        'optimizer': optimizer.state_dict(),
                        'wandb_id': None}

                # Save last, best and delete
                torch.save(ckpt, last)
                if best_fitness == fi:
                    torch.save(ckpt, best)
                if best_sl[f'{sparsity_ratio:.3f}'] == fi:
                    torch.save(ckpt, wdir / 'best_sl_{:.3f}.pt'.format(sparsity_ratio))
                # if (best_fitness == fi) and (epoch >= 200):
                #     torch.save(ckpt, wdir / 'best_{:03d}.pt'.format(epoch))
                # if epoch == 0:
                #     torch.save(ckpt, wdir / 'epoch_{:03d}.pt'.format(epoch))
                # elif ((epoch+1) % 25) == 0:
                #     torch.save(ckpt, wdir / 'epoch_{:03d}.pt'.format(epoch))
                # elif epoch >= (epochs-5):
                #     torch.save(ckpt, wdir / 'epoch_{:03d}.pt'.format(epoch))
                del ckpt

        # end epoch ----------------------------------------------------------------------------------------------------
    # end training
    
    sl = sorted(best_sl.keys(), key=lambda x:float(x))[-1]
    best_sl_model = wdir / 'best_sl_{}.pt'.format(sl)
    
    if rank in [-1, 0]:
        # Plots
        if plots:
            plot_results(save_dir=save_dir)  # save as results.png
        # Test best.pt
        logger.info('%g epochs completed in %.3f hours.\n' % (epoch - start_epoch + 1, (time.time() - t0) / 3600))
        if opt.data.endswith('coco.yaml') and nc == 80:  # if COCO
            # for m in (last, best, best_sl_model) if best.exists() else (last):  # speed, mAP tests
            results, _, _ = test.test(opt.data,
                                        batch_size=batch_size * 2,
                                        imgsz=imgsz_test,
                                        conf_thres=0.001,
                                        iou_thres=0.7,
                                        model=attempt_load(best_sl_model, device).half(), 
                                        single_cls=opt.single_cls,
                                        dataloader=testloader,
                                        save_dir=save_dir,
                                        save_json=True,
                                        plots=False,
                                        is_coco=is_coco,
                                        v5_metric=opt.v5_metric)

        # Strip optimizers
        final = best if best.exists() else last  # final model
        for f in last, best:
            if f.exists():
                strip_optimizer(f)  # strip optimizers
        if opt.bucket:
            os.system(f'gsutil cp {final} gs://{opt.bucket}/weights')  # upload
    else:
        dist.destroy_process_group()
    torch.cuda.empty_cache()
    
    return best_sl_model

def finetune(opt, model, dataloader, testloader, device):
    save_dir, epochs, rank, batch_size, total_batch_size = increment_path(Path(opt.project) / f'{opt.name}', exist_ok=opt.exist_ok | opt.evolve), opt.epochs, opt.global_rank, opt.batch_size, opt.total_batch_size
    cuda = device.type != 'cpu'
    plots = not opt.evolve
    
    if type(save_dir) is str:
        os.makedirs(save_dir, exist_ok=True)
        save_dir = Path(save_dir)
    
    # Directories
    wdir = save_dir / 'weights'
    wdir.mkdir(parents=True, exist_ok=True)  # make dir
    last = wdir / 'last.pt'
    best = wdir / 'best.pt'
    prune = wdir / 'prune.pt'
    results_file = save_dir / 'results.txt'
    
    # save prune model before finetune
    prune_model = deepcopy(model.module if is_parallel(model) else model)
    for p in prune_model.parameters():
        p.requires_grad_(False)
    ckpt = {'model': prune_model.half(),
            'ema': None}
    torch.save(ckpt, prune)
    logger.info(colorstr(f'Pruning after Finetune before the model is saved in:{prune}'))
    
    # Sparsity Learning Hyperparameters
    with open(opt.hyp) as f:
        hyp = yaml.load(f, Loader=yaml.SafeLoader)  # load hyps
    logger.info(colorstr('finetune hyperparameters: ') + ', '.join(f'{k}={v}' for k, v in hyp.items()))
    
    # Optimizer
    nbs = 64  # nominal batch size
    accumulate = max(round(nbs / total_batch_size), 1)  # accumulate loss before optimizing
    hyp['weight_decay'] *= total_batch_size * accumulate / nbs  # scale weight_decay
    logger.info(f"Scaled weight_decay = {hyp['weight_decay']}")

    pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
    for k, v in model.named_modules():
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
            pg2.append(v.bias)  # biases
        if isinstance(v, nn.BatchNorm2d):
            pg0.append(v.weight)  # no decay
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
            if 'dbb_avg.bn' in k or 'dbb_1x1.bn' in k or 'dbb_1x1_kxk.bn1' in k:
                continue
            pg1.append(v.weight)  # apply decay
        if hasattr(v, 'im'):
            if hasattr(v.im, 'implicit'):           
                pg0.append(v.im.implicit)
            else:
                for iv in v.im:
                    pg0.append(iv.implicit)
        if hasattr(v, 'imc'):
            if hasattr(v.imc, 'implicit'):           
                pg0.append(v.imc.implicit)
            else:
                for iv in v.imc:
                    pg0.append(iv.implicit)
        if hasattr(v, 'imb'):
            if hasattr(v.imb, 'implicit'):           
                pg0.append(v.imb.implicit)
            else:
                for iv in v.imb:
                    pg0.append(iv.implicit)
        if hasattr(v, 'imo'):
            if hasattr(v.imo, 'implicit'):           
                pg0.append(v.imo.implicit)
            else:
                for iv in v.imo:
                    pg0.append(iv.implicit)
        if hasattr(v, 'ia'):
            if hasattr(v.ia, 'implicit'):           
                pg0.append(v.ia.implicit)
            else:
                for iv in v.ia:
                    pg0.append(iv.implicit)
        if hasattr(v, 'attn'):
            if hasattr(v.attn, 'logit_scale'):   
                pg0.append(v.attn.logit_scale)
            if hasattr(v.attn, 'q_bias'):   
                pg0.append(v.attn.q_bias)
            if hasattr(v.attn, 'v_bias'):  
                pg0.append(v.attn.v_bias)
            if hasattr(v.attn, 'relative_position_bias_table'):  
                pg0.append(v.attn.relative_position_bias_table)
        if hasattr(v, 'rbr_dense'):
            if hasattr(v.rbr_dense, 'weight_rbr_origin'):  
                pg0.append(v.rbr_dense.weight_rbr_origin)
            if hasattr(v.rbr_dense, 'weight_rbr_avg_conv'): 
                pg0.append(v.rbr_dense.weight_rbr_avg_conv)
            if hasattr(v.rbr_dense, 'weight_rbr_pfir_conv'):  
                pg0.append(v.rbr_dense.weight_rbr_pfir_conv)
            if hasattr(v.rbr_dense, 'weight_rbr_1x1_kxk_idconv1'): 
                pg0.append(v.rbr_dense.weight_rbr_1x1_kxk_idconv1)
            if hasattr(v.rbr_dense, 'weight_rbr_1x1_kxk_conv2'):   
                pg0.append(v.rbr_dense.weight_rbr_1x1_kxk_conv2)
            if hasattr(v.rbr_dense, 'weight_rbr_gconv_dw'):   
                pg0.append(v.rbr_dense.weight_rbr_gconv_dw)
            if hasattr(v.rbr_dense, 'weight_rbr_gconv_pw'):   
                pg0.append(v.rbr_dense.weight_rbr_gconv_pw)
            if hasattr(v.rbr_dense, 'vector'):   
                pg0.append(v.rbr_dense.vector)

    if opt.adam:
        optimizer = optim.Adam(pg0, lr=hyp['lr0'], betas=(hyp['momentum'], 0.999))  # adjust beta1 to momentum
    else:
        optimizer = optim.SGD(pg0, lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)

    optimizer.add_param_group({'params': pg1, 'weight_decay': hyp['weight_decay']})  # add pg1 with weight_decay
    optimizer.add_param_group({'params': pg2})  # add pg2 (biases)
    logger.info('Optimizer groups: %g .bias, %g conv.weight, %g other' % (len(pg2), len(pg1), len(pg0)))
    del pg0, pg1, pg2

    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    # https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html#OneCycleLR
    if opt.linear_lr:
        lf = lambda x: (1 - x / (epochs - 1)) * (1.0 - hyp['lrf']) + hyp['lrf']  # linear
    else:
        lf = one_cycle(1, hyp['lrf'], epochs)  # cosine 1->hyp['lrf']
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    # plot_lr_scheduler(optimizer, scheduler, epochs)

    # EMA
    ema = ModelEMA(model) if rank in [-1, 0] else None
    
    start_epoch, best_fitness = 0, 0.0
    
    # Image sizes
    gs = max(int(model.stride.max()), 32)  # grid size (max stride)
    nl = model.model[-1].nl  # number of detection layers (used for scaling hyp['obj'])
    imgsz, imgsz_test = [check_img_size(x, gs) for x in opt.img_size]  # verify imgsz are gs-multiples

    # DP mode
    if cuda and rank == -1 and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    
    # SyncBatchNorm
    if opt.sync_bn and cuda and rank != -1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
        logger.info('Using SyncBatchNorm()')
    
    # DDP mode
    if cuda and rank != -1:
        model = DDP(model, device_ids=[opt.local_rank], output_device=opt.local_rank,
                    # nn.MultiheadAttention incompatibility with DDP https://github.com/pytorch/pytorch/issues/26698
                    find_unused_parameters=any(isinstance(layer, nn.MultiheadAttention) for layer in model.modules()))

    # Model parameters
    hyp['box'] *= 3. / nl  # scale to layers
    hyp['cls'] *= nc / 80. * 3. / nl  # scale to classes and layers
    hyp['obj'] *= (imgsz / 640) ** 2 * 3. / nl  # scale to image size and layers
    hyp['label_smoothing'] = opt.label_smoothing
    model.nc = nc  # attach number of classes to model
    model.hyp = hyp  # attach hyperparameters to model
    model.gr = 1.0  # iou loss ratio (obj_loss = 1.0 or iou)
    model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc  # attach class weights
    model.names = names
    
    # Start training
    t0 = time.time()
    nw = max(round(hyp['warmup_epochs'] * nb), 1000)  # number of warmup iterations, max(3 epochs, 1k iterations)
    # nw = min(nw, (epochs - start_epoch) / 2 * nb)  # limit warmup to < 1/2 of training
    maps = np.zeros(nc)  # mAP per class
    results = (0, 0, 0, 0, 0, 0, 0)  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
    scheduler.last_epoch = start_epoch - 1  # do not move
    scaler = amp.GradScaler(enabled=cuda)
    compute_loss_ota = ComputeLossOTA(model)  # init loss class
    compute_loss = ComputeLoss(model)  # init loss class
    logger.info(f'Image sizes {imgsz} train, {imgsz_test} test\n'
                f'Using {dataloader.num_workers} dataloader workers\n'
                f'Logging results to {save_dir}\n'
                f'Starting training for {epochs} epochs...')

    for epoch in range(start_epoch, epochs):  # epoch ------------------------------------------------------------------
        model.train()

        # Update image weights (optional)
        if opt.image_weights:
            # Generate indices
            if rank in [-1, 0]:
                cw = model.class_weights.cpu().numpy() * (1 - maps) ** 2 / nc  # class weights
                iw = labels_to_image_weights(dataset.labels, nc=nc, class_weights=cw)  # image weights
                dataset.indices = random.choices(range(dataset.n), weights=iw, k=dataset.n)  # rand weighted idx
            # Broadcast if DDP
            if rank != -1:
                indices = (torch.tensor(dataset.indices) if rank == 0 else torch.zeros(dataset.n)).int()
                dist.broadcast(indices, 0)
                if rank != 0:
                    dataset.indices = indices.cpu().numpy()

        # Update mosaic border
        # b = int(random.uniform(0.25 * imgsz, 0.75 * imgsz + gs) // gs * gs)
        # dataset.mosaic_border = [b - imgsz, -b]  # height, width borders

        mloss = torch.zeros(4, device=device)  # mean losses
        if rank != -1:
            dataloader.sampler.set_epoch(epoch)
        pbar = enumerate(dataloader)
        logger.info(('\n' + '%10s' * 8) % ('Epoch', 'gpu_mem', 'box', 'obj', 'cls', 'total', 'labels', 'img_size'))
        if rank in [-1, 0]:
            pbar = tqdm(pbar, total=nb)  # progress bar
        optimizer.zero_grad()
        for i, (imgs, targets, paths, _) in pbar:  # batch -------------------------------------------------------------
            ni = i + nb * epoch  # number integrated batches (since train start)
            imgs = imgs.to(device, non_blocking=True).float() / 255.0  # uint8 to float32, 0-255 to 0.0-1.0

            # Warmup
            if ni <= nw:
                xi = [0, nw]  # x interp
                # model.gr = np.interp(ni, xi, [0.0, 1.0])  # iou loss ratio (obj_loss = 1.0 or iou)
                accumulate = max(1, np.interp(ni, xi, [1, nbs / total_batch_size]).round())
                for j, x in enumerate(optimizer.param_groups):
                    # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                    x['lr'] = np.interp(ni, xi, [hyp['warmup_bias_lr'] if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(ni, xi, [hyp['warmup_momentum'], hyp['momentum']])

            # Multi-scale
            if opt.multi_scale:
                sz = random.randrange(imgsz * 0.5, imgsz * 1.5 + gs) // gs * gs  # size
                sf = sz / max(imgs.shape[2:])  # scale factor
                if sf != 1:
                    ns = [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]]  # new shape (stretched to gs-multiple)
                    imgs = F.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)

            # Forward
            with amp.autocast(enabled=cuda):
                pred = model(imgs)  # forward
                if 'loss_ota' not in hyp or hyp['loss_ota'] == 1:
                    loss, loss_items = compute_loss_ota(pred, targets.to(device), imgs)  # loss scaled by batch_size
                else:
                    loss, loss_items = compute_loss(pred, targets.to(device))  # loss scaled by batch_size
                if rank != -1:
                    loss *= opt.world_size  # gradient averaged between devices in DDP mode
                if opt.quad:
                    loss *= 4.

            # Backward
            scaler.scale(loss).backward()
            # Optimize
            if ni % accumulate == 0:
                scaler.step(optimizer)  # optimizer.step
                scaler.update()
                optimizer.zero_grad()
                if ema:
                    ema.update(model)

            # Print
            if rank in [-1, 0]:
                mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
                mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
                s = ('%10s' * 2 + '%10.4g' * 6) % (
                    '%g/%g' % (epoch, epochs - 1), mem, *mloss, targets.shape[0], imgs.shape[-1])
                pbar.set_description(s)

                # Plot
                if plots and ni < 10:
                    f = save_dir / f'train_batch{ni}.jpg'  # filename
                    Thread(target=plot_images, args=(imgs, targets, paths, f), daemon=True).start()
                    # if tb_writer:
                    #     tb_writer.add_image(f, result, dataformats='HWC', global_step=epoch)
                    #     tb_writer.add_graph(torch.jit.trace(model, imgs, strict=False), [])  # add model graph

            # end batch ------------------------------------------------------------------------------------------------
        # end epoch ----------------------------------------------------------------------------------------------------
        
        # Scheduler
        lr = [x['lr'] for x in optimizer.param_groups]  # for tensorboard
        scheduler.step()

        # DDP process 0 or single-GPU
        if rank in [-1, 0]:
            # mAP
            ema.update_attr(model, include=['yaml', 'nc', 'hyp', 'gr', 'names', 'stride', 'class_weights'])
            final_epoch = epoch + 1 == epochs
            if not opt.notest or final_epoch:  # Calculate mAP
                results, maps, times = test.test(data_dict,
                                                 batch_size=batch_size * 2,
                                                 imgsz=imgsz_test,
                                                 model=ema.ema,
                                                 single_cls=opt.single_cls,
                                                 dataloader=testloader,
                                                 save_dir=save_dir,
                                                 verbose=nc < 50 and final_epoch,
                                                 plots=plots and final_epoch,
                                                 compute_loss=compute_loss,
                                                 is_coco=is_coco,
                                                 v5_metric=opt.v5_metric)

            # Write
            with open(results_file, 'a') as f:
                f.write(s + '%10.4g' * 7 % results + '\n')  # append metrics, val_loss
            if len(opt.name) and opt.bucket:
                os.system('gsutil cp %s gs://%s/results/results%s.txt' % (results_file, opt.bucket, opt.name))

            # Log
            tags = ['train/box_loss', 'train/obj_loss', 'train/cls_loss',  # train loss
                    'metrics/precision', 'metrics/recall', 'metrics/mAP_0.5', 'metrics/mAP_0.5:0.95',
                    'val/box_loss', 'val/obj_loss', 'val/cls_loss',  # val loss
                    'x/lr0', 'x/lr1', 'x/lr2']  # params

            # Update best mAP
            fi = fitness(np.array(results).reshape(1, -1))  # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
            if fi > best_fitness:
                best_fitness = fi

            # Save model
            if (not opt.nosave) or (final_epoch and not opt.evolve):  # if save
                ckpt = {'epoch': epoch,
                        'best_fitness': best_fitness,
                        'training_results': results_file.read_text(),
                        'model': deepcopy(model.module if is_parallel(model) else model).half(),
                        'ema': deepcopy(ema.ema).half(),
                        'updates': ema.updates,
                        'optimizer': optimizer.state_dict(),
                        'wandb_id': None}

                # Save last, best and delete
                torch.save(ckpt, last)
                if best_fitness == fi:
                    torch.save(ckpt, best)
                # if (best_fitness == fi) and (epoch >= 200):
                #     torch.save(ckpt, wdir / 'best_{:03d}.pt'.format(epoch))
                # if epoch == 0:
                #     torch.save(ckpt, wdir / 'epoch_{:03d}.pt'.format(epoch))
                # elif ((epoch+1) % 25) == 0:
                #     torch.save(ckpt, wdir / 'epoch_{:03d}.pt'.format(epoch))
                # elif epoch >= (epochs-5):
                #     torch.save(ckpt, wdir / 'epoch_{:03d}.pt'.format(epoch))
                del ckpt

        # end epoch ----------------------------------------------------------------------------------------------------
    # end training
    if rank in [-1, 0]:
        # Plots
        if plots:
            plot_results(save_dir=save_dir)  # save as results.png
        # Test best.pt
        logger.info('%g epochs completed in %.3f hours.\n' % (epoch - start_epoch + 1, (time.time() - t0) / 3600))
        if opt.data.endswith('coco.yaml') and nc == 80:  # if COCO
            for m in (last, best) if best.exists() else (last):  # speed, mAP tests
                results, _, _ = test.test(opt.data,
                                          batch_size=batch_size * 2,
                                          imgsz=imgsz_test,
                                          conf_thres=0.001,
                                          iou_thres=0.7,
                                          model=attempt_load(m, device).half(),
                                          single_cls=opt.single_cls,
                                          dataloader=testloader,
                                          save_dir=save_dir,
                                          save_json=True,
                                          plots=False,
                                          is_coco=is_coco,
                                          v5_metric=opt.v5_metric)

        # Strip optimizers
        final = best if best.exists() else last  # final model
        for f in last, best:
            if f.exists():
                strip_optimizer(f)  # strip optimizers
        if opt.bucket:
            os.system(f'gsutil cp {final} gs://{opt.bucket}/weights')  # upload
    else:
        dist.destroy_process_group()
    torch.cuda.empty_cache()
    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 代码来自:BiliBli 魔鬼面具up主
    parser.add_argument('--weights', type=str, default='', help='trained weights path')
    parser.add_argument('--data', type=str, default='/home/hjj/Desktop/dataset/dataset_head/data.yaml', help='data.yaml path')
    parser.add_argument('--hyp', type=str, default='data/hyp.scratch.tiny.yaml', help='hyperparameters path')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch-size', type=int, default=16, help='total batch size for all GPUs')
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='[train, test] image sizes')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--notest', action='store_true', help='only test final epoch')
    parser.add_argument('--noautoanchor', action='store_true', help='disable autoanchor check')
    parser.add_argument('--evolve', action='store_true', help='evolve hyperparameters')
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
    parser.add_argument('--cache-images', action='store_true', help='cache images for faster training')
    parser.add_argument('--image-weights', action='store_true', help='use weighted image selection for training')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%%')
    parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class')
    parser.add_argument('--adam', action='store_true', help='use torch.optim.Adam() optimizer')
    parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
    parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')
    parser.add_argument('--workers', type=int, default=4, help='maximum number of dataloader workers')
    parser.add_argument('--project', default='runs/train', help='save to project/name')
    parser.add_argument('--entity', default=None, help='W&B entity')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--quad', action='store_true', help='quad dataloader')
    parser.add_argument('--linear-lr', action='store_true', help='linear LR')
    parser.add_argument('--label-smoothing', type=float, default=0.0, help='Label smoothing epsilon')
    parser.add_argument('--upload_dataset', action='store_true', help='Upload dataset as W&B artifact table')
    parser.add_argument('--bbox_interval', type=int, default=-1, help='Set bounding-box image logging interval for W&B')
    parser.add_argument('--save_period', type=int, default=-1, help='Log model after every "save_period" epoch')
    parser.add_argument('--artifact_alias', type=str, default="latest", help='version of dataset artifact to be used')
    parser.add_argument('--freeze', nargs='+', type=int, default=[0], help='Freeze layers: backbone of yolov7=50, first3=0 1 2')
    parser.add_argument('--v5-metric', action='store_true', help='assume maximum recall as 1.0 in AP calculation')
    
    # ------------------------------BiliBili 魔鬼面具up主------------------------------
    # prune
    parser.add_argument('--prune_method', type=str, default=None, help='prune method')
    parser.add_argument('--speed_up', type=float, default=2.0, help='speed up')
    parser.add_argument("--global_pruning", action="store_true", default=False)
    parser.add_argument("--max_sparsity", type=float, default=1.0)
    parser.add_argument("--iterative_steps", type=int, default=200)
    
    # ------------------------------BiliBili 魔鬼面具up主------------------------------
    # sparsity learning
    parser.add_argument("--reg", type=float, default=5e-4)
    parser.add_argument("--delta_reg", type=float, default=1e-4, help='for growing regularization')
    parser.add_argument("--sl_hyp", type=str, default='data/hyp.scratch.sl.yaml', help='hyperparameters path for sparsity learning')
    parser.add_argument("--sl_epochs", type=int, default=100)
    parser.add_argument("--sl_model", type=str, default="", help='sparsity learning trained model weights')
    
    # ------------------------------BiliBili 魔鬼面具up主------------------------------
    parser.add_argument("--reg_decay_type", type=str, default='linear', choices=['constant', 'linear', 'step'], help='reg decay type choice in sparsity learning')
    parser.add_argument("--reg_decay", type=float, default=0.01)
    parser.add_argument("--reg_decay_step", type=int, default=10, help='reg decay step size in sparsity learning and reg_decay_type==step')
    
    opt = parser.parse_args()

    # Set DDP variables
    opt.world_size = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    opt.global_rank = int(os.environ['RANK']) if 'RANK' in os.environ else -1
    set_logging(opt.global_rank)

    # Resume
    wandb_run = check_wandb_resume(opt)
    # opt.hyp = opt.hyp or ('hyp.finetune.yaml' if opt.weights else 'hyp.scratch.yaml')
    opt.data, opt.hyp = check_file(opt.data), check_file(opt.hyp)  # check files
    assert len(opt.weights), 'either --weights must be specified'
    opt.img_size.extend([opt.img_size[-1]] * (2 - len(opt.img_size)))  # extend to 2 sizes (train, test)
    opt.name = 'evolve' if opt.evolve else opt.name
    opt.save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok | opt.evolve)  # increment run

    # DDP mode
    opt.total_batch_size = opt.batch_size
    device = select_device(opt.device, batch_size=opt.batch_size)
    if opt.local_rank != -1:
        assert torch.cuda.device_count() > opt.local_rank
        torch.cuda.set_device(opt.local_rank)
        device = torch.device('cuda', opt.local_rank)
        dist.init_process_group(backend='nccl', init_method='env://')  # distributed backend
        assert opt.batch_size % opt.world_size == 0, '--batch-size must be multiple of CUDA device count'
        opt.batch_size = opt.total_batch_size // opt.world_size

    # Data
    with open(opt.data) as f:
        data_dict = yaml.load(f, Loader=yaml.SafeLoader)  # data dict
    is_coco = opt.data.endswith('coco.yaml')
    
    nc = 1 if opt.single_cls else int(data_dict['nc'])  # number of classes
    names = ['item'] if opt.single_cls and len(data_dict['names']) != 1 else data_dict['names']  # class names
    assert len(names) == nc, '%g names found for nc=%g dataset in %s' % (len(names), nc, opt.data)  # check
    with torch_distributed_zero_first(opt.global_rank):
        check_dataset(data_dict)  # check
    train_path = data_dict['train']
    test_path = data_dict['val']
    
    # Model
    weights = opt.weights
    pretrained = weights.endswith('.pt')
    if pretrained:
        model = torch.load(weights, map_location=device)
        model = model['ema' if model.get('ema') else 'model'].float()
        for p in model.parameters():
            p.requires_grad_(True)
        model.info(img_size=opt.img_size[0])
        logger.info(f'Loaded {weights}')  # report
    else:
        assert weights.endswith('.pt'), "compress need weights."
    
    # Image sizes
    gs = max(int(model.stride.max()), 32)  # grid size (max stride)
    nl = model.model[-1].nl  # number of detection layers (used for scaling hyp['obj'])
    imgsz, imgsz_test = [check_img_size(x, gs) for x in opt.img_size]  # verify imgsz are gs-multiples
    
    # Hyperparameters
    with open(opt.hyp) as f:
        hyp = yaml.load(f, Loader=yaml.SafeLoader)  # load hyps
    
    # Trainloader
    dataloader, dataset = create_dataloader(train_path, imgsz, opt.batch_size, gs, opt,
                                            hyp=hyp, augment=True, cache=opt.cache_images, rect=opt.rect, rank=opt.global_rank,
                                            world_size=opt.world_size, workers=opt.workers,
                                            image_weights=opt.image_weights, quad=opt.quad, prefix=colorstr('train dataset: '))
    mlc = np.concatenate(dataset.labels, 0)[:, 0].max()  # max label class
    nb = len(dataloader)  # number of batches
    assert mlc < nc, 'Label class %g exceeds nc=%g in %s. Possible class labels are 0-%g' % (mlc, nc, opt.data, nc - 1)

    # Process 0
    if opt.global_rank in [-1, 0]:
        testloader = create_dataloader(test_path, imgsz_test, opt.batch_size * 2, gs, opt,  # testloader
                                       hyp=hyp, cache=opt.cache_images and not opt.notest, rect=False, rank=-1,
                                       world_size=opt.world_size, workers=opt.workers,
                                       pad=0.5, prefix=colorstr('val dataset: '))[0]

        if not opt.resume:
            labels = np.concatenate(dataset.labels, 0)
            c = torch.tensor(labels[:, 0])  # classes

            # Anchors
            if not opt.noautoanchor:
                check_anchors(dataset, model=model, thr=hyp['anchor_t'], imgsz=imgsz)
            model.half().float()  # pre-reduce anchor precision
    
    prune_dataloader, _ = create_dataloader(train_path, imgsz, opt.batch_size // 2, gs, opt,
                                            hyp=hyp, augment=True, cache=False, rect=opt.rect, rank=opt.global_rank,
                                            world_size=opt.world_size, workers=opt.workers,
                                            image_weights=opt.image_weights, quad=opt.quad, prefix=colorstr('prune dataset: '))
    
    # opt info
    logger.info(opt)
    
    # get prune
    example_inputs = torch.randn((1, 3, imgsz, imgsz)).to(device)
    sparsity_learning, imp, prune = get_pruner(opt, model, example_inputs)
    
    # sparsity_learning
    if sparsity_learning and not opt.sl_model:
        opt.sl_model = sparsity_learning_train(opt, model, prune, dataloader, testloader, device)
    
    if sparsity_learning:
        model = torch.load(opt.sl_model, map_location=device)
        model = model['ema' if model.get('ema') else 'model'].float()
        for p in model.parameters():
            p.requires_grad_(True)
        _, imp, prune = get_pruner(opt, model, example_inputs)
        logger.info(f'Loaded sparsity_learning weights from {opt.sl_model}')  # report
    
    # pruning
    model = model_prune(opt, model, imp, prune, example_inputs, testloader, imgsz_test, prune_dataloader)
    
    # test fuse
    fuse_model = deepcopy(model)
    for p in fuse_model.parameters():
            p.requires_grad_(False)
    fuse_model.fuse()
    del fuse_model

    # finetune
    finetune(opt, model, dataloader, testloader, device)