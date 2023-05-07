import argparse
import os
import pprint
import timeit
from tensorboardX import SummaryWriter
import torch.distributed as dist # 分布式训练
import torch.optim as optim

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

import numpy as np

from nets.fastsegformer import FastSegFormer
from utils.utils import weights_init, create_logger, get_lr_scheduler
from configs import update_config
from configs import default_config
from utils.loss import CrossEntropy, Focal_loss



# TODO: train.py

def parse_args():
    parser = argparse.ArgumentParser(description='Train segmentation network')
    
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        default="configs/cityscapes/fastsegformer_city.yaml",
                        type=str)
    parser.add_argument('--seed', type=int, default=300)    
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)
    

    args = parser.parse_args()
    update_config(default_config, args)

    return args

def main():
    args = parse_args()
    
    if args.seed > 0:
        import random
        print('Seeding with', args.seed)
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        
    # logs
    logger, final_output_dir, tb_log_dir = create_logger(
        default_config, args.cfg, 'train')

    logger.info(pprint.pformat(args))
    logger.info(default_config)

    writer_dict = {
        'writer': SummaryWriter(tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }
        
    
    
    # cudnn relative setting
    cudnn.benchmark = default_config.CUDNN.BENCHMARK
    cudnn.deterministic = default_config.CUDNN.DETERMINISTIC
    cudnn.enabled = default_config.CUDNN.ENABLED
    gpus = list(default_config.GPUS)
    
    ngpus_per_node  = torch.cuda.device_count()
    if default_config.DISTRIBUTED:
        dist.init_process_group(backend="nccl")
        local_rank  = int(os.environ["LOCAL_RANK"])
        rank        = int(os.environ["RANK"])
        device      = torch.device("cuda", local_rank)
        if local_rank == 0:
            print(f"[{os.getpid()}] (rank = {rank}, local_rank = {local_rank}) training...")
            print("Gpu Device Count : ", ngpus_per_node)
    
    if ngpus_per_node != len(gpus):
        print("The gpu numbers do not match!")
        return 0
    
    if default_config.MODEL.PRETRAINED:
        model = FastSegFormer(num_classes=default_config.DATASET.NUM_CLASSES, 
                                                 pretrained=default_config.MODEL.PRETRAINED,
                                                 Pyramid="multiscale",
                                                 cnn_branch=True,
                                                 fork_feat=False
                                            )
    else:
        model = FastSegFormer(num_classes=default_config.DATASET.NUM_CLASSES, 
                                                 pretrained=default_config.MODEL.PRETRAINED,
                                                 Pyramid="multiscale",
                                                 cnn_branch=True,
                                                 fork_feat=False
                                            )
        weights_init(model, init_type='kaiming')
        
    batch_size = default_config.TRAIN.BATCH_SIZE_PER_GPU * len(gpus)
    
    # Freeze backbone
    if default_config.TRAIN.FREEZE_BACKBONE:
        model.freeze_backbone()
       
    
    # Dataloader
    crop_size = (default_config.TRAIN.IMAGE_SIZE[1], default_config.TRAIN.IMAGE_SIZE[0])
    train_dataset = eval('datasets.'+ default_config.DATASET.DATASET)(
                        root=default_config.DATASET.ROOT,
                        list_path=default_config.DATASET.TRAIN_SET,
                        num_classes=default_config.DATASET.NUM_CLASSES,
                        multi_scale=default_config.TRAIN.MULTI_SCALE,
                        flip=default_config.TRAIN.FLIP,
                        ignore_label=default_config.TRAIN.IGNORE_LABEL,
                        base_size=default_config.TRAIN.BASE_SIZE,
                        crop_size=crop_size,
                        scale_factor=default_config.TRAIN.SCALE_FACTOR)
    trainloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=default_config.TRAIN.SHUFFLE,
        num_workers=default_config.WORKERS,
        pin_memory=default_config.PIN_MEMORY,
        drop_last=True)
    
    
    val_size = (default_config.VAL.IMAGE_SIZE[1], default_config.VAL.IMAGE_SIZE[0])
    val_dataset = eval('datasets.'+default_config.DATASET.DATASET)(
                        root=default_config.DATASET.ROOT,
                        list_path=default_config.DATASET.VAL_SET,
                        num_classes=default_config.DATASET.NUM_CLASSES,
                        multi_scale=default_config.VAL.MULTI_SCALE,
                        flip=default_config.VAL.FLIP,
                        ignore_label=default_config.TRAIN.IGNORE_LABEL,
                        base_size=default_config.VAL.BASE_SIZE,
                        crop_size=val_size)
    
    valloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=default_config.VAL.BATCH_SIZE_PER_GPU * len(gpus),
        shuffle=False,
        num_workers=default_config.WORKERS,
        pin_memory=default_config.PIN_MEMORY,
        drop_last=False)

    # criterion
    
    if default_config.LOSS.LOSS_NAME == 'CE':
        seg_criterion = CrossEntropy(ignore_label=default_config.TRAIN.IGNORE_LABEL,
                                    weight=train_dataset.class_weights)
    elif default_config.LOSS.LOSS_NAME == 'Focal':
        seg_criterion = Focal_loss(ignore_label=default_config.TRAIN.IGNORE_LABEL,
                                    weight=train_dataset.class_weights)
    else:
        raise ValueError('Currently, ' + '{}'.format(default_config.LOSS.LOSS_NAME) + 'loss is not supported!')
    
    model_train = model.train()
    
    # 是否采用低精度
    if default_config.FP16 == True:
        from torch.cuda.amp import GradScaler
        scaler = GradScaler()
    else:
        scaler = None
    # 训练环境是单机多卡 or 分布式环境（多机多卡）
    if default_config.SYNC_BN and  ngpus_per_node > 1 and default_config.DISTRIBUTED:
        model_train = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_train)
    else:
       raise ValueError("Sync_bn is not support in one gpu or not distributed.")
    
    if default_config.DISTRIBUTED:
        model_train = model_train.cuda(local_rank)
        model_train = torch.nn.parallel.DistributedDataParallel(model_train, device_ids=[local_rank], find_unused_parameters=True)
    else:
        model_train = nn.DataParallel(model_train, device_ids=gpus).cuda()
        
    # fit start_lr 
    nbs             = 16
    lr_limit_max    = 1e-4 if default_config.TRAIN.OPTIMIZER == 'adam' else 1e-1
    lr_limit_min    = 1e-4 if default_config.TRAIN.OPTIMIZER == 'adam' else 5e-4
    Init_lr_fit     = min(max(batch_size / nbs * default_config.TRAIN.LR, lr_limit_min), lr_limit_max)
    Min_lr_fit      = min(max(batch_size / nbs * (0.01 * default_config.TRAIN.LR), lr_limit_min * 1e-2), lr_limit_max * 1e-2)
        
    # optimizer
    optimizer = {
            'adam'  : optim.Adam(model.parameters(), Init_lr_fit, betas = (default_config.TRAIN.MOMENTUM, 0.999), weight_decay = default_config.TRAIN.WD),
            'sgd'   : optim.SGD(model.parameters(), Init_lr_fit, momentum=default_config.TRAIN.MOMENTUM, nesterov=True, weight_decay = default_config.TRAIN.WD)
        }[default_config.TRAIN.OPTIMIZER]
    
    # lr change
    lr_scheduler_func = get_lr_scheduler(default_config.LR_DECAY, Init_lr_fit, Min_lr_fit, default_config.TRAIN.FREEZE_EPOCH)
    
    train_epoch_iters = int(train_dataset.__len__() / default_config.TRAIN.BATCH_SIZE_PER_GPU / len(gpus)) # train 每个epoch需要迭代的次数
    val_epoch_iters = int(val_dataset.__len__() / default_config.VAL.BATCH_SIZE_PER_GPU / len(gpus)) # train 每个epoch需要迭代的次数
    
    # Train
    best_mIoU = 0
    last_epoch = 0
    flag_rm = default_config.AUTO_RESUME     # 中断训练的恢复
    
    if flag_rm:
        model_state_file = os.path.join(final_output_dir, 'checkpoint.pth')
        if os.path.isfile(model_state_file):
            checkpoint = torch.load(model_state_file, map_location={'cuda:0': 'cpu'})
            best_mIoU = checkpoint['best_mIoU']
            last_epoch = checkpoint['epoch']
            dct = checkpoint['state_dict']
            
            model.load_state_dict({k.replace('model.', ''): v for k, v in dct.items() if k.startswith('model.')})
            optimizer.load_state_dict(checkpoint['optimizer'])
            logger.info("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))
            
    start = timeit.default_timer()   # 计算程序运行的时间
    end_epoch = default_config.TRAIN.END_EPOCH
    num_iters = default_config.TRAIN.END_EPOCH * train_epoch_iters
    
    for epoch in range(last_epoch, end_epoch):
        
        if epoch <= default_config.TRAIN.FREEZE_EPOCH:
            x
        
    
    
    
    
    
    
    
    
    
    




if __name__ == '__main__':
    parse_args()
