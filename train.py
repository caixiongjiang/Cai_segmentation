import argparse
import os
import pprint
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

import numpy as np

import nets
from utils.utils import weights_init, create_logger
from configs import update_config
from configs import default_config

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
    if torch.cuda.device_count() != len(gpus):
        print("The gpu numbers do not match!")
        return 0
    
    if default_config.MODEL.PRETRAINED:
        model = nets.fastsegformer.FastSegFormer(num_classes=default_config.DATASET.NUM_CLASSES, 
                                                 pretrained=default_config.MODEL.PRETRAINED,
                                                 Pyramid="multiscale",
                                                 cnn_branch=True,
                                                 fork_feat=False
                                            )
    else:
        model = nets.fastsegformer.FastSegFormer(num_classes=default_config.DATASET.NUM_CLASSES, 
                                                 pretrained=default_config.MODEL.PRETRAINED,
                                                 Pyramid="multiscale",
                                                 cnn_branch=True,
                                                 fork_feat=False
                                            )
        weights_init(model, init_type='kaiming')
        
    batch_size = default_config.TRAIN.BATCH_SIZE_PER_GPU * len(gpus)
       
    
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
    
    
    
    
    
    
    
    
    




if __name__ == '__main__':
    parse_args()
