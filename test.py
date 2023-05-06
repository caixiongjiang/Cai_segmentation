from utils.loss import CrossEntropy, Focal_loss 
from utils.utils import create_logger
from configs import default_config
from train import parse_args
import pprint

import torch

def log_test():
    """
    Test some function with other modules in ROOT dir
    """
    args = parse_args()
    logger, final_output_dir, tensorboard_log_dir = create_logger(default_config, args.cfg, 'train')
    logger.info(pprint.pformat(args))
    logger.info(default_config)
    print(final_output_dir)
    print(tensorboard_log_dir)








if __name__ == '__main__':
    log_test()