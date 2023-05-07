from utils.loss import CrossEntropy, Focal_loss 
from utils.utils import create_logger
from configs import default_config
from train import parse_args

from nets.fastsegformer.fastsegformer import FastSegFormer

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
    
    
def model_test():
    model = FastSegFormer(num_classes=4, pretrained=False, Pyramid="multiscale", fork_feat=True, cnn_branch=True)
    a,b,c,d,e = model.getModelSize(model)
    img = torch.randn((1, 3, 224, 224))
    model.get_Flops_params(model, img)
    # print(model)
    outputs = model(img)
    # print(outputs.shape)
    for output in outputs:
        print(output.shape)








if __name__ == '__main__':
    log_test()
    # model_test()