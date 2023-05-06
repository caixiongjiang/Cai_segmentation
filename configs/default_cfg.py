from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from yacs.config import CfgNode as CN


_C = CN()

_C.LOG_DIR = 'logs'                     # 训练日志地址
_C.LOG_WEIGHT_DIR = 'logs/log_weights'  # 过程权重保存的地址
_C.GPUS = (0,)                          # 训练使用的设备
_C.WORKERS = 2                          # 线程数
_C.SAVE_PERIOD = 5                      # 权重保存间距
_C.AUTO_RESUME = False                  # 自动恢复训练
_C.PIN_MEMORY = True                    # 锁页内存

# Cudnn related params
_C.CUDNN = CN()
_C.CUDNN.BENCHMARK = True               # 使用benchmark，加速运行
_C.CUDNN.DETERMINISTIC = False          # 如果每次输入的iterator都不同，则为True避免这种情况
_C.CUDNN.ENABLED = True                 # ---->引起问题的代码行

# common params for NETWORK
_C.MODEL = CN()
_C.MODEL.NAME = 'fastsegformer'                                            # 模型名字
_C.MODEL.PRETRAINED = True                                                 # True or False
_C.MODEL.NUM_OUTPUTS = 1                                                   # 模型最后输出的个数（用于多头损失）


_C.LOSS = CN()                                                              
_C.LOSS.LOSS_NAME = 'CE'                                                   # 损失函数类型
_C.LOSS.CLASS_BALANCE = False                                              # 类别平衡
_C.LOSS.BALANCE_WEIGHTS = [0.5, 0.5]                                       # 多头损失之间的权重比例，加起来一般为1
_C.LOSS.SB_WEIGHTS = 1.0                                                   # 单头损失的权重

# DATASET related params
_C.DATASET = CN()
_C.DATASET.ROOT = 'data/'
_C.DATASET.DATASET = 'cityscapes'
_C.DATASET.NUM_CLASSES = 19
_C.DATASET.TRAIN_SET = 'list/cityscapes/train.lst'
_C.DATASET.VAL_SET = 'list/cityscapes/val.lst'

# training
_C.TRAIN = CN()
_C.TRAIN.MODE = 'normal'                                                    # 默认为normal, 还有就是蒸馏训练（KD）
_C.TRAIN.IMAGE_SIZE = [1024, 1024]                                          # width * height
_C.TRAIN.BASE_SIZE = 2048                                                   # 缩放因子需要的原始的尺寸
_C.TRAIN.FLIP = True                                                        # 是否进行翻转
_C.TRAIN.MULTI_SCALE = True                                                 # 是否进行多尺度训练
_C.TRAIN.SCALE_FACTOR = 16                                                  # 缩放因子,代表输出的图像为[2048, 2048] / 16 = [128, 128](PIDNet)

_C.TRAIN.LR = 0.01
_C.TRAIN.EXTRA_LR = 0.001

_C.TRAIN.OPTIMIZER = 'sgd'                                                   # SGD or Adam
_C.TRAIN.MOMENTUM = 0.9
_C.TRAIN.WD = 0.0001                                                         # if SGD weight_decay = 0.0001 , if Adam weight_decay = None
_C.TRAIN.NESTEROV = False
_C.TRAIN.IGNORE_LABEL = -1

_C.TRAIN.BEGIN_EPOCH = 0
_C.TRAIN.END_EPOCH = 1000

_C.TRAIN.RESUME = False

_C.TRAIN.BATCH_SIZE_PER_GPU = 32
_C.TRAIN.SHUFFLE = True

# Validation
_C.VAL = CN()
_C.VAL.IMAGE_SIZE = [2048, 1024]                                             # width * height 原图为[2048, 1024]
_C.VAL.BASE_SIZE = 2048
_C.VAL.BATCH_SIZE_PER_GPU = 32
_C.VAL.MODEL_WEIGHT = ''
_C.VAL.FLIP_TEST = False
_C.VAL.MULTI_SCALE = False

_C.VAL.OUTPUT_INDEX = -1                                                     # 多输出时选择进行验证的头


def update_config(cfg, args):
    cfg.defrost()
    
    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)

    cfg.freeze()


if __name__ == '__main__':
    import sys
    with open(sys.argv[1], 'w') as f:
        print(_C, file=f)
