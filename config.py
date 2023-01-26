from yacs.config import CfgNode as CN


_C = CN()
# random seed number
_C.SEED = 0
# number of gpus per node
_C.NUM_GPUS = 4
_C.VISIBLE_DEVICES = 0
# directory to save result txt file
_C.RESULT_DIR = 'results'

_C.PROTRANS = CN()
_C.PROTRANS.ENABLE = False
_C.PROTRANS.LAMBDA_CE = 1.0
_C.PROTRANS.LAMBDA_AGGR = 15.0
_C.PROTRANS.LAMBDA_SEP = 1.0
_C.PROTRANS.T = 1.0
# FE stands for Feature Extractor
_C.PROTRANS.FE = CN()
_C.PROTRANS.FE.NO_TRANSFORM = True
_C.PROTRANS.SAMPLE_PROTO = True
_C.PROTRANS.ALPHA = 0.0001

_C.DATA_LOADER = CN()
# the number of data loading workers per gpu
_C.DATA_LOADER.NUM_WORKERS = 4
_C.DATA_LOADER.PIN_MEMORY = True
_C.DATA_LOADER.DROP_LAST = True

_C.DATA = CN()
_C.DATA.BASE_DIR = '/mnt/workspace/gusrl/data/TransferLearning/'
_C.DATA.MEAN = [0.485, 0.456, 0.406]
_C.DATA.STD = [0.229, 0.224, 0.225]
_C.DATA.SAMPLING_RATES = 1.0

_C.TRAIN = CN()
_C.TRAIN.ENABLE = True
_C.TRAIN.BATCH_SIZE = 64
_C.TRAIN.DATASET = 'Aircraft'
_C.TRAIN.SPLIT = 'train'
# directory to save checkpoints
_C.TRAIN.CHECKPOINT_DIR = 'checkpoints'
# path to checkpoint to resume training
_C.TRAIN.RESUME = ''
# epoch period to save checkpoint
_C.TRAIN.CHECKPOINT_PERIOD = 200
# epoch period to evaluate on a validation set
_C.TRAIN.EVAL_PERIOD = 5
# iteration frequency to print progress meter
_C.TRAIN.PRINT_FREQ = 1

_C.VAL = CN()
_C.VAL.SPLIT = 'val'

_C.TEST = CN()
_C.TEST.ENABLE = True
_C.TEST.SPLIT = 'test'

_C.MODEL = CN()
_C.MODEL.NUM_CLASSES = 100
_C.MODEL.ARCH = 'resnet50'
# whether to use pretrained weights from torchvision.
_C.MODEL.ARCH_PRETRAINED = False
_C.MODEL.FC = CN()
_C.MODEL.FC.BIAS = True

_C.SOLVER = CN()
_C.SOLVER.START_EPOCH = 0
_C.SOLVER.MAX_EPOCH = 100
_C.SOLVER.OPTIMIZING_METHOD = 'sgd'
_C.SOLVER.BASE_LR = 0.001
_C.SOLVER.WEIGHT_DECAY = 0.0001
_C.SOLVER.MOMENTUM = 0.9
_C.SOLVER.NESTEROV = True
_C.SOLVER.DAMPENING = 0.0
_C.SOLVER.LR_POLICY = 'cosine'
_C.SOLVER.COSINE_END_LR = 0.0
_C.SOLVER.COSINE_AFTER_WARMUP = False
_C.SOLVER.WARMUP_EPOCHS = 0
_C.SOLVER.WARMUP_START_LR = 0.001
# learning rate of last fc layer is scaled by fc_lr_ratio
_C.SOLVER.FC_LR_RATIO = 10.0


def get_cfg_defaults():

    return _C.clone()
