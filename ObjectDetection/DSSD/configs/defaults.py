import torch.cuda
from yacs.config import CfgNode as CN

_C = CN()

_C.MODEL = CN()
_C.MODEL.META_ARCHITECTURE = 'DSSDDetector'
_C.MODEL.DEVICE = 'cuda:1' if torch.cuda.is_available() else 'cpu'
# match default boxes to any ground truth with jaccard overlap higher than a threshold (0.5)
_C.MODEL.THRESHOLD = 0.5
_C.MODEL.NUM_CLASSES = 21
# Hard negative mining
_C.MODEL.NEG_POS_RATIO = 3
_C.MODEL.CENTER_VARIANCE = 0.1
_C.MODEL.SIZE_VARIANCE = 0.2

# ---------------------------------------------------------------------------- #
# Backbone
# ---------------------------------------------------------------------------- #
_C.MODEL.BACKBONE = CN()
_C.MODEL.BACKBONE.NAME = 'resnet101'
_C.MODEL.BACKBONE.OUT_CHANNELS = (512, 1024, 1024, 1024, 1024, 1024) # 40, 20, 10, 5, 3, 1
_C.MODEL.BACKBONE.PRETRAINED = True

# ---------------------------------------------------------------------------- #
# Decoder
# ---------------------------------------------------------------------------- #
_C.MODEL.DECODER = CN()
_C.MODEL.DECODER.WITH_DECODER = True
_C.MODEL.DECODER.NAME = 'DSSDDecoder'
_C.MODEL.DECODER.OUT_CHANNELS = [1024, 512, 512, 512, 512, 512] # 1, 3, 5, 10, 20, 40
_C.MODEL.DECODER.DECONV_KERNEL_SIZE = [3, 1, 2, 2, 2]
_C.MODEL.DECODER.ELMW_TYPE = "prod"  # ["sum", "prod"]

# -----------------------------------------------------------------------------
# PRIORS
# -----------------------------------------------------------------------------
_C.MODEL.PRIORS = CN()
_C.MODEL.PRIORS.FEATURE_MAPS = [40, 20, 10, 5, 3, 1]
_C.MODEL.PRIORS.STRIDES = [8, 16, 32, 64, 107, 320]  # strides = image_size // feature_maps
_C.MODEL.PRIORS.MIN_SIZES = [30, 60, 111, 162, 213, 264]
_C.MODEL.PRIORS.MAX_SIZES = [60, 111, 162, 213, 264, 315]  # [60, 111, 162, 213, 264, 315]
_C.MODEL.PRIORS.ASPECT_RATIOS = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
# When has 1 aspect ratio, every location has 4 boxes, 2 ratio 6 boxes.
# #boxes = 2 + #ratio * 2
_C.MODEL.PRIORS.BOXES_PER_LOCATION = [4, 6, 6, 6, 4, 4]  # number of boxes per feature map location
_C.MODEL.PRIORS.CLIP = True

# -----------------------------------------------------------------------------
# Box Head
# -----------------------------------------------------------------------------
_C.MODEL.BOX_HEAD = CN()
_C.MODEL.BOX_HEAD.NAME = 'DSSDBoxHead'
_C.MODEL.BOX_HEAD.PREDICTOR = 'DSSDBoxPredictor_C'

# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
_C.INPUT = CN()
# Image size
_C.INPUT.IMAGE_SIZE = 320
# Values to be used for image normalization, RGB layout
_C.INPUT.PIXEL_MEAN = [123, 117, 104]

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASETS = CN()
# List of the dataset names for training, as present in paths_catalog.py
_C.DATASETS.TRAIN = ['voc_2012_trainval']
# List of the dataset names for validation, as present in paths_catalog.py
_C.DATASETS.VAL = ['voc_2012_val']
# List of the dataset names for testing, as present in paths_catalog.py
_C.DATASETS.TEST = ['voc_2007_test']

# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATA_LOADER = CN()
# Number of data loading threads
_C.DATA_LOADER.NUM_WORKERS = 0
_C.DATA_LOADER.PIN_MEMORY = True

# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()
# train configs
_C.SOLVER.MAX_ITER = 120000
_C.SOLVER.LR_STEPS = [80000, 100000]
_C.SOLVER.GAMMA = 0.1
_C.SOLVER.BATCH_SIZE = 4
_C.SOLVER.LR = 1e-3
_C.SOLVER.MOMENTUM = 0.9
_C.SOLVER.WEIGHT_DECAY = 5e-4
_C.SOLVER.WARMUP_FACTOR = 1.0 / 3
_C.SOLVER.WARMUP_ITERS = 500

# ---------------------------------------------------------------------------- #
# Specific test options
# ---------------------------------------------------------------------------- #
_C.TEST = CN()
_C.TEST.NMS_THRESHOLD = 0.45
_C.TEST.CONFIDENCE_THRESHOLD = 0.01
_C.TEST.MAX_PER_CLASS = -1
_C.TEST.MAX_PER_IMAGE = 100
_C.TEST.BATCH_SIZE = 1

_C.OUTPUT_DIR = './outputs/'
