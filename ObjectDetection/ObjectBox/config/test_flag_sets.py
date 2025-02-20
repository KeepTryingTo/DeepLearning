"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2024/12/10-9:02
@CSDN   : https://blog.csdn.net/Keep_Trying_Go?spm=1010.2135.3001.5421
"""

from absl import flags,app
from pathlib import Path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # ObjectBox root directory

FLAGS = flags.FLAGS

# Note: set (exp = 'pascal') or (exp = 'coco')
exp = 'pascal'

if exp == 'coco':
    # GENERAL
    flags.DEFINE_string('cfg',  str(ROOT / 'config/objectBox_COCO.yaml'), 'model config file')
    flags.DEFINE_string('data',  str(ROOT / 'data/coco.yaml'), 'coco.yaml path')
    flags.DEFINE_string('exp', 'coco', 'coco or pascal')

elif exp == 'pascal':
    # GENERAL
    flags.DEFINE_string('cfg', str(ROOT / 'config/objectBox_VOC.yaml'), 'model config file')
    flags.DEFINE_string('data', str(ROOT / 'data/VOC.yaml'), 'VOC.yaml path')
    flags.DEFINE_string('exp', 'pascal', 'coco or pascal')

else:
    raise NotImplementedError

# GENERAL
flags.DEFINE_string('project', r'/home/ff/myProject/KGT/myProjects/myProjects/ObjectBox-master/runs',
                    'save to project/name')
flags.DEFINE_bool('exist_ok', False, 'existing project/name ok, do not increment')
flags.DEFINE_bool('WANDB', False, 'wandb?')

flags.DEFINE_string('weights',
                    default=r'/home/ff/myProject/KGT/myProjects/myProjects/ObjectBox-master/runs/pascal_28/weights/best.pt',
                    help='pretrain weights, checkpoint path, objectbox.pt in test time')
flags.DEFINE_string('hyp',  str(ROOT / 'hyp.yaml'), 'hyperparameters path')

# TRAIN
flags.DEFINE_string('resume', None, 'resume most recent training, weights')
flags.DEFINE_string('name', exp, 'renames experiment folder exp{N} to exp{N}_{name} if supplied')
flags.DEFINE_integer('epochs', 1000, 'max number of epochs')
flags.DEFINE_integer('batch_size', 16, 'total batch size for all GPUs')
flags.DEFINE_string('device', 'cpu', 'cuda device, i.e. 0 or 0,1,2,3 or cpu')
flags.DEFINE_integer('imgsz', 640, 'image sizes, [640, 640]')

flags.DEFINE_float('conf_thres', 0.0001, 'object confidence threshold')
flags.DEFINE_float('iou_thres', 0.45, 'IoU threshold for NMS')
flags.DEFINE_integer('workers', 8, 'maximum number of dataloader workers')
flags.DEFINE_string('task','val', 'train, val, test')

flags.DEFINE_bool('rect', False, "rectangular training")
flags.DEFINE_bool('nosave', False, "only save final checkpoint")
flags.DEFINE_bool('noval', False, "only test final epoch")
flags.DEFINE_string('bucket', '', 'gsutil bucket')
flags.DEFINE_bool('cache_images', False, 'cache images for faster training')
flags.DEFINE_bool('image_weights', False, 'use weighted image selection for training')
flags.DEFINE_bool('multi_scale', False, 'vary img-size +/- 50%%')
flags.DEFINE_bool('single_cls', False, 'train as single-class dataset')
flags.DEFINE_bool('adam', False, 'use torch.optim.Adam() optimizer')
flags.DEFINE_bool('sync_bn', False, 'use SyncBatchNorm, only available in DDP mode')
flags.DEFINE_integer('entity', None, 'W&B entity')
flags.DEFINE_bool('quad', False, 'quad dataloader')
flags.DEFINE_bool('linear_lr', False, 'linear LR')
flags.DEFINE_float('label_smoothing', 0.0, 'Label smoothing epsilon')
flags.DEFINE_bool('upload_dataset', False, 'Upload dataset as W&B artifact table')
flags.DEFINE_integer('bbox_interval', -1, 'Set bounding-box image logging interval for W&B')
flags.DEFINE_integer('save_period', -1, 'Log model after every "save_period" epoch')
flags.DEFINE_string('artifact_alias', 'latest', 'version of dataset artifact to be used')
flags.DEFINE_integer('world_size', None, 'world_size')
flags.DEFINE_integer('global_rank', None, 'global_rank')
flags.DEFINE_integer('local_rank', -1, 'DDP parameter, do not modify')
flags.DEFINE_integer('freeze', 0, 'Number of layers to freeze. backbone=10, all=24')
flags.DEFINE_integer('patience', 100, 'EarlyStopping patience (epochs without improvement)')

flags.DEFINE_string('save_dir', r'/home/ff/myProject/KGT/myProjects/myProjects/ObjectBox-master', 'save results')
flags.DEFINE_bool('save_json', True, 'save a cocoapi-compatible JSON results file')
flags.DEFINE_bool('augment', False, 'augmented inference')
flags.DEFINE_bool('verbose', True, 'report mAP by class')
flags.DEFINE_bool('save_txt', True, 'save results to *.txt')
flags.DEFINE_bool('save_hybrid', False, 'for hybrid auto-labelling')
flags.DEFINE_bool('save_conf', True, 'save auto-label confidences')
flags.DEFINE_string('source',
                    default=r'/home/ff/myProject/KGT/myProjects/myProjects/ObjectBox-master/images',
                    help='file/dir/URL/glob/screen/0(webcam)')  ###
flags.DEFINE_bool('view_img', False, 'show results')
flags.DEFINE_bool('save_crop', False, 'save cropped prediction boxes')
flags.DEFINE_bool('hide_labels', False, 'hide labels')
flags.DEFINE_bool('hide_conf', True, 'hide confidences')
flags.DEFINE_bool('visualize', False, 'visualize features')
flags.DEFINE_float('vis_conf_thres', 0.2, 'confidence threshold for visualization')