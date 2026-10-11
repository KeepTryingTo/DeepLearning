"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2024/10/31-20:05
@CSDN   : https://blog.csdn.net/Keep_Trying_Go?spm=1010.2135.3001.5421
"""

import logging
import os

import torch
import torch.utils.data
from tqdm import tqdm

from dataset.datasets.evaluation import evaluate
from utiles.mk import mkdir
from utiles import dist_util
from dataset.build import make_data_loader

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


from dataset.datasets.voc import BatchCollator
from configs.defaults import _C as cfg
from dataset.build_transforms import build_transforms

def compute_on_dataset(model, data_loader, device):
    results_dict = []
    for batch in tqdm(data_loader):
        images, targets, image_ids = batch
        cpu_device = torch.device("cpu")
        with torch.no_grad():
            detections, cls_logits, bbox_pred = model(images.to(device))

            outputs = [o.to(cpu_device) for o in detections]
        results_dict.append(
            {img_id: result for img_id, result in zip(image_ids, outputs)}
        )

    return results_dict


def inference(model, data_loader,
              dataset_name, device,image_size,
              output_folder=None,
              use_cached=False, **kwargs):
    dataset = data_loader.dataset
    logger = logging.getLogger("DSSD.inference")
    logger.info("Evaluating {} dataset({} images):".format(dataset_name, len(dataset)))
    predictions_path = os.path.join(output_folder, 'predictions.pth')
    #TODO 是否保存当前的模型并且默认的use_cached = False
    if use_cached and os.path.exists(predictions_path):
        predictions = torch.load(predictions_path, map_location='cpu')
    else:
        predictions = compute_on_dataset(model, data_loader, device)

    #TODO 默认是会保存这个模型文件的
    if output_folder:
        torch.save(predictions, predictions_path)
    return evaluate(dataset=dataset,device = device,
                    predictions=predictions,image_size = image_size,
                    output_dir=output_folder,
                    **kwargs)

@torch.no_grad()
def do_evaluation(cfg, model,device = 'cpu',  mode="test", **kwargs):
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model = model.module
    model.eval()
    _dataset = cfg.DATASETS.VAL if mode=="val" else cfg.DATASETS.TEST

    data_loaders_val = make_data_loader(cfg, is_train=False)
    eval_results = []
    #TODO 要验证的数据名称
    _dataset = cfg.DATASETS.VAL if mode == "val" else cfg.DATASETS.TEST

    image_size = cfg.INPUT.IMAGE_SIZE

    for dataset_name, data_loader in zip(_dataset, data_loaders_val):
        #TODO 保存结果的文件
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
        if not os.path.exists(output_folder):
            mkdir(output_folder)
        #TODO 正式开始数据集验证
        eval_result = inference(model, data_loader, dataset_name,
                                device,image_size, output_folder, **kwargs)
        eval_results.append(eval_result)
    eval_results.append(eval_result)

    return eval_results

def createCfg(config_file = r'configs/resnet101_dssd320_voc0712.yaml'):
    cfg.merge_from_file(config_file)
    cfg.freeze()
    return cfg


if __name__ == '__main__':
    from models.dssd_detector import DSSDDetector
    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'

    cfg = createCfg(config_file=r'../configs/resnet101_dssd320_voc0712.yaml')
    model = DSSDDetector(cfg)
    checkpoint = torch.load(r'./weights/voc_265_dssd.pth.tar',
                            map_location='cpu')['model']
    model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()
    print("INFO===>success loading model")

    do_evaluation(
        cfg=cfg,
        model=model,
        device=device,
        mode='test'
    )
    pass
