import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'

import argparse
import torch
from torchvision import transforms

from retinanet import model
from retinanet.dataloader import CocoDataset, Resizer, Normalizer
from retinanet import coco_eval

assert torch.__version__.split('.')[0] == '1'

print('CUDA available: {}'.format(torch.cuda.is_available()))


def main(args=None):
    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')

    parser.add_argument('--coco_path', type=str,
                        default=r'/home/ff/myProject/KGT/myProjects/myDataset/coco',
                        help='Path to COCO directory')
    parser.add_argument('--model_path', help='Path to model', type=str,
                        default=r'./weights/coco_retinanet_3.pt')

    parser = parser.parse_args(args)

    dataset_val = CocoDataset(parser.coco_path, set_name='val2014',
                              transform=transforms.Compose([Normalizer(), Resizer()]))

    # Create the model
    retinanet = model.resnet50(num_classes=dataset_val.num_classes(),
                               pretrained=True)

    checkpoint = torch.load(parser.model_path,map_location='cpu')
    from collections import OrderedDict
    state_dict = OrderedDict()
    for key,values in checkpoint.items():
        if 'module.' in key:
            state_dict[key[7:]] = values
        else:
            state_dict[key] = values

    use_gpu = True

    if use_gpu:
        if torch.cuda.is_available():
            retinanet = retinanet.cuda()

    if torch.cuda.is_available():
        retinanet.load_state_dict(state_dict)
        retinanet = torch.nn.DataParallel(retinanet).cuda()
    else:
        retinanet.load_state_dict(state_dict)
        retinanet = torch.nn.DataParallel(retinanet)

    retinanet.training = False
    retinanet.eval()
    retinanet.module.freeze_bn()

    coco_eval.evaluate_coco(dataset_val, retinanet)


if __name__ == '__main__':
    main()
