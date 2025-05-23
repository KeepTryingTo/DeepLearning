import json
import tempfile

from pycocotools.cocoeval import COCOeval
from torch.autograd import Variable

from dataset.cocodataset import *
from utils.utils import *


class COCOAPIEvaluator():
    """
    COCO AP Evaluation class.
    All the data in the val2017 dataset are processed \
    and evaluated by COCO API.
    """
    def __init__(self, model_type, data_dir, img_size, confthre, nmsthre):
        """
        Args:
            model_type (str): model name specified in config file
            data_dir (str): dataset root directory
            img_size (int): image size after preprocess. images are resized \
                to squares whose shape is (img_size, img_size).
            confthre (float):
                confidence threshold ranging from 0 to 1, \
                which is defined in the config file.
            nmsthre (float):
                IoU threshold of non-max supression ranging from 0 to 1.
        """

        augmentation = {'LRFLIP': False, 'JITTER': 0, 'RANDOM_PLACING': False,
                        'HUE': 0, 'SATURATION': 0, 'EXPOSURE': 0, 'RANDOM_DISTORT': False}

        self.dataset = COCODataset(model_type=model_type,
                                   data_dir=data_dir,
                                   img_size=img_size,
                                   augmentation=augmentation,
                                   json_file='instances_val2014.json',
                                   name='val2014')
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset, batch_size=1, shuffle=False, num_workers=0)
        self.img_size = img_size
        self.confthre = 0.005 # from darknet
        self.nmsthre = nmsthre # 0.45 (darknet)

    def evaluate(self, model):
        """
        COCO average precision (aP) Evaluation. Iterate inference on the test dataset
        and the results are evaluated by COCO API.
        Args:
            model : model object
        Returns:
            ap (dict) : 
                dict containing COCO average precisions with various IoU threshold \
                and object sizes. If no object is detected, False is assigned for \
                'valid' field and all aP fields are set to 0.
        """
        model.eval()
        cuda = torch.cuda.is_available()
        Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        ids = []
        data_dict = []
        dataiterator = iter(self.dataloader)
        while True: # all the data in val2017
            try:
                img, _, info_img, id_ = next(dataiterator)  # load a batch
            except StopIteration:
                break
            info_img = [float(info) for info in info_img]
            id_ = int(id_)
            ids.append(id_)
            with torch.no_grad():
                img = Variable(img.type(Tensor))
                outputs = model(img)
                outputs = postprocess(
                    prediction=outputs,
                    num_classes=80,
                    conf_thre=self.confthre,
                    nms_thre=self.nmsthre
                )
                if outputs[0] is None:
                    continue
                outputs = outputs[0].cpu().data

            for output in outputs:
                x1 = float(output[0])
                y1 = float(output[1])
                x2 = float(output[2])
                y2 = float(output[3])
                label = self.dataset.class_ids[int(output[6])]
                box = yolobox2label((y1, x1, y2, x2), info_img)
                bbox = [box[1], box[0], box[3] - box[1], box[2] - box[0]]
                score = float(output[4].data.item() * output[5].data.item()) # object score * class score
                A = {"image_id": id_, "category_id": label, "bbox": bbox,
                     "score": score, "segmentation": []} # COCO json format
                data_dict.append(A)

            print('evaluating image {}'.format(id_))

        annType = ['segm', 'bbox', 'keypoints']

        ap = {
            'valid': False,
            'aP5095': 0.0,
            'aP50': 0.0,
            'aP75': 0.0,
            'aP5095_S': 0.0,
            'aP5095_M': 0.0,
            'aP5095_L': 0.0
        }

        # Evaluate the Dt (detection) json comparing with the ground truth
        if len(data_dict) > 0:
            cocoGt = self.dataset.coco
            # workaround: temporarily write data to json file because pycocotools can't process dict in py36.
            _, tmp = tempfile.mkstemp()
            json.dump(data_dict, open(tmp, 'w'))
            cocoDt = cocoGt.loadRes(tmp)
            cocoEval = COCOeval(self.dataset.coco, cocoDt, annType[1])
            cocoEval.params.imgIds = ids
            cocoEval.evaluate()
            cocoEval.accumulate()
            cocoEval.summarize()

            ap['valid'] = True
            ap['aP5095'] = cocoEval.stats[0]
            ap['aP50'] = cocoEval.stats[1]
            ap['aP75'] = cocoEval.stats[2]
            ap['aP5095_S'] = cocoEval.stats[3]
            ap['aP5095_M'] = cocoEval.stats[4]
            ap['aP5095_L'] = cocoEval.stats[5]

        return ap
