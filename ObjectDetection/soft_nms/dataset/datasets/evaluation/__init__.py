from dataset.datasets.voc import VOCDataset
# from dataset.datasets.coco import COCODataset

from dataset.datasets.evaluation.voc.voc_eval import voc_evaluation
from dataset.datasets.evaluation.coco.coco_eval import coco_evaluation


def evaluate(dataset,device, predictions,image_size,
             output_dir, **kwargs):
    """evaluate dataset using different methods based on dataset type.
    Args:
        dataset: Dataset object
        predictions(list[(boxes, labels, scores)]): Each item in the list represents the
            prediction results for one image. And the index should match the dataset index.
        output_dir: output folder, to save evaluation files or results.
    Returns:
        evaluation result
    """
    #TODO output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", 'VOC')
    args = dict(
        dataset=dataset, predictions=predictions,
        image_size = image_size,device = device,
        output_dir=output_dir, **kwargs,
    )
    return voc_evaluation(**args)
    # elif isinstance(dataset, COCODataset):
    #     return coco_evaluation(**args)

