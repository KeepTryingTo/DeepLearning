import pickle
import torch.utils.data as data

from data import *
from utils.trainutils import Timer


def _infer(model, args, cfg, dataset,device):
    state = torch.load(args.test_model)
    model.load_state_dict(state['model'])

    num_images = len(dataset)
    #labelmap = (COCO_CLASSES, VOC_CLASSES)['num_classes']
    labelmap = VOC_CLASSES
    all_boxes = [[[] for _ in range(num_images)]
                 for _ in range(len(labelmap)+1)]

    det_file = os.path.join(args.eval_folder, 'detection.pkl')
    data_loader = data.DataLoader(dataset, batch_size=1,
                                  num_workers=0, shuffle=False,
                                  pin_memory=False)
    timer = Timer('')
    print('Start evaluation')

    model.eval()
    for i, batch_data in enumerate(data_loader):

        images, gt, height, width = batch_data
        images = images.to(device)

        timer.tic()
        with torch.no_grad():
            detections = model(images).data
        t = timer.toc()

        for j in range(1, detections.size(1)):

            dets = detections[0, j, :]
            mask = dets[:, 0].gt(0.).expand(5, dets.size(0)).t()
            dets = torch.masked_select(dets, mask).view(-1, 5)

            if dets.size(0) == 0:
                continue
            boxes = dets[:, 1:]
            boxes[:, 0] *= width
            boxes[:, 2] *= width
            boxes[:, 1] *= height
            boxes[:, 3] *= height
            scores = dets[:, 0].cpu().numpy()
            cls_dets = np.hstack((boxes.cpu().numpy(),
                                  scores[:, np.newaxis])).astype(np.float32,
                                                                 copy=False)
            all_boxes[j][i] = cls_dets

        print('Detect %d/%d %.4f'%(i+1, len(dataset), t))

    with open(det_file, 'wb') as f:
        pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

    print("Evaluating detections")
    dataset.evaluate_detections(all_boxes, args.eval_folder)
