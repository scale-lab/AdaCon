import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.ops.boxes import box_convert
import torchvision
import torch 
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import json
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from utils.utils import *
from utils.datasets_native import COCODataset
from model.retinanet import retinanet_resnet50_fpn, retinanet_mobilenet_fpn, adacon_retinanet_resnet50_fpn

def evaluate_coco(val_dataloader, model, detection_threshold=0.1):
    model.eval()

    with torch.no_grad():
        results = []
        image_ids = []
        for batch_i, (imgs, paths, reg_targets, cls_targets, ids, shapes) in enumerate(tqdm(val_dataloader)):
            image_ids.append(ids[0])
            image_array = np.array(imgs[0])
            imgs = imgs.to(device)
            outputs = model(imgs) # get the predictions on the image

            # get all the scores
            scores = list(outputs[0]['scores'].detach().cpu().numpy())

            # index of those scores which are above a certain threshold
            thresholded_preds_inidices = [scores.index(i) for i in scores if i > detection_threshold]
            
            # get all the predicted bounding boxes
            bboxes = box_convert(outputs[0]['boxes'], 'xyxy', 'xywh').detach().cpu().numpy()
            
            # get boxes above the threshold score
            boxes = bboxes[np.array(scores) >= detection_threshold].astype(np.int32)
            
            # get all the predicited class names
            labels = outputs[0]['labels'].cpu().numpy()
            labels = [labels[i] for i in thresholded_preds_inidices]

            scores = [scores[i] for i in thresholded_preds_inidices]

            if boxes.shape[0] > 0:
                for box, label, score in zip(boxes, labels, scores):
                    # append detection for each positively labeled class
                    image_result = {
                        'image_id'    : ids[0],
                        'category_id' : int(label),
                        'score'       : float(score),
                        'bbox'        : box.tolist(),
                    }

                    # append detection to results
                    results.append(image_result)

        if not len(results):
            print("No Results !!")
            return

        # write output
        json.dump(results, open('coco_bbox_results.json', 'w'), indent=4)

        cocoGt = COCO(val_ann)  # initialize COCO ground truth api
        cocoDt = cocoGt.loadRes('coco_bbox_results.json')  # initialize COCO pred api
        
        # run COCO evaluation
        cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')

        cocoEval.params.imgIds = image_ids  # [:32]  # only evaluate these images

        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()

        return cocoEval.stats

if __name__ == '__main__':
    val_ann = "data/coco/annotations/instances_val2017.json"
    val_root = "data/coco/images/val2017/"

    weights = None

    batch_size = 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    val_dataset = COCODataset(root=val_root,
                            annFile=val_ann,
                            transform=transforms.ToTensor())

    val_dataloader = DataLoader(val_dataset,
                                batch_size=batch_size,
                                num_workers=min(1,8),
                                pin_memory=True,
                                collate_fn=val_dataset.collate_fn)

    # model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    model = retinanet_resnet50_fpn(pretrained=True)

    if weights:
        model.load_state_dict(torch.load(weights))

    model.eval()
    count_parameters(model)
    model.to(device)

    evaluate_coco(val_dataloader, model, detection_threshold=0.1)