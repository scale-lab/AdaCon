import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.ops.boxes import box_convert
import torch 
import random
from PIL import Image
import os
from pycocotools.coco import COCO
from utils.utils import *
import cv2
import torch.nn.functional as F

def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image

class COCODataset(datasets.CocoDetection):
    def __init__(
        self, root, annFile, train=False, multiscale=False, img_size=416, transform=None, target_transform=None, transforms=None):
        super().__init__(root, transforms, transform, target_transform)

        self.coco = COCO(annFile)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.img_size = img_size
        self.multiscale = multiscale
        self.min_size = self.img_size - 3 * 32
        self.max_size = self.img_size + 3 * 32
        self.batch_count = 0
        self.train = train

    def __getitem__(self, index: int):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)

        path = coco.loadImgs(img_id)[0]['file_name']

        img = Image.open(os.path.join(self.root, path)).convert('RGB')

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img_id, os.path.join(self.root, path), img, target, img.shape

    def collate_fn(self, batch):
        ids, paths, imgs, targets, shapes = list(zip(*batch))

        # Selects new image size every tenth batch
        if self.multiscale and self.batch_count % 10 == 0:
            self.img_size = random.choice(range(self.min_size, self.max_size + 1, 32))
        
        # Resize images to input shape
        if self.train:
            imgs = torch.stack([resize(img, self.img_size) for img in imgs])
        
        new_shapes = [img.shape for img in imgs]

        self.batch_count += 1
        reg_targets = []
        cls_targets = []

        for target, shape, new_shape in zip(targets, shapes, new_shapes):
            if len(target) == 0:
                continue
            
            x_scale = new_shape[1]/shape[1]
            y_scale = new_shape[2]/shape[2]

            boxes = box_convert(torch.tensor([batch_target['bbox'] for batch_target in target]), 'xywh', 'xyxy') 
            boxes[:,0] *= x_scale
            boxes[:,1] *= y_scale
            boxes[:,2] *= x_scale
            boxes[:,3] *= y_scale

            reg_targets.append(boxes)
            cls_targets.append(torch.tensor([batch_target['category_id'] for batch_target in target]))

        return imgs, paths, reg_targets, cls_targets, ids, shapes