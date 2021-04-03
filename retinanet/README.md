# AdaCon RetinaNet implementation
**NOTE**: This code is modified from [torchvision scripts](https://github.com/pytorch/vision/tree/master/references/detection)

Run Adative Training using 
```
python train.py --dataset coco --model retinanet_resnet50_fpn --epochs 26 
                --lr-steps 16 22 --aspect-ratio-group-factor 3 --lr 0.01 
                --data-path data/coco --batch-size 16 --trainable-backbone-layers 0
                --pretrained-backbone --backbone-weights backbone_coco.pth --adaptive
                --clusters clusters.data --active-branch 0
```


Run Adaptive Inference using 
```
python train.py --dataset coco --model retinanet_resnet50_fpn --epochs 26 
                --lr-steps 16 22 --aspect-ratio-group-factor 3 --lr 0.01 
                --data-path data/coco --batch-size 16 --trainable-backbone-layers 0
                --pretrained-backbone --backbone-weights backbone_coco.pth --adaptive
                --clusters clusters.data --active-branch 0 --test-only
```

Run Baseline Training using 
```
python train.py --dataset coco --model retinanet_resnet50_fpn --epochs 26 
                --lr-steps 16 22 --aspect-ratio-group-factor 3 --lr 0.01 
                --data-path data/coco --batch-size 16 --trainable-backbone-layers 0
                --pretrained-backbone
```


Run Baseline Inference using 
```
python train.py --dataset coco --model retinanet_resnet50_fpn --epochs 26 
                --lr-steps 16 22 --aspect-ratio-group-factor 3 --lr 0.01 
                --data-path data/coco --batch-size 16 --trainable-backbone-layers 0
                --pretrained --test-only
```