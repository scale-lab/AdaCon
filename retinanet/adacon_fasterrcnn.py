from torchvision.models.detection.backbone_utils import mobilenet_backbone, _validate_trainable_layers
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.utils import load_state_dict_from_url
from torchvision.ops import MultiScaleRoIAlign
from torchvision.models.detection.rpn import RPNHead, RegionProposalNetwork
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.models.detection.faster_rcnn import TwoMLPHead, FastRCNNPredictor, _fasterrcnn_mobilenet_v3_large_fpn
# from torchvision.models.detection.generalized_rcnn import GeneralizedRCNN
from adacon_model import get_class_to_cluster_map, get_cluster_to_class_map
import torch.nn.functional as F
from torchvision.ops import boxes as box_ops

from collections import OrderedDict
import torch
from torch import nn, Tensor
import warnings
from typing import Tuple, List, Dict, Optional, Union
import copy 

model_urls = {
    'fasterrcnn_resnet50_fpn_coco':
        'https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth',
    'fasterrcnn_mobilenet_v3_large_320_fpn_coco':
        'https://download.pytorch.org/models/fasterrcnn_mobilenet_v3_large_320_fpn-907ea3f9.pth',
    'fasterrcnn_mobilenet_v3_large_fpn_coco':
        'https://download.pytorch.org/models/fasterrcnn_mobilenet_v3_large_fpn-fb6a3cc7.pth'
}

def _adacon_fasterrcnn_mobilenet_v3_large_fpn(weights_name, pretrained=False, progress=True, num_classes=91,
                                       pretrained_backbone=True, trainable_backbone_layers=None, ckpt=None,
                                       branches_weights=None, backbone_weights=None, bc_weights=None, num_branches=0, **kwargs):
    trainable_backbone_layers = _validate_trainable_layers(
        pretrained or pretrained_backbone, trainable_backbone_layers, 6, 3)

    if pretrained:
        pretrained_backbone = False
    backbone = mobilenet_backbone("mobilenet_v3_large", pretrained_backbone, True,
                                  trainable_layers=trainable_backbone_layers)

    anchor_sizes = ((32, 64, 128, 256, 512, ), ) * 3
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)

    model = AdaConFasterRCNN(backbone, num_classes, rpn_anchor_generator=AnchorGenerator(anchor_sizes, aspect_ratios),
                       **kwargs)
    if pretrained_backbone:
        print("Loaded Backbone ", backbone_weights)
        model.backbone.load_state_dict(torch.load(backbone_weights)['backbone'])
        model.rpn.load_state_dict(torch.load(backbone_weights)['rpn'])

    if pretrained:
        if backbone_weights:
            print("Loading backbone")
            model.backbone.load_state_dict(torch.load(backbone_weights)['backbone'])
            model.rpn.load_state_dict(torch.load(backbone_weights)['rpn'])
            if branches_weights:
                print("Loading Branches")
                for branch in range(num_branches):
                    model.roi_heads[branch].load_state_dict(torch.load(branches_weights[branch])['head'])
            if bc_weights:
                print("Loading branch controller")
                model.branch_controller.load_state_dict(torch.load(bc_weights)['branch_controller'])
        elif ckpt:
            model.load_state_dict(torch.load(ckpt), strict=False)
        else:
            if model_urls.get(weights_name, None) is None:
                raise ValueError("No checkpoint is available for model {}".format(weights_name))
            state_dict = load_state_dict_from_url(model_urls[weights_name], progress=progress)
            model.load_state_dict(state_dict)
    return model

def fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=False, progress=True, num_classes=91, pretrained_backbone=True,
                                          trainable_backbone_layers=None, **kwargs):
    weights_name = "fasterrcnn_mobilenet_v3_large_320_fpn_coco"
    defaults = {
        "min_size": 320,
        "max_size": 640,
        "rpn_pre_nms_top_n_test": 150,
        "rpn_post_nms_top_n_test": 150,
        "rpn_score_thresh": 0.05,
    }

    kwargs = {**defaults, **kwargs}
    return _fasterrcnn_mobilenet_v3_large_fpn(weights_name, pretrained=pretrained, progress=progress,
                                              num_classes=num_classes, pretrained_backbone=pretrained_backbone,
                                              trainable_backbone_layers=trainable_backbone_layers, **kwargs)

def adacon_fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=False, progress=True, num_classes=91, pretrained_backbone=True,
                                          trainable_backbone_layers=None, branches_weights=None, ckpt=None,
                                          backbone_weights=None, bc_weights=None, num_branches=0, **kwargs):
    weights_name = "fasterrcnn_mobilenet_v3_large_320_fpn_coco"
    defaults = {
        "min_size": 320,
        "max_size": 640,
        "rpn_pre_nms_top_n_test": 150,
        "rpn_post_nms_top_n_test": 150,
        "rpn_score_thresh": 0.05,
    }

    kwargs = {**defaults, **kwargs}
    return _adacon_fasterrcnn_mobilenet_v3_large_fpn(weights_name, pretrained=pretrained, progress=progress,
                                              num_classes=num_classes, pretrained_backbone=pretrained_backbone, ckpt=ckpt,
                                              trainable_backbone_layers=trainable_backbone_layers, num_branches=num_branches,
                                              branches_weights=branches_weights, backbone_weights=backbone_weights, bc_weights=bc_weights, **kwargs)


class AdaConFasterRCNN(nn.Module):
    def __init__(self, backbone, num_classes,
                 clusters, 
                 active_branch=0, num_branches=4,
                 oracle=False, singleb=False,
                 enable_branch_controller=False,
                 # transform parameters
                 min_size=800, max_size=1333,
                 image_mean=None, image_std=None,
                 # RPN parameters
                 rpn_anchor_generator=None, rpn_head=None,
                 rpn_pre_nms_top_n_train=2000, rpn_pre_nms_top_n_test=1000,
                 rpn_post_nms_top_n_train=2000, rpn_post_nms_top_n_test=1000,
                 rpn_nms_thresh=0.7,
                 rpn_fg_iou_thresh=0.7, rpn_bg_iou_thresh=0.3,
                 rpn_batch_size_per_image=256, rpn_positive_fraction=0.5,
                 rpn_score_thresh=0.0,
                 # Box parameters
                 box_roi_pool=None, box_head=None, box_predictor=None,
                 box_score_thresh=0.05, box_nms_thresh=0.5, box_detections_per_img=100,
                 box_fg_iou_thresh=0.5, box_bg_iou_thresh=0.5,
                 box_batch_size_per_image=512, box_positive_fraction=0.25,
                 bbox_reg_weights=None):
        super().__init__()

        if not hasattr(backbone, "out_channels"):
            raise ValueError(
                "backbone should contain an attribute out_channels "
                "specifying the number of output channels (assumed to be the "
                "same for all the levels)")

        assert isinstance(rpn_anchor_generator, (AnchorGenerator, type(None)))
        assert isinstance(box_roi_pool, (MultiScaleRoIAlign, type(None)))

        self.active_branch = active_branch
        self.clusters = clusters
        self.num_branches = len(clusters)
        self.class_to_cluster_dict = get_class_to_cluster_map(clusters)
        self.cluster_to_class_map = get_cluster_to_class_map(clusters)
        self.oracle = oracle
        self.singleb = singleb
        self.enable_branch_controller = enable_branch_controller
        
        if num_classes is not None:
            if box_predictor is not None:
                raise ValueError("num_classes should be None when box_predictor is specified")
        else:
            if box_predictor is None:
                raise ValueError("num_classes should not be None when box_predictor "
                                 "is not specified")
        self.backbone = backbone
        out_channels = backbone.out_channels

        if rpn_anchor_generator is None:
            anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
            aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
            rpn_anchor_generator = AnchorGenerator(
                anchor_sizes, aspect_ratios
            )
        if rpn_head is None:
            rpn_head = RPNHead(
                out_channels, rpn_anchor_generator.num_anchors_per_location()[0]
            )

        rpn_pre_nms_top_n = dict(training=rpn_pre_nms_top_n_train, testing=rpn_pre_nms_top_n_test)
        rpn_post_nms_top_n = dict(training=rpn_post_nms_top_n_train, testing=rpn_post_nms_top_n_test)

        self.rpn = RegionProposalNetwork(
            rpn_anchor_generator, rpn_head,
            rpn_fg_iou_thresh, rpn_bg_iou_thresh,
            rpn_batch_size_per_image, rpn_positive_fraction,
            rpn_pre_nms_top_n, rpn_post_nms_top_n, rpn_nms_thresh,
            score_thresh=rpn_score_thresh)

        heads = nn.ModuleList()
        for i in range(self.num_branches):
            box_roi_pool = MultiScaleRoIAlign(
                featmap_names=['0', '1', '2', '3'],
                output_size=7,
                sampling_ratio=2)

            resolution = box_roi_pool.output_size[0]
            representation_size = 1024 // 2
            box_head = TwoMLPHead(
                out_channels * resolution ** 2,
                representation_size)

            box_predictor = FastRCNNPredictor(
                representation_size,
                len(self.clusters[i]))

            heads.append(RoIHeads(
                box_roi_pool, box_head, box_predictor,
                box_fg_iou_thresh, box_bg_iou_thresh,
                box_batch_size_per_image, box_positive_fraction,
                bbox_reg_weights,
                box_score_thresh, box_nms_thresh, box_detections_per_img))
        self.roi_heads = heads
        print(len(self.roi_heads))
        self.branch_controller = None ## TODO: implement Branch controller

        if image_mean is None:
            image_mean = [0.485, 0.456, 0.406]
        if image_std is None:
            image_std = [0.229, 0.224, 0.225]
        self.transform = GeneralizedRCNNTransform(min_size, max_size, image_mean, image_std)
        
        self._has_warned = False

    @torch.jit.unused
    def eager_outputs(self, losses, detections):
        # type: (Dict[str, Tensor], List[Dict[str, Tensor]]) -> Union[Dict[str, Tensor], List[Dict[str, Tensor]]]
        if self.training:
            return losses

        return detections

    def forward(self, images, targets=None):
        # type: (List[Tensor], Optional[List[Dict[str, Tensor]]]) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]
        """
        Args:
            images (list[Tensor]): images to be processed
            targets (list[Dict[Tensor]]): ground-truth boxes present in the image (optional)
        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).
        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        if self.training:
            assert targets is not None
            for target in targets:
                boxes = target["boxes"]
                if isinstance(boxes, torch.Tensor):
                    if len(boxes.shape) != 2 or boxes.shape[-1] != 4:
                        raise ValueError("Expected target boxes to be a tensor"
                                         "of shape [N, 4], got {:}.".format(
                                             boxes.shape))
                else:
                    raise ValueError("Expected target boxes to be of type "
                                     "Tensor, got {:}.".format(type(boxes)))

        original_image_sizes: List[Tuple[int, int]] = []
        for img in images:
            val = img.shape[-2:]
            assert len(val) == 2
            original_image_sizes.append((val[0], val[1]))

        images, targets = self.transform(images, targets)

        # Check for degenerate boxes
        # TODO: Move this to a function
        if targets is not None:
            for target_idx, target in enumerate(targets):
                boxes = target["boxes"]
                degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
                if degenerate_boxes.any():
                    # print the first degenerate box
                    bb_idx = torch.where(degenerate_boxes.any(dim=1))[0][0]
                    degen_bb: List[float] = boxes[bb_idx].tolist()
                    raise ValueError("All bounding boxes should have positive height and width."
                                     " Found invalid box {} for target at index {}."
                                     .format(degen_bb, target_idx))

        features = self.backbone(images.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([('0', features)])
        proposals, proposal_losses = self.rpn(images, features, targets)
        
        if not self.training and self.oracle:
            combined_detections = []
            for i in range(self.num_branches):
                self.active_branch = i
                # targets = self._get_cluster_targets(targets)

                detections, detector_losses = self.roi_heads[self.active_branch](features, proposals, images.image_sizes, targets)
                detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)

                detections = self._map_outputs_from_cluster(detections)
                combined_detections.append(detections)

            detections[0]['scores'] = torch.cat([detection[0]['scores'] for detection in combined_detections])
            detections[0]['labels'] = torch.cat([detection[0]['labels'] for detection in combined_detections])
            detections[0]['boxes'] = torch.cat([detection[0]['boxes'] for detection in combined_detections])
            keep = box_ops.batched_nms(detections[0]['boxes'], detections[0]['scores'], detections[0]['labels'], 0.5)
            # print("keep",len(keep))
            detections[0]['scores'] = detections[0]['scores'][keep]
            detections[0]['labels'] = detections[0]['labels'][keep]
            detections[0]['boxes'] = detections[0]['boxes'][keep]
        else:
            targets = self._get_cluster_targets(targets)

            detections, detector_losses = self.roi_heads[self.active_branch](features, proposals, images.image_sizes, targets)
            detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)

            detections = self._map_outputs_from_cluster(detections)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)

        if torch.jit.is_scripting():
            if not self._has_warned:
                warnings.warn("RCNN always returns a (Losses, Detections) tuple in scripting")
                self._has_warned = True
            return losses, detections
        else:
            return self.eager_outputs(losses, detections)

    def _get_cluster_targets(self, targets):
        modified_targets = []
        for i, targets_per_image in enumerate(targets):
            targets_per_image = self._map_targets_to_cluster(targets_per_image)
            if targets_per_image is not None:
                modified_targets.append(targets_per_image)

        return modified_targets

    def _map_targets_to_cluster(self, targets_per_image):
        labels = []
        boxes = []
        for l, b in zip(targets_per_image['labels'], targets_per_image['boxes']):
            if l in self.clusters[self.active_branch]:
                new_label = self.class_to_cluster_dict[self.active_branch][l.item()]
                labels.append(torch.tensor(new_label, device=l.get_device()))
                boxes.append(b)
        if len(labels) > 0 and len(boxes) > 0:
            return {'labels': torch.stack(labels, dim=0), 
                    'boxes': torch.stack(boxes, dim=0)}
        else:
            return None
        
    def _map_outputs_from_cluster(self, detections):
        for i, detection in enumerate(detections):
            labels = []
            for l in detection['labels']:
                new_label = self.cluster_to_class_map[self.active_branch][l.item()]
                labels.append(torch.tensor(new_label, device=l.get_device()))
            if len(labels) > 0:
                detections[i]['labels'] = torch.stack(labels, dim=0)
        return detections

class AdaConRCNNBranchController(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        conv = []
        conv.append(nn.Conv2d(in_channels, in_channels//2, kernel_size=3, stride=2, padding=1))
        conv.append(nn.ReLU())
        conv.append(nn.Conv2d(in_channels//2, in_channels//2, kernel_size=3, stride=2, padding=1))
        conv.append(nn.ReLU())

        self.conv = nn.Sequential(*conv)
        
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, num_classes)
        self.num_classes = num_classes

    def compute_loss(self, targets, outputs):
        # type: (List[Dict[str, Tensor]], Dict[str, Tensor], List[Tensor], List[Tensor]) -> Dict[str, Tensor]
        loss = nn.CrossEntropyLoss()
        return {'bc_loss': loss(outputs, targets)}

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x)