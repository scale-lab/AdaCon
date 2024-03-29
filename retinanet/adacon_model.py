##############################################################################
################ THIS CODE IS ADAPTED FROM CODE AT https://github.com/pytorch/vision/blob/master/torchvision/models/detection/retinanet.py
##############################################################################

from torchvision.models.detection.backbone_utils import resnet_fpn_backbone, _validate_trainable_layers
from torchvision.models.utils import load_state_dict_from_url
from torchvision.models.detection._utils import overwrite_eps
from torchvision.models.detection import _utils as det_utils
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.models.detection.retinanet import RetinaNet, _sum
from torchvision.ops.feature_pyramid_network import LastLevelP6P7
from torchvision.ops.focal_loss import sigmoid_focal_loss
from torchvision.ops import boxes as box_ops
from torchvision.models.detection.anchor_utils import AnchorGenerator

import math 

import torchvision
import torch
from torch import nn
import torch.nn.functional as F

import warnings
from collections import OrderedDict

def get_class_to_cluster_map(clusters_list):
    class_to_cluster_list = []
    ## create the class-cluster map to be used for labels in split training
    for cluster in clusters_list:
        class_to_cluster = {}
        for i, element in enumerate(cluster):
            class_to_cluster[element] = i

        class_to_cluster_list.append(class_to_cluster)

    return class_to_cluster_list

def get_cluster_to_class_map(clusters_list):
    cluster_to_class = []
    ## create the class-cluster map to be used for labels in split training
    for cluster in clusters_list:
        class_to_cluster = {}
        for i, element in enumerate(cluster):
            class_to_cluster[i] = element

        cluster_to_class.append(class_to_cluster)

    return cluster_to_class

model_urls = {
    'retinanet_resnet50_fpn_coco':
        'https://download.pytorch.org/models/retinanet_resnet50_fpn_coco-eeacb38b.pth',
}

def retinanet_resnet50_fpn(pretrained=False, progress=True,
                           num_classes=91, pretrained_backbone=True, trainable_backbone_layers=None, 
                           ckpt=None, backbone_weights=None, head_weights=None, **kwargs):
    trainable_backbone_layers = _validate_trainable_layers(
        pretrained or pretrained_backbone, trainable_backbone_layers, 5, 3)

    if pretrained:
        # no need to download the backbone if pretrained is set
        pretrained_backbone = False

    # skip P2 because it generates too many anchors (according to their paper)
    backbone = resnet_fpn_backbone('resnet50', pretrained_backbone, returned_layers=[2, 3, 4],
                                   extra_blocks=LastLevelP6P7(256, 256), trainable_layers=trainable_backbone_layers)
    model = RetinaNet(backbone, num_classes, **kwargs)
    
    if pretrained_backbone:
        model.backbone.load_state_dict(torch.load(backbone_weights))
        # overwrite_eps(model, 0.0)
    
    if pretrained:
        if ckpt:
            # model.load_state_dict(torch.load(ckpt), strict=False)
            print("Loading ", backbone_weights)
            model.backbone.load_state_dict(torch.load(backbone_weights))
            model.head.load_state_dict(torch.load(head_weights))
        else:
            state_dict = load_state_dict_from_url(model_urls['retinanet_resnet50_fpn_coco'],
                                                progress=progress)
            model.load_state_dict(state_dict)

        # torch.save(model.state_dict(), "retinanet_coco.pt")
        overwrite_eps(model, 0.0)
    return model

def adacon_retinanet_resnet50_fpn(clusters, active_branch=0, num_branches=4, pretrained=False, progress=True,
                                  num_classes=91, pretrained_backbone=True, trainable_backbone_layers=None,
                                  ckpt=None, backbone_weights=None, pretrained_branches=None, branches_weights=None,
                                  bc_weights=None, deploy=None, **kwargs):
    print("Trainable backbone layers", trainable_backbone_layers)
    trainable_backbone_layers = _validate_trainable_layers(
        pretrained or pretrained_backbone, trainable_backbone_layers, 5, 0)
    print("Trainable backbone layers", trainable_backbone_layers)

    if pretrained:
        # no need to download the backbone if pretrained is set
        pretrained_backbone = False
    
    # skip P2 because it generates too many anchors (according to their paper)
    backbone = resnet_fpn_backbone('resnet50', pretrained_backbone, returned_layers=[2, 3, 4],
                                   extra_blocks=LastLevelP6P7(256, 256), trainable_layers=trainable_backbone_layers)
    model = AdaConRetinaNet(backbone, num_classes, clusters,
                     active_branch=active_branch, num_branches=num_branches, **kwargs)

    if pretrained_backbone:
        print("Loaded Backbone ", backbone_weights)
        model.backbone.load_state_dict(torch.load(backbone_weights)['model'])
        # overwrite_eps(model, 0.0)

    if pretrained:
        if deploy:
            state_dict = torch.load(deploy)
            print("Loading Backbone")
            model.backbone.load_state_dict(state_dict['backbone'])
            print("Loading BC")
            model.branch_controller.load_state_dict(state_dict['branch_controller'])
            for i in range(num_branches):
                print("Loading Head", i)
                model.heads[i].load_state_dict(state_dict['heads'][i])
        elif backbone_weights:
            print("Loading backbone")
            model.backbone.load_state_dict(torch.load(backbone_weights)['model'])
            if branches_weights:
                print("Loading Branches")
                for branch in range(num_branches):
                    model.heads[branch].load_state_dict(torch.load(branches_weights[branch])['head'], strict=False)
            if bc_weights:
                print("Loading branch controller")
                model.branch_controller.load_state_dict(torch.load(bc_weights)['branch_controller'])
        elif ckpt:
            model.load_state_dict(torch.load(ckpt), strict=False)
        else:
            state_dict = load_state_dict_from_url(model_urls['retinanet_resnet50_fpn_coco'],
                                                progress=progress)
            model.load_state_dict(state_dict)

        overwrite_eps(model, 0.0)

    return model

def retinanet_mobilenet_fpn(pretrained=False, progress=True,
                           num_classes=91, pretrained_backbone=True, trainable_backbone_layers=None, **kwargs):
    trainable_backbone_layers = _validate_trainable_layers(
        pretrained or pretrained_backbone, trainable_backbone_layers, 5, 3)

    if pretrained:
        # no need to download the backbone if pretrained is set
        pretrained_backbone = False
    
    # skip P2 because it generates too many anchors (according to their paper)
    backbone = torchvision.models.mobilenet_v2(pretrained=pretrained_backbone).features
    backbone.out_channels = 1280

    anchor_generator = AnchorGenerator(
        sizes=((32, 64, 128, 256, 512),),
        aspect_ratios=((0.5, 1.0, 2.0),)
        )

    model = RetinaNet(backbone, num_classes, anchor_generator=anchor_generator, **kwargs)
    # if pretrained:
    #     state_dict = load_state_dict_from_url(model_urls['retinanet_resnet50_fpn_coco'],
    #                                           progress=progress)
    #     model.load_state_dict(state_dict)
    #     overwrite_eps(model, 0.0)
    return model

class AdaConRetinaNet(nn.Module):
    __annotations__ = {
        'box_coder': det_utils.BoxCoder,
        'proposal_matcher': det_utils.Matcher,
    }

    def __init__(self, backbone, num_classes, 
                 clusters, 
                 active_branch=0, num_branches=4,
                 # transform parameters
                 min_size=800, max_size=1333,
                 image_mean=None, image_std=None,
                 # Anchor parameters
                 anchor_generator=None, heads=None,
                 proposal_matcher=None,
                 score_thresh=0.05,
                 nms_thresh=0.5,
                 detections_per_img=300,
                 fg_iou_thresh=0.5, bg_iou_thresh=0.4,
                 topk_candidates=1000, oracle=False, singleb=False, 
                 multib=False, bc_thres=0.4,
                 enable_branch_controller=False):
        super().__init__()

        if not hasattr(backbone, "out_channels"):
            raise ValueError(
                "backbone should contain an attribute out_channels "
                "specifying the number of output channels (assumed to be the "
                "same for all the levels)")
        self.backbone = backbone

        self.active_branch = active_branch
        self.clusters = clusters
        self.num_branches = len(clusters)
        self.class_to_cluster_dict = get_class_to_cluster_map(clusters)
        self.cluster_to_class_map = get_cluster_to_class_map(clusters)
        self.oracle = oracle
        self.singleb = singleb
        self.multib = multib
        self.bc_thres = bc_thres
        self.enable_branch_controller = enable_branch_controller
        assert isinstance(anchor_generator, (AnchorGenerator, type(None)))

        if anchor_generator is None:
            anchor_sizes = tuple((x, int(x * 2 ** (1.0 / 3)), int(x * 2 ** (2.0 / 3))) for x in [32, 64, 128, 256, 512])
            aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
            anchor_generator = AnchorGenerator(
                anchor_sizes, aspect_ratios
            )
        self.anchor_generator = anchor_generator

        if heads is None:
            heads = nn.ModuleList()
            for i in range(num_branches):
                comp_factor = 80/len(clusters[i]) # Change number w/ dataset (80 is # classes for coco)
                heads.append(AdaConRetinaNetHead(backbone.out_channels, \
                        anchor_generator.num_anchors_per_location()[0], \
                        len(self.clusters[i]), comp_factor=comp_factor))
        self.heads = heads
        print(backbone.out_channels, "backbone.out_channels")
        self.branch_controller = AdaConRetinaNetBranchController(backbone.out_channels, len(self.clusters))

        if proposal_matcher is None:
            proposal_matcher = det_utils.Matcher(
                fg_iou_thresh,
                bg_iou_thresh,
                allow_low_quality_matches=True,
            )
        self.proposal_matcher = proposal_matcher

        self.box_coder = det_utils.BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))

        if image_mean is None:
            image_mean = [0.485, 0.456, 0.406]
        if image_std is None:
            image_std = [0.229, 0.224, 0.225]
        self.transform = GeneralizedRCNNTransform(min_size, max_size, image_mean, image_std)

        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.detections_per_img = detections_per_img
        self.topk_candidates = topk_candidates
        self.executed_branches = 0
        # used only on torchscript mode
        self._has_warned = False

    @torch.jit.unused
    def eager_outputs(self, losses, detections):
        # type: (Dict[str, Tensor], List[Dict[str, Tensor]]) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]
        if self.training:
            return losses

        return detections

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
                if l.get_device() < 0:
                    device = "cpu"
                else:
                    device = l.get_device()
                labels.append(torch.tensor(new_label, device=device))
            if len(labels) > 0:
                detections[i]['labels'] = torch.stack(labels, dim=0)
        return detections

    def _count_common_elements(self, list1, list2):
        return sum(el in list1 for el in list2)

    def _get_branch_controller_targets(self, targets):
        bc_targets: List[Tensor] = []
        for targets_per_image in targets:
            clusters_cnt = [self._count_common_elements(targets_per_image['labels'], self.clusters[i])
                            for i in range(self.num_branches)]
            bc_targets.append(torch.tensor(clusters_cnt.index(max(clusters_cnt)), 
                            device=targets_per_image['labels'].get_device()))

        return torch.stack(bc_targets, dim=0)

    def _eval_branch_controller(self, targets, outputs):
        targets = self._get_branch_controller_targets(targets)
        return {'accuracy': torch.count_nonzero(targets==outputs)/len(targets)}

    def compute_loss(self, targets, head_outputs, anchors):
        # type: (List[Dict[str, Tensor]], Dict[str, Tensor], List[Tensor]) -> Dict[str, Tensor]
        if self.enable_branch_controller:
            targets = self._get_branch_controller_targets(targets)
            return self.branch_controller.compute_loss(targets, head_outputs)
        else:
            matched_idxs = []
            for i, (anchors_per_image, targets_per_image) in enumerate(zip(anchors, targets)):
                targets_per_image = self._map_targets_to_cluster(targets_per_image)
                # change in src
                targets[i] = targets_per_image

                if targets_per_image == None:
                    matched_idxs.append(torch.full((anchors_per_image.size(0),), -1, dtype=torch.int64))
                    continue

                if targets_per_image['boxes'].numel() == 0:
                    matched_idxs.append(torch.full((anchors_per_image.size(0),), -1, dtype=torch.int64))
                    continue

                match_quality_matrix = box_ops.box_iou(targets_per_image['boxes'], anchors_per_image)
                matched_idxs.append(self.proposal_matcher(match_quality_matrix))

                return self.heads[self.active_branch].compute_loss(targets, head_outputs, anchors, matched_idxs)

    def postprocess_detections(self, head_outputs, anchors, image_shapes):
        # type: (Dict[str, List[Tensor]], List[List[Tensor]], List[Tuple[int, int]]) -> List[Dict[str, Tensor]]
        class_logits = head_outputs['cls_logits']
        box_regression = head_outputs['bbox_regression']

        num_images = len(image_shapes)

        detections: List[Dict[str, Tensor]] = []

        for index in range(num_images):
            box_regression_per_image = [br[index] for br in box_regression]
            logits_per_image = [cl[index] for cl in class_logits]
            anchors_per_image, image_shape = anchors[index], image_shapes[index]

            image_boxes = []
            image_scores = []
            image_labels = []

            for box_regression_per_level, logits_per_level, anchors_per_level in \
                    zip(box_regression_per_image, logits_per_image, anchors_per_image):
                num_classes = logits_per_level.shape[-1]

                # remove low scoring boxes
                scores_per_level = torch.sigmoid(logits_per_level).flatten()
                keep_idxs = scores_per_level > self.score_thresh
                scores_per_level = scores_per_level[keep_idxs]
                topk_idxs = torch.where(keep_idxs)[0]

                # keep only topk scoring predictions
                num_topk = min(self.topk_candidates, topk_idxs.size(0))
                scores_per_level, idxs = scores_per_level.topk(num_topk)
                topk_idxs = topk_idxs[idxs]

                anchor_idxs = topk_idxs // num_classes
                labels_per_level = topk_idxs % num_classes

                boxes_per_level = self.box_coder.decode_single(box_regression_per_level[anchor_idxs],
                                                               anchors_per_level[anchor_idxs])
                boxes_per_level = box_ops.clip_boxes_to_image(boxes_per_level, image_shape)

                image_boxes.append(boxes_per_level)
                image_scores.append(scores_per_level)
                image_labels.append(labels_per_level)

            image_boxes = torch.cat(image_boxes, dim=0)
            image_scores = torch.cat(image_scores, dim=0)
            image_labels = torch.cat(image_labels, dim=0)

            # non-maximum suppression
            keep = box_ops.batched_nms(image_boxes, image_scores, image_labels, self.nms_thresh)
            keep = keep[:self.detections_per_img]

            detections.append({
                'boxes': image_boxes[keep],
                'scores': image_scores[keep],
                'labels': image_labels[keep],
            })

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

        # get the original image sizes
        original_image_sizes: List[Tuple[int, int]] = []
        for img in images:
            val = img.shape[-2:]
            assert len(val) == 2
            original_image_sizes.append((val[0], val[1]))

        # transform the input
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

        # get the features from the backbone
        features = self.backbone(images.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([('0', features)])

        # TODO: Do we want a list or a dict?
        features = list(features.values())
        
        # create the set of anchors
        anchors = self.anchor_generator(images, features)

        losses = {}
        detections: List[Dict[str, Tensor]] = []
        if self.training:
            assert targets is not None
            # compute the losses
            # compute the retinanet heads outputs using the features
            if self.enable_branch_controller:
                outputs = self.branch_controller(features[-3])
                losses = self.compute_loss(targets, outputs, anchors)
            else:
                head_outputs = self.heads[self.active_branch](features)
                losses = self.compute_loss(targets, head_outputs, anchors)
        else:
            if self.enable_branch_controller:
                return self._eval_branch_controller(targets, torch.argmax(self.branch_controller(features[-3]), dim=1))
            elif self.multib:
                bc_out = self.branch_controller(features[-3].clone())
                _, active_branches = torch.nonzero(bc_out > self.bc_thres, as_tuple=True)
                if len(active_branches) == 0:
                    active_branches = [torch.argmax(bc_out, dim=1)]
                self.executed_branches += len(active_branches)
                # print(active_branches.item(), bc_out)
                combined_detections = []
                for i in range(self.num_branches):
                    if i in active_branches:
                        self.active_branch = i
                        # compute the retinanet heads outputs using the features
                        head_outputs = self.heads[self.active_branch](features)
                        # recover level sizes
                        num_anchors_per_level = [x.size(2) * x.size(3) for x in features]
                        HW = 0
                        for v in num_anchors_per_level:
                            HW += v
                        HWA = head_outputs['cls_logits'].size(1)
                        A = HWA // HW
                        num_anchors_per_level = [hw * A for hw in num_anchors_per_level]

                        # split outputs per level
                        split_head_outputs: Dict[str, List[Tensor]] = {}
                        for k in head_outputs:
                            split_head_outputs[k] = list(head_outputs[k].split(num_anchors_per_level, dim=1))
                        split_anchors = [list(a.split(num_anchors_per_level)) for a in anchors]

                        # compute the detections
                        detections = self.postprocess_detections(split_head_outputs, split_anchors, images.image_sizes)
                        detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)

                        detections = self._map_outputs_from_cluster(detections)
                        combined_detections.append(detections)
                    
                detections[0]['scores'] = torch.cat([detection[0]['scores'] for detection in combined_detections])
                detections[0]['labels'] = torch.cat([detection[0]['labels'] for detection in combined_detections])
                detections[0]['boxes'] = torch.cat([detection[0]['boxes'] for detection in combined_detections])
                # print(len(detections[0]['boxes']))
                keep = box_ops.batched_nms(detections[0]['boxes'], detections[0]['scores'], detections[0]['labels'], self.nms_thresh)
                # print("keep",len(keep))
                detections[0]['scores'] = detections[0]['scores'][keep]
                detections[0]['labels'] = detections[0]['labels'][keep]
                detections[0]['boxes'] = detections[0]['boxes'][keep]

            elif self.oracle:
                combined_detections = []
                for i in range(self.num_branches):
                    self.active_branch = i
                    # compute the retinanet heads outputs using the features
                    head_outputs = self.heads[self.active_branch](features)
                    # recover level sizes
                    num_anchors_per_level = [x.size(2) * x.size(3) for x in features]
                    HW = 0
                    for v in num_anchors_per_level:
                        HW += v
                    HWA = head_outputs['cls_logits'].size(1)
                    A = HWA // HW
                    num_anchors_per_level = [hw * A for hw in num_anchors_per_level]

                    # split outputs per level
                    split_head_outputs: Dict[str, List[Tensor]] = {}
                    for k in head_outputs:
                        split_head_outputs[k] = list(head_outputs[k].split(num_anchors_per_level, dim=1))
                    split_anchors = [list(a.split(num_anchors_per_level)) for a in anchors]

                    # compute the detections
                    detections = self.postprocess_detections(split_head_outputs, split_anchors, images.image_sizes)
                    detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)

                    detections = self._map_outputs_from_cluster(detections)
                    combined_detections.append(detections)
                    
                detections[0]['scores'] = torch.cat([detection[0]['scores'] for detection in combined_detections])
                detections[0]['labels'] = torch.cat([detection[0]['labels'] for detection in combined_detections])
                detections[0]['boxes'] = torch.cat([detection[0]['boxes'] for detection in combined_detections])
                keep = box_ops.batched_nms(detections[0]['boxes'], detections[0]['scores'], detections[0]['labels'], self.nms_thresh)
                # print("keep",len(keep))
                detections[0]['scores'] = detections[0]['scores'][keep]
                detections[0]['labels'] = detections[0]['labels'][keep]
                detections[0]['boxes'] = detections[0]['boxes'][keep]
            else:
                if self.singleb:
                    bc_out = self.branch_controller(features[-3].clone())
                    self.active_branch = torch.argmax(bc_out, dim=1)
                # compute the retinanet heads outputs using the features
                head_outputs = self.heads[self.active_branch](features)
                # recover level sizes
                num_anchors_per_level = [x.size(2) * x.size(3) for x in features]
                HW = 0
                for v in num_anchors_per_level:
                    HW += v
                HWA = head_outputs['cls_logits'].size(1)
                A = HWA // HW
                num_anchors_per_level = [hw * A for hw in num_anchors_per_level]

                # split outputs per level
                split_head_outputs: Dict[str, List[Tensor]] = {}
                for k in head_outputs:
                    split_head_outputs[k] = list(head_outputs[k].split(num_anchors_per_level, dim=1))
                split_anchors = [list(a.split(num_anchors_per_level)) for a in anchors]

                # compute the detections
                detections = self.postprocess_detections(split_head_outputs, split_anchors, images.image_sizes)
                detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)

                detections = self._map_outputs_from_cluster(detections)
        if torch.jit.is_scripting():
            if not self._has_warned:
                warnings.warn("RetinaNet always returns a (Losses, Detections) tuple in scripting")
                self._has_warned = True
            return losses, detections
        return self.eager_outputs(losses, detections)

class AdaConRetinaNetHead(nn.Module):
    """
    A regression and classification head for use in RetinaNet.
    Args:
        in_channels (int): number of channels of the input feature
        num_anchors (int): number of anchors to be predicted
        num_classes (int): number of classes to be predicted
    """

    def __init__(self, in_channels, num_anchors, num_classes, comp_factor):
        super().__init__()
        self.classification_head = AdaConRetinaNetClassificationHead(in_channels, num_anchors, num_classes, comp_factor=comp_factor)
        self.regression_head = AdaConRetinaNetRegressionHead(in_channels, num_anchors, comp_factor=comp_factor)

    def compute_loss(self, targets, head_outputs, anchors, matched_idxs):
        # type: (List[Dict[str, Tensor]], Dict[str, Tensor], List[Tensor], List[Tensor]) -> Dict[str, Tensor]
        return {
            'classification': self.classification_head.compute_loss(targets, head_outputs, matched_idxs),
            'bbox_regression': self.regression_head.compute_loss(targets, head_outputs, anchors, matched_idxs),
        }

    def forward(self, x):
        # type: (List[Tensor]) -> Dict[str, Tensor]
        return {
            'cls_logits': self.classification_head(x),
            'bbox_regression': self.regression_head(x)
        }

class AdaConRetinaNetBranchController(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        conv = []
        conv.append(nn.Conv2d(in_channels, in_channels//2, kernel_size=3, stride=2, padding=1))
        conv.append(nn.ReLU())
        conv.append(nn.Conv2d(in_channels//2, in_channels//2, kernel_size=3, stride=2, padding=1))
        conv.append(nn.ReLU())
        if num_classes < 5:
            conv.append(nn.Conv2d(in_channels//2, in_channels//4, kernel_size=3, stride=2, padding=1))
            conv.append(nn.ReLU())
            conv.append(nn.Conv2d(in_channels//4, in_channels//4, kernel_size=3, stride=2, padding=1))
            conv.append(nn.ReLU())
        else:
            conv.append(nn.Conv2d(in_channels//2, in_channels//2, kernel_size=3, stride=2, padding=1))
            conv.append(nn.ReLU())
            conv.append(nn.Conv2d(in_channels//2, in_channels//2, kernel_size=3, stride=2, padding=1))
            conv.append(nn.ReLU())

        self.conv = nn.Sequential(*conv)

        if num_classes < 5:
            self.fc1 = nn.Linear(64, 32)
        else:
            self.fc1 = nn.Linear(128, 32)
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
        if self.training:
            return F.softmax(x)
        else:
            return F.sigmoid(x)

class AdaConRetinaNetClassificationHead(nn.Module):
    """
    A classification head for use in RetinaNet.
    Args:
        in_channels (int): number of channels of the input feature
        num_anchors (int): number of anchors to be predicted
        num_classes (int): number of classes to be predicted
    """

    def __init__(self, in_channels, num_anchors, num_classes, comp_factor = 2, prior_probability=0.01):
        super().__init__()

        conv = []
        conv.append(nn.Conv2d(in_channels, int(in_channels/comp_factor), kernel_size=3, stride=1, padding=1))
        conv.append(nn.ReLU())
        for _ in range(3):
            conv.append(nn.Conv2d(int(in_channels/comp_factor), int(in_channels/comp_factor), kernel_size=3, stride=1, padding=1))
            conv.append(nn.ReLU())
        self.conv = nn.Sequential(*conv)

        for layer in self.conv.children():
            if isinstance(layer, nn.Conv2d):
                torch.nn.init.normal_(layer.weight, std=0.01)
                torch.nn.init.constant_(layer.bias, 0)

        self.cls_logits = nn.Conv2d(int(in_channels/comp_factor), num_anchors * num_classes, kernel_size=3, stride=1, padding=1)
        torch.nn.init.normal_(self.cls_logits.weight, std=0.01)
        torch.nn.init.constant_(self.cls_logits.bias, -math.log((1 - prior_probability) / prior_probability))

        self.num_classes = num_classes
        self.num_anchors = num_anchors

        # This is to fix using det_utils.Matcher.BETWEEN_THRESHOLDS in TorchScript.
        # TorchScript doesn't support class attributes.
        # https://github.com/pytorch/vision/pull/1697#issuecomment-630255584
        self.BETWEEN_THRESHOLDS = det_utils.Matcher.BETWEEN_THRESHOLDS

    def compute_loss(self, targets, head_outputs, matched_idxs):
        # type: (List[Dict[str, Tensor]], Dict[str, Tensor], List[Tensor]) -> Tensor
        losses = []

        cls_logits = head_outputs['cls_logits']

        for targets_per_image, cls_logits_per_image, matched_idxs_per_image in zip(targets, cls_logits, matched_idxs):
            if targets_per_image == None:
                continue
            # determine only the foreground
            foreground_idxs_per_image = matched_idxs_per_image >= 0
            num_foreground = foreground_idxs_per_image.sum()

            # create the target classification
            gt_classes_target = torch.zeros_like(cls_logits_per_image)
            gt_classes_target[
                foreground_idxs_per_image,
                targets_per_image['labels'][matched_idxs_per_image[foreground_idxs_per_image]]
            ] = 1.0
            # find indices for which anchors should be ignored
            valid_idxs_per_image = matched_idxs_per_image != self.BETWEEN_THRESHOLDS

            # compute the classification loss
            losses.append(sigmoid_focal_loss(
                cls_logits_per_image[valid_idxs_per_image],
                gt_classes_target[valid_idxs_per_image],
                reduction='sum',
            ) / max(1, num_foreground))

        return _sum(losses) / len(targets)

    def forward(self, x):
        # type: (List[Tensor]) -> Tensor
        all_cls_logits = []

        for features in x:
            cls_logits = self.conv(features)
            cls_logits = self.cls_logits(cls_logits)

            # Permute classification output from (N, A * K, H, W) to (N, HWA, K).
            N, _, H, W = cls_logits.shape
            cls_logits = cls_logits.view(N, -1, self.num_classes, H, W)
            cls_logits = cls_logits.permute(0, 3, 4, 1, 2)
            cls_logits = cls_logits.reshape(N, -1, self.num_classes)  # Size=(N, HWA, 4)

            all_cls_logits.append(cls_logits)

        return torch.cat(all_cls_logits, dim=1)

class AdaConRetinaNetRegressionHead(nn.Module):
    """
    A regression head for use in RetinaNet.
    Args:
        in_channels (int): number of channels of the input feature
        num_anchors (int): number of anchors to be predicted
    """
    __annotations__ = {
        'box_coder': det_utils.BoxCoder,
    }

    def __init__(self, in_channels, num_anchors, comp_factor=2):
        super().__init__()

        conv = []
        conv.append(nn.Conv2d(in_channels, int(in_channels/comp_factor), kernel_size=3, stride=1, padding=1))
        conv.append(nn.ReLU())
        for _ in range(3):
            conv.append(nn.Conv2d(int(in_channels/comp_factor), int(in_channels/comp_factor), kernel_size=3, stride=1, padding=1))
            conv.append(nn.ReLU())
        self.conv = nn.Sequential(*conv)

        self.bbox_reg = nn.Conv2d(int(in_channels/comp_factor), num_anchors * 4, kernel_size=3, stride=1, padding=1)
        torch.nn.init.normal_(self.bbox_reg.weight, std=0.01)
        torch.nn.init.zeros_(self.bbox_reg.bias)

        for layer in self.conv.children():
            if isinstance(layer, nn.Conv2d):
                torch.nn.init.normal_(layer.weight, std=0.01)
                torch.nn.init.zeros_(layer.bias)

        self.box_coder = det_utils.BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))

    def compute_loss(self, targets, head_outputs, anchors, matched_idxs):
        # type: (List[Dict[str, Tensor]], Dict[str, Tensor], List[Tensor], List[Tensor]) -> Tensor
        losses = []

        bbox_regression = head_outputs['bbox_regression']

        for targets_per_image, bbox_regression_per_image, anchors_per_image, matched_idxs_per_image in \
                zip(targets, bbox_regression, anchors, matched_idxs):
            if targets_per_image == None:
                continue
            # determine only the foreground indices, ignore the rest
            foreground_idxs_per_image = torch.where(matched_idxs_per_image >= 0)[0]
            num_foreground = foreground_idxs_per_image.numel()

            # select only the foreground boxes
            matched_gt_boxes_per_image = targets_per_image['boxes'][matched_idxs_per_image[foreground_idxs_per_image]]
            bbox_regression_per_image = bbox_regression_per_image[foreground_idxs_per_image, :]
            anchors_per_image = anchors_per_image[foreground_idxs_per_image, :]

            # compute the regression targets
            target_regression = self.box_coder.encode_single(matched_gt_boxes_per_image, anchors_per_image)

            # compute the loss
            losses.append(torch.nn.functional.l1_loss(
                bbox_regression_per_image,
                target_regression,
                reduction='sum'
            ) / max(1, num_foreground))

        return _sum(losses) / max(1, len(targets))

    def forward(self, x):
        # type: (List[Tensor]) -> Tensor
        all_bbox_regression = []

        for features in x:
            bbox_regression = self.conv(features)
            bbox_regression = self.bbox_reg(bbox_regression)

            # Permute bbox regression output from (N, 4 * A, H, W) to (N, HWA, 4).
            N, _, H, W = bbox_regression.shape
            bbox_regression = bbox_regression.view(N, -1, 4, H, W)
            bbox_regression = bbox_regression.permute(0, 3, 4, 1, 2)
            bbox_regression = bbox_regression.reshape(N, -1, 4)  # Size=(N, HWA, 4)

            all_bbox_regression.append(bbox_regression)

        return torch.cat(all_bbox_regression, dim=1)

