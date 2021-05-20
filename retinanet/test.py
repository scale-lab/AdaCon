####################################################################################################################
################ THIS CODE IS ADAPTED FROM CODE AT https://github.com/pytorch/vision/tree/master/references/detection
####################################################################################################################

import datetime
import os
import time

import torch
import torch.utils.data
import torchvision
import torchvision.models.detection
import torchvision.models.detection.mask_rcnn
from adacon_fasterrcnn import adacon_fasterrcnn_mobilenet_v3_large_320_fpn, fasterrcnn_mobilenet_v3_large_320_fpn
from adacon_model import retinanet_resnet50_fpn, adacon_retinanet_resnet50_fpn
from coco_utils import get_coco, get_coco_kp

from group_by_aspect_ratio import GroupedBatchSampler, create_aspect_ratio_groups
from engine import run_on_device, evaluate

import presets
import utils
from prettytable import PrettyTable
from profiler import Profiler

def get_image_size_range(img_size):
    if img_size == 416:
        return 400, 500
    elif img_size == 320:
        return 300, 400
    elif img_size == 512:
        return 500, 600
    else:
        return img_size, img_size
    
def parse_clusters_config(path):
    """Parses the clusters configuration file"""
    print("Reading clusters file")

    clusters = []
    with open(path, 'r') as f:
        for line in f:
            cs = line.split(",")
            clusters.append([int(c) for c in cs])

    for i, clus in enumerate(clusters):
        print("Cluster ", i, "has ", len(clus), " classes")
    
    coco80to91 = coco80_to_coco91_class()
    clusters = [[coco80to91[c] for c in cluster] for cluster in clusters]
    return clusters

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params+=param
    print(table)
    print("Total Trainable Params: ", total_params)
    return total_params

def freeze_adacon_retinanet_all_non_active_layers(model, active_branch, train_bc):
    for param in model.backbone.parameters():
        param.requires_grad = False
    
    if train_bc:
        for i, head in enumerate(model.heads):
            for param in head.parameters():
                param.requires_grad = False
    else:
        for param in model.branch_controller.parameters():
            param.requires_grad = False
        for i, head in enumerate(model.heads):
            if i == active_branch:
                continue
            for param in head.parameters():
                param.requires_grad = False
    
def freeze_adacon_rcnn_all_non_active_layers(model, active_branch, train_bc):
    for param in model.backbone.parameters():
        param.requires_grad = False

    for param in model.rpn.parameters():
        param.requires_grad = False    
    if train_bc:
        for i, head in enumerate(model.roi_heads):
            for param in head.parameters():
                param.requires_grad = False
    else:
        # for param in model.branch_controller.parameters():
        #     param.requires_grad = False
        for i, head in enumerate(model.roi_heads):
            if i == active_branch:
                continue
            print("Freezing head", i)
            for param in head.parameters():
                param.requires_grad = False

def coco80_to_coco91_class():  # converts 80-index (val2014) to 91-index (paper)
    # https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34,
         35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
         64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]
    return x

def get_dataset(name, image_set, transform, data_path):
    paths = {
        "coco": (data_path, get_coco, 91),
        "coco_kp": (data_path, get_coco_kp, 2)
    }
    p, ds_fn, num_classes = paths[name]

    ds = ds_fn(p, image_set=image_set, transforms=transform)
    return ds, num_classes


def get_transform(train):
    return presets.DetectionPresetTrain() if train else presets.DetectionPresetEval()


def main(args):
    args.distributed = False

    device = torch.device(args.device)

    # Data loading code
    print("Loading data")

    dataset_test, num_classes = get_dataset(args.dataset, "val", get_transform(train=False), args.data_path)

    print("Creating data loaders")
    test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=args.batch_size,
        sampler=test_sampler, num_workers=0,
        collate_fn=utils.collate_fn)

    print("Creating model")
    
    min_size, max_size = get_image_size_range(args.img_size)
    if args.adaptive:
        clusters = parse_clusters_config(args.clusters)
        active_branch = args.active_branch
        if args.model == "retinanet":
            model = adacon_retinanet_resnet50_fpn(clusters=clusters, active_branch=active_branch, num_branches=len(clusters),
                                num_classes=num_classes, trainable_backbone_layers=0,
                                pretrained=args.pretrained, pretrained_backbone=args.pretrained_backbone, branches_weights=args.branches,
                                backbone_weights=args.backbone_weights, bc_weights=args.branch_controller,
                                min_size=min_size, max_size=max_size, deploy=args.deploy)
            freeze_adacon_retinanet_all_non_active_layers(model, active_branch, False)

        elif args.model == "rcnn":
            model = adacon_fasterrcnn_mobilenet_v3_large_320_fpn(clusters=clusters, active_branch=active_branch, num_branches=len(clusters),
                                num_classes=num_classes, trainable_backbone_layers=0,
                                pretrained=args.pretrained, pretrained_backbone=args.pretrained_backbone, branches_weights=args.branches,
                                backbone_weights=args.backbone_weights, bc_weights=args.branch_controller,
                                min_size=min_size, max_size=max_size)
            freeze_adacon_rcnn_all_non_active_layers(model, active_branch, False)

        if args.oracle:
            model.oracle = True
        if args.single:
            model.singleb = True
        if args.multi:
            model.multib = True
        model.bc_thres = args.bc_thres
        count_parameters(model)
        model.to(device)
    else:
        if args.model == "retinanet":
            model = retinanet_resnet50_fpn(num_classes=num_classes, trainable_backbone_layers=0,
                            pretrained=args.pretrained, pretrained_backbone=args.pretrained_backbone, 
                            backbone_weights="retinanet_coco_backbone.pt", head_weights="retinanet_coco_head.pt",
                            min_size=min_size, max_size=max_size)
        
        elif args.model == "rcnn":
            model = fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=args.pretrained, min_size=min_size, max_size=max_size)
            count_parameters(model)
            count_parameters(model.backbone)
            count_parameters(model.rpn)
            count_parameters(model.roi_heads)
        model.to(device)

    if args.profile:
        run_on_device(model, data_loader_test, device=device)
        return
    else:
        profiler = Profiler()
        if args.adaptive:
            params = profiler.profile_params(model, len(clusters))
        else:
            params = profiler.profile_params(model, 1)

        input = torch.randn(1, 3, args.img_size, args.img_size)
        input = input.to(device)
        if args.adaptive:
            macs = profiler.profile_macs(model, input, len(clusters))
        else:
            macs = profiler.profile_macs(model, input, 1)

        accuracy_stats = evaluate(model, data_loader_test, device=device)

        # Img Size, # Branches, Mode (single, multi, baseline), branches/inf, map(50:95), 
        # map(50), map(75), map(s), map(m), map(l), Total Param, Backbone param, branch param,
        # Dynamic param, Total Macs, Backbone Macs, branch Macs, Dynamic Macs
        
        results = []
        if not args.adaptive:
            mode = "baseline"
            results.extend([args.img_size, 1, mode, 1])
        else:
            if args.single:
                mode = "single"
                branches_per_image = 1
            elif args.multi:
                mode = "multi" + str(args.bc_thres)
                branches_per_image = model.executed_branches/len(data_loader_test)
            else:
                mode = "oracle"
                branches_per_image = model.executed_branches/len(data_loader_test)

            results.extend([args.img_size, len(clusters), mode, branches_per_image])
        
        results.extend(accuracy_stats[0][0:6])
        results.extend(params)
        results.append(params[1]+params[2])
        results.extend(macs)
        results.append(macs[1]+macs[2]*branches_per_image)
        print("Results ", results)

    return

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description=__doc__)

    parser.add_argument('--data-path', default='/datasets01/COCO/022719/', help='dataset')
    parser.add_argument('--dataset', default='coco', help='dataset')
    parser.add_argument('--model', default='maskrcnn_resnet50_fpn', help='model')
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('-b', '--batch-size', default=2, type=int,
                        help='images per gpu, the total batch size is $NGPU x batch_size')
    parser.add_argument('--backbone-weights', dest="backbone_weights", type=str, default='backbone_coco.pth', help='load backbone weights')
    parser.add_argument('--deploy', dest="deploy", type=str, help='combined weights for deployment')
    parser.add_argument('--trainable-backbone-layers', default=None, type=int,
                        help='number of trainable layers of backbone')
    parser.add_argument('--bc-thres', dest="bc_thres", default=0.4, type=float,
                        help='branch controller threshold')
    parser.add_argument(
        "--pretrained",
        dest="pretrained",
        help="Use pre-trained models from the modelzoo",
        action="store_true",
    )
    parser.add_argument(
        "--pretrained-backbone",
        dest="pretrained_backbone",
        help="Use pre-trained backbone models from the modelzoo",
        action="store_true",
    )
    parser.add_argument(
        "--adaptive",
        dest="adaptive",
        help="Enable Adaptive mode",
        action="store_true",
    )
    parser.add_argument(
        "--oracle",
        dest="oracle",
        help="Enable oracle Adaptive mode",
        action="store_true",
    )
    parser.add_argument(
        "--single",
        dest="single",
        help="Enable single execution Adaptive mode",
        action="store_true",
    )
    parser.add_argument(
        "--multi",
        dest="multi",
        help="Enable multi execution Adaptive mode",
        action="store_true",
    )
    parser.add_argument(
        "--clusters",
        dest="clusters", type=str,
        help="Clusters file to create the adaptive model"
    )
    parser.add_argument(
        "--img-size",
        dest="img_size", type=int, default=416,
        help="Input image size to model"
    )
    parser.add_argument(
        "--profile",
        dest="profile",
        help="Enable profiling Adaptive mode",
        action="store_true",
    )
    parser.add_argument('--active-branch', dest="active_branch", default=0, type=int,
                        help='active branch in the adaptive model')

    parser.add_argument('--branches', nargs='+', help='trained branches for adaptive test', required=False)
    parser.add_argument('--branch_controller', type=str, help='trained branch controller for adaptive test', required=False)

    args = parser.parse_args()

    main(args)
