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
from adacon_model import retinanet_resnet50_fpn, adacon_retinanet_resnet50_fpn
from coco_utils import get_coco, get_coco_kp

from group_by_aspect_ratio import GroupedBatchSampler, create_aspect_ratio_groups
from engine import train_one_epoch, evaluate

import presets
import utils
from prettytable import PrettyTable

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
        # if not parameter.requires_grad: continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params+=param
    print(table)
    print("Total Trainable Params: ", total_params)
    return total_params

def freeze_all_non_active_layers(model, active_branch, train_bc):
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
    # utils.init_distributed_mode(args)
    args.distributed = False
    print(args)

    device = torch.device(args.device)

    # Data loading code
    print("Loading data")

    dataset, num_classes = get_dataset(args.dataset, "train", get_transform(train=True), args.data_path)
    dataset_test, _ = get_dataset(args.dataset, "val", get_transform(train=False), args.data_path)

    print("Creating data loaders")
    # if args.distributed:
    #     train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    #     test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test)
    # else:
    train_sampler = torch.utils.data.RandomSampler(dataset)
    test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    if args.aspect_ratio_group_factor >= 0:
        group_ids = create_aspect_ratio_groups(dataset, k=args.aspect_ratio_group_factor)
        train_batch_sampler = GroupedBatchSampler(train_sampler, group_ids, args.batch_size)
    else:
        train_batch_sampler = torch.utils.data.BatchSampler(
            train_sampler, args.batch_size, drop_last=True)

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_sampler=train_batch_sampler, num_workers=args.workers,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=args.batch_size,
        sampler=test_sampler, num_workers=args.workers,
        collate_fn=utils.collate_fn)

    print("Creating model")
    kwargs = {
        "trainable_backbone_layers": args.trainable_backbone_layers
    }
    if "rcnn" in args.model:
        if args.rpn_score_thresh is not None:
            kwargs["rpn_score_thresh"] = args.rpn_score_thresh
    
    min_size, max_size = get_image_size_range(args.img_size)
    if args.adaptive:
        clusters = parse_clusters_config(args.clusters)
        active_branch = args.active_branch
        model = adacon_retinanet_resnet50_fpn(clusters=clusters, active_branch=active_branch, num_branches=len(clusters),
                            num_classes=num_classes, trainable_backbone_layers=args.trainable_backbone_layers,
                            pretrained=args.pretrained, pretrained_backbone=args.pretrained_backbone, branches_weights=args.branches,
                            backbone_weights=args.backbone_weights, bc_weights=args.branch_controller,
                            min_size=min_size, max_size=max_size, ckpt=args.resume)
        freeze_all_non_active_layers(model, active_branch, args.enable_branch_controller)

        if args.enable_branch_controller:
            model.enable_branch_controller = True
    else:
        model = retinanet_resnet50_fpn(num_classes=num_classes, trainable_backbone_layers=args.trainable_backbone_layers,
                        pretrained=args.pretrained, pretrained_backbone=args.pretrained_backbone, 
                        backbone_weights="retinanet_coco_backbone.pt", head_weights="retinanet_coco_head.pt",
                        min_size=min_size, max_size=max_size, ckpt=args.resume)
    
    count_parameters(model)
    model.to(device)

    model_without_ddp = model
    # if args.distributed:
    #     model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
    #     model_without_ddp = model.module

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_steps, gamma=args.lr_gamma)

    if args.resume:
        if args.adaptive:
            checkpoint = torch.load(args.resume, map_location='cpu')
            model_without_ddp.backbone.load_state_dict(torch.load(args.resume)['backbone'])
            model_without_ddp.heads[0].load_state_dict(torch.load(args.resume)['head'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
            model_without_ddp.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1

    # utils.save_on_master({
    #     'model': model_without_ddp.state_dict(),
    #     'backbone': model_without_ddp.backbone.state_dict(),
    #     'head': model_without_ddp.heads[active_branch].state_dict(),
    #     'optimizer': optimizer.state_dict(),
    #     'lr_scheduler': lr_scheduler.state_dict(),
    #     'args': args},
    #     os.path.join(args.output_dir, 'retinanet_coco.pth'))
    # exit()

    if args.test_only:
        if args.oracle:
            model.oracle = True
        if args.single:
            model.singleb = True
        evaluate(model, data_loader_test, device=device)
        return

    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        # if args.distributed:
        #     train_sampler.set_epoch(epoch)
        train_one_epoch(model, optimizer, data_loader, device, epoch, args.print_freq)
        lr_scheduler.step()
        if args.output_dir:
            if args.adaptive:
                if args.enable_branch_controller:
                    utils.save_on_master({
                    'branch_controller': model_without_ddp.branch_controller.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'args': args,
                    'epoch': epoch},
                    os.path.join(args.output_dir, 'model_branchcontroller{}_{}.pth'.format(len(clusters), epoch)))
                else:
                    utils.save_on_master({
                        'model': model_without_ddp.state_dict(),
                        'backbone': model_without_ddp.backbone.state_dict(),
                        'head': model_without_ddp.heads[active_branch].state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'args': args,
                        'epoch': epoch},
                        os.path.join(args.output_dir, 'model_branch{}_{}.pth'.format(active_branch, epoch)))
            else:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'args': args,
                    'epoch': epoch},
                    os.path.join(args.output_dir, 'model_{}.pth'.format(epoch)))

        # evaluate after every epoch
        evaluate(model, data_loader_test, device=device)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


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
    parser.add_argument('--epochs', default=26, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--lr', default=0.02, type=float,
                        help='initial learning rate, 0.02 is the default value for training '
                        'on 8 gpus and 2 images_per_gpu')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--lr-step-size', default=8, type=int, help='decrease lr every step-size epochs')
    parser.add_argument('--lr-steps', default=[16, 22], nargs='+', type=int, help='decrease lr every step-size epochs')
    parser.add_argument('--lr-gamma', default=0.1, type=float, help='decrease lr by a factor of lr-gamma')
    parser.add_argument('--print-freq', default=20, type=int, help='print frequency')
    parser.add_argument('--output-dir', default='.', help='path where to save')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--backbone-weights', dest="backbone_weights", default='backbone_coco.pth', help='load backbone weights')
    parser.add_argument('--start_epoch', default=0, type=int, help='start epoch')
    parser.add_argument('--aspect-ratio-group-factor', default=3, type=int)
    parser.add_argument('--rpn-score-thresh', default=None, type=float, help='rpn score threshold for faster-rcnn')
    parser.add_argument('--trainable-backbone-layers', default=None, type=int,
                        help='number of trainable layers of backbone')
    parser.add_argument(
        "--test-only",
        dest="test_only",
        help="Only test the model",
        action="store_true",
    )
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
        "--clusters",
        dest="clusters", type=str,
        help="Clusters file to create the adaptive model"
    )
    parser.add_argument(
        "--img-size",
        dest="img_size", type=int, default=416,
        help="Input image size to model"
    )
    parser.add_argument('--active-branch', dest="active_branch", default=0, type=int,
                        help='active branch in the adaptive model')

    parser.add_argument('--branches', nargs='+', help='trained branches for adaptive test', required=False)
    parser.add_argument('--branch_controller', type=str, help='trained branch controller for adaptive test', required=False)

    parser.add_argument('--enable-branch-controller', dest="enable_branch_controller", action="store_true",
                        help='Train Branch Controller')
    # distributed training parameters
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')

    args = parser.parse_args()

    if args.output_dir:
        utils.mkdir(args.output_dir)

    main(args)
