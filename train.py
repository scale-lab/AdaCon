import argparse

import torch.distributed as dist
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter

from test import test  # import test.py to get mAP after each epoch
from models import *
from utils.datasets import *
from utils.utils import *
from torch.autograd import Variable

mixed_precision = True
try:  # Mixed precision training https://github.com/NVIDIA/apex
    from apex import amp
except:
    print('Apex recommended for faster mixed precision training: https://github.com/NVIDIA/apex')
    mixed_precision = False  # not installed

wdir = 'weights' + os.sep  # weights dir
last = wdir
best = wdir
results_file = ""

# Hyperparameters
hyp = {'giou': 3.54,  # giou loss gain
       'cls': 37.4,  # cls loss gain
       'cls_pw': 1.0,  # cls BCELoss positive_weight
       'obj': 64.3,  # obj loss gain (*=img_size/320 if img_size != 320)
       'obj_pw': 1.0,  # obj BCELoss positive_weight
       'iou_t': 0.20,  # iou training threshold
       'lr0': 0.01,  # initial learning rate (SGD=5E-3, Adam=5E-4)
       'lrf': 0.0005,  # final learning rate (with cos scheduler)
       'momentum': 0.937,  # SGD momentum
       'weight_decay': 0.0005,  # optimizer weight decay
       'fl_gamma': 0.0,  # focal loss gamma (efficientDet default is gamma=1.5)
       'hsv_h': 0.0138,  # image HSV-Hue augmentation (fraction)
       'hsv_s': 0.678,  # image HSV-Saturation augmentation (fraction)
       'hsv_v': 0.36,  # image HSV-Value augmentation (fraction)
       'degrees': 1.98 * 0,  # image rotation (+/- deg)
       'translate': 0.05 * 0,  # image translation (+/- fraction)
       'scale': 0.05 * 0,  # image scale (+/- gain)
       'shear': 0.641 * 0}  # image shear (+/- deg)

# Overwrite hyp with hyp*.txt (optional)
f = glob.glob('hyp*.txt')
if f:
    print('Using %s' % f[0])
    for k, v in zip(hyp.keys(), np.loadtxt(f[0])):
        hyp[k] = v

# Print focal loss if gamma > 0
if hyp['fl_gamma']:
    print('Using FocalLoss(gamma=%g)' % hyp['fl_gamma'])

def train_branch_controller():
    # Get data configuration
    init_seeds()
    data_dict = parse_data_cfg(opt.data)
    train_path = data_dict['train']
    test_path = data_dict['valid']
    nc = 1 if opt.single_cls else int(data_dict['classes'])  # number of classes
    clusters = parse_clusters_config(opt.clusters)
    class_to_cluster_list = get_class_to_cluster_map(clusters)

    # Create and load the backbone
    backbone = Backbone(opt.backbone_cfg).to(device)
    backbone.load_darknet_weights(opt.backbone_weights,75)
    backbone.eval()

    # Create the  branch controller
    branch_controller = BranchController(opt.branch_controller_cfg, len(clusters)).to(device)
    count_parameters(branch_controller)
    if opt.branch_controller_weights:
        branch_controller.load_state_dict(torch.load(opt.branch_controller_weights))

    # Dataset
    dataset = ClustersDataset(train_path, augment=True, multiscale=opt.multi_scale, clusters=clusters)
    dataset_valid = ClustersDataset(test_path, augment=True, multiscale=False, clusters=clusters)

    # Dataloader
    batch_size = min(opt.batch_size, len(dataset))
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             num_workers=nw,
                                             shuffle=not opt.rect,  # Shuffle=True unless rectangular training is used
                                             pin_memory=True,
                                             collate_fn=dataset.collate_fn)



    dataloader_valid = torch.utils.data.DataLoader(dataset_valid,
                                             batch_size=batch_size,
                                             num_workers=nw,
                                             shuffle=True,  # Shuffle=True unless rectangular training is used
                                             pin_memory=True,
                                             collate_fn=dataset.collate_fn)

    criterion = nn.MSELoss()
    learning_rate = 0.001
    momentum = 0.9

    best_accuracy = 0
    best_model = branch_controller

    # Optimizer 
    # optimizer = torch.optim.SGD(branch_controller.parameters(),lr=learning_rate, momentum=momentum)
    optimizer = torch.optim.Adam(branch_controller.parameters(), lr=0.0001)

    epochs = 10
    for epoch in range(epochs):
        start_time = time.time()
        pbar = tqdm(enumerate(dataloader), total=len(dataloader))
        for batch_i, (_, imgs, targets) in pbar: # TODO convert to dataloader
            branch_controller.train()
            imgs = imgs.to(device)
            targets = targets.to(device)
            backbone_out = backbone(imgs)
            backbone.layer_outputs = []            
            output = branch_controller(backbone_out)
            loss = criterion(output, targets)
            loss.backward()
            
            optimizer.step()    # Does the update
            optimizer.zero_grad()
            
            branch_controller.seen += imgs.size(0)
            if batch_i % 100 == 0:
                print(batch_i,len(dataloader), loss.cpu().detach().numpy())
 
        print("Evaluate on ", len(dataloader_valid))
        right_predictions = 0
        all_predictions = 0
        mse = 0
        for batch_i, (_, imgs, targets) in enumerate(dataloader_valid):
            branch_controller.eval()
            imgs = Variable(imgs.to(device))
            targets = Variable(targets.to(device), requires_grad=False)

            backbone_out = backbone(imgs)
            backbone.layer_outputs = []            
            output = branch_controller(backbone_out)

            output = torch.argmax(output.cpu(), dim=1).numpy()
            targets = torch.argmax(targets.cpu(), dim=1).numpy()
            # print(output, targets)
            right_predictions += np.count_nonzero(targets==output)
            all_predictions += len(targets)

            mse += np.mean(np.square(targets-output))
            
        print("Accuracy of Validation = ", 100.0*right_predictions/all_predictions)
        if best_accuracy < right_predictions/all_predictions:
            best_accuracy = right_predictions/all_predictions
            best_model = branch_controller

            print("Saving best model with accuracy", best_accuracy)
            bc_filename = wdir + "bc" + opt.name +".pt"
            torch.save(best_model.state_dict(), bc_filename)


def train(hyp):
    cfg = opt.cfg
    data = opt.data
    epochs = opt.epochs  # 500200 batches at bs 64, 117263 images = 273 epochs
    batch_size = opt.batch_size
    accumulate = max(round(64 / batch_size), 1)  # accumulate n times before optimizer update (bs 64)
    weights = opt.weights  # initial training weights
    imgsz_min, imgsz_max, imgsz_test = opt.img_size  # img sizes (min, max, test)

    clusters = None
    class_to_cluster_list = None
    if opt.adaptive:
        clusters = parse_clusters_config(opt.clusters)
        class_to_cluster_list = get_class_to_cluster_map(clusters)

    # Image Sizes
    gs = 32  # (pixels) grid size
    assert math.fmod(imgsz_min, gs) == 0, '--img-size %g must be a %g-multiple' % (imgsz_min, gs)
    opt.multi_scale |= imgsz_min != imgsz_max  # multi if different (min, max)
    if opt.multi_scale:
        if imgsz_min == imgsz_max:
            imgsz_min //= 1.5
            imgsz_max //= 0.667
        grid_min, grid_max = imgsz_min // gs, imgsz_max // gs
        imgsz_min, imgsz_max = int(grid_min * gs), int(grid_max * gs)
    img_size = imgsz_max  # initialize with max size

    # Configure run
    init_seeds()
    data_dict = parse_data_cfg(data)
    train_path = data_dict['train']
    test_path = data_dict['valid']
    nc = 1 if opt.single_cls else int(data_dict['classes'])  # number of classes
    hyp['cls'] *= nc / 80  # update coco-tuned hyp['cls'] to current dataset

    # Remove previous results
    for f in glob.glob('*_batch*.jpg') + glob.glob(results_file):
        os.remove(f)

    # Initialize model
    model = Darknet(cfg).to(device)

    backbone = None
    if opt.adaptive:
        backbone = Backbone(opt.backbone_cfg).to(device)
        backbone.load_darknet_weights(opt.backbone_weights,75)

    # Optimizer
    pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
    for k, v in dict(model.named_parameters()).items():
        if '.bias' in k:
            pg2 += [v]  # biases
        elif 'Conv2d.weight' in k:
            pg1 += [v]  # apply weight_decay
        else:
            pg0 += [v]  # all else

    if opt.adam:
        # hyp['lr0'] *= 0.1  # reduce lr (i.e. SGD=5E-3, Adam=5E-4)
        optimizer = optim.Adam(pg0, lr=hyp['lr0'])
        # optimizer = AdaBound(pg0, lr=hyp['lr0'], final_lr=0.1)
    else:
        optimizer = optim.SGD(pg0, lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)
    optimizer.add_param_group({'params': pg1, 'weight_decay': hyp['weight_decay']})  # add pg1 with weight_decay
    optimizer.add_param_group({'params': pg2})  # add pg2 (biases)
    print('Optimizer groups: %g .bias, %g Conv2d.weight, %g other' % (len(pg2), len(pg1), len(pg0)))
    del pg0, pg1, pg2

    start_epoch = 0
    best_fitness = 0.0
    # attempt_download(weights)
    if weights and weights.endswith('.pt'):  # pytorch format
        # possible weights are '*.pt', 'yolov3-spp.pt', 'yolov3-tiny.pt' etc.
        ckpt = torch.load(weights, map_location=device)

        # load model
        try:
            ckpt['model'] = {k: v for k, v in ckpt['model'].items() if model.state_dict()[k].numel() == v.numel()}
            model.load_state_dict(ckpt['model'], strict=False)
        except KeyError as e:
            s = "%s is not compatible with %s. Specify --weights '' or specify a --cfg compatible with %s. " \
                "See https://github.com/ultralytics/yolov3/issues/657" % (opt.weights, opt.cfg, opt.weights)
            raise KeyError(s) from e

        # load optimizer
        if ckpt['optimizer'] is not None:
            optimizer.load_state_dict(ckpt['optimizer'])
            best_fitness = ckpt['best_fitness']

        # load results
        if ckpt.get('training_results') is not None:
            with open(results_file, 'w') as file:
                file.write(ckpt['training_results'])  # write results.txt

        # epochs
        start_epoch = ckpt['epoch'] + 1
        if epochs < start_epoch:
            print('%s has been trained for %g epochs. Fine-tuning for %g additional epochs.' %
                  (opt.weights, ckpt['epoch'], epochs))
            epochs += ckpt['epoch']  # finetune additional epochs

        del ckpt

    elif weights and len(weights) > 0:  # darknet format
        # possible weights are '*.weights', 'yolov3-tiny.conv.15',  'darknet53.conv.74' etc.
        print(weights)
        load_darknet_weights(model, weights)

    if opt.freeze_layers:
        output_layer_indices = [idx - 1 for idx, module in enumerate(model.module_list) if isinstance(module, YOLOLayer)]
        freeze_layer_indices = [x for x in range(len(model.module_list)) if
                                (x not in output_layer_indices) and
                                (x - 1 not in output_layer_indices)]
        for idx in freeze_layer_indices:
            for parameter in model.module_list[idx].parameters():
                parameter.requires_grad_(False)

    # Mixed precision training https://github.com/NVIDIA/apex
    if mixed_precision:
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1', verbosity=0)

    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    lf = lambda x: (((1 + math.cos(x * math.pi / epochs)) / 2) ** 1.0) * 0.95 + 0.05  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    scheduler.last_epoch = start_epoch - 1  # see link below
    # https://discuss.pytorch.org/t/a-problem-occured-when-resuming-an-optimizer/28822

    # Initialize distributed training
    if device.type != 'cpu' and torch.cuda.device_count() > 1 and torch.distributed.is_available():
        dist.init_process_group(backend='nccl',  # 'distributed backend'
                                init_method='tcp://127.0.0.1:9999',  # distributed training init method
                                world_size=1,  # number of nodes for distributed training
                                rank=0)  # distributed training node rank
        model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
        model.yolo_layers = model.module.yolo_layers  # move yolo layer indices to top level

    # Dataset
    dataset = LoadImagesAndLabels(train_path, img_size, batch_size,
                                  augment=True,
                                  hyp=hyp,  # augmentation hyperparameters
                                  rect=opt.rect,  # rectangular training
                                  cache_images=opt.cache_images,
                                  single_cls=opt.single_cls)

    # Dataloader
    batch_size = min(batch_size, len(dataset))
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             num_workers=nw,
                                             shuffle=not opt.rect,  # Shuffle=True unless rectangular training is used
                                             pin_memory=True,
                                             collate_fn=dataset.collate_fn)

    # Testloader
    testloader = torch.utils.data.DataLoader(LoadImagesAndLabels(test_path, imgsz_test, batch_size,
                                                                 hyp=hyp,
                                                                 rect=True,
                                                                 cache_images=opt.cache_images,
                                                                 single_cls=opt.single_cls),
                                             batch_size=batch_size,
                                             num_workers=nw,
                                             pin_memory=True,
                                             collate_fn=dataset.collate_fn)

    # Model parameters
    model.nc = nc  # attach number of classes to model
    model.hyp = hyp  # attach hyperparameters to model
    model.gr = 1.0  # giou loss ratio (obj_loss = 1.0 or giou)
    model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device)  # attach class weights

    # Model EMA
    # ema = torch_utils.ModelEMA(model)

    if opt.prune:
        model.prune()
        # ema.update(model)
        is_coco = any([x in data for x in ['coco.data', 'coco2014.data', 'coco2017.data']]) and model.nc == 80
        results, maps = test(cfg,
                            data,
                            batch_size=batch_size,
                            imgsz=imgsz_test,
                            model=model,
                            save_json=is_coco,
                            single_cls=opt.single_cls,
                            dataloader=testloader,
                            multi_label=False, 
                            clusters=clusters,
                            class_to_cluster_list=class_to_cluster_list,
                            cluster_idx=opt.cluster_idx,
                            backbone=backbone)

    # Start training
    nb = len(dataloader)  # number of batches
    n_burn = max(3 * nb, 500)  # burn-in iterations, max(3 epochs, 500 iterations)
    maps = np.zeros(nc)  # mAP per class
    # torch.autograd.set_detect_anomaly(True)
    results = (0, 0, 0, 0, 0, 0, 0)  # 'P', 'R', 'mAP', 'F1', 'val GIoU', 'val Objectness', 'val Classification'
    t0 = time.time()
    print('Image sizes %g - %g train, %g test' % (imgsz_min, imgsz_max, imgsz_test))
    print('Using %g dataloader workers' % nw)
    print('Starting training for %g epochs...' % epochs)
    for epoch in range(start_epoch, epochs):  # epoch ------------------------------------------------------------------
        model.train()

        # Update image weights (optional)
        if dataset.image_weights:
            w = model.class_weights.cpu().numpy() * (1 - maps) ** 2  # class weights
            image_weights = labels_to_image_weights(dataset.labels, nc=nc, class_weights=w)
            dataset.indices = random.choices(range(dataset.n), weights=image_weights, k=dataset.n)  # rand weighted idx

        mloss = torch.zeros(4).to(device)  # mean losses
        print(('\n' + '%10s' * 8) % ('Epoch', 'gpu_mem', 'GIoU', 'obj', 'cls', 'total', 'targets', 'img_size'))
        pbar = tqdm(enumerate(dataloader), total=nb)  # progress bar
        for i, (imgs, targets, paths, _) in pbar:  # batch -------------------------------------------------------------
            ni = i + nb * epoch  # number integrated batches (since train start)
            imgs = imgs.to(device).float() / 255.0  # uint8 to float32, 0 - 255 to 0.0 - 1.0
            targets = targets.to(device)

            # Burn-in
            if ni <= n_burn:
                xi = [0, n_burn]  # x interp
                model.gr = np.interp(ni, xi, [0.0, 1.0])  # giou loss ratio (obj_loss = 1.0 or giou)
                accumulate = max(1, np.interp(ni, xi, [1, 64 / batch_size]).round())
                for j, x in enumerate(optimizer.param_groups):
                    # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                    x['lr'] = np.interp(ni, xi, [0.1 if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
                    x['weight_decay'] = np.interp(ni, xi, [0.0, hyp['weight_decay'] if j == 1 else 0.0])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(ni, xi, [0.9, hyp['momentum']])

            # Multi-Scale
            if opt.multi_scale:
                if ni / accumulate % 1 == 0:  #  adjust img_size (67% - 150%) every 1 batch
                    img_size = random.randrange(grid_min, grid_max + 1) * gs
                sf = img_size / max(imgs.shape[2:])  # scale factor
                if sf != 1:
                    ns = [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]]  # new shape (stretched to 32-multiple)
                    imgs = F.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)

            # Forward
            if backbone is None:
                pred = model(imgs)
            else:
                back_out = backbone(imgs)
                pred = model(back_out, out=backbone.layer_outputs)
                backbone.layer_outputs = []
                # In training, we convert targets labels to predictions labels to compute the loss for this branch
                # TO DO: Think about the otherway around, would that overwhelm the branch???
                targets = map_labels_to_cluster(targets, clusters, class_to_cluster_list, opt.cluster_idx, device)
                if targets.shape[0] == 0:
                    continue
            
            # Loss
            loss, loss_items = compute_loss(pred, targets, model)
            if not torch.isfinite(loss):
                print('WARNING: non-finite loss, ending training ', loss_items)
                return results

            # Backward
            loss *= batch_size / 64  # scale loss
            if mixed_precision:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            # Optimize
            if ni % accumulate == 0:
                optimizer.step()
                optimizer.zero_grad()
                # ema.update(model)

            # Print
            mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
            mem = '%.3gG' % (torch.cuda.memory_cached() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
            s = ('%10s' * 2 + '%10.3g' * 6) % ('%g/%g' % (epoch, epochs - 1), mem, *mloss, len(targets), img_size)
            pbar.set_description(s)

            # end batch ------------------------------------------------------------------------------------------------

        # Update scheduler
        scheduler.step()

        # Process epoch results
        # ema.update_attr(model)
        final_epoch = epoch + 1 == epochs
        if not opt.notest or final_epoch:  # Calculate mAP
            is_coco = any([x in data for x in ['coco.data', 'coco2014.data', 'coco2017.data']]) and model.nc == 80
            results, maps = test(cfg,
                                 data,
                                 batch_size=batch_size,
                                 imgsz=imgsz_test,
                                 model=model,
                                 save_json=final_epoch and is_coco,
                                 single_cls=opt.single_cls,
                                 dataloader=testloader,
                                 multi_label=ni > n_burn, 
                                 clusters=clusters,
                                 class_to_cluster_list=class_to_cluster_list,
                                 cluster_idx=opt.cluster_idx,
                                 backbone=backbone)

        # Write
        with open(results_file, 'a') as f:
            f.write(s + '%10.3g' * 7 % results + '\n')  # P, R, mAP, F1, test_losses=(GIoU, obj, cls)
        if len(opt.name) and opt.bucket:
            os.system('gsutil cp results.txt gs://%s/results/results%s.txt' % (opt.bucket, opt.name))

        # Tensorboard
        if tb_writer:
            tags = ['train/giou_loss', 'train/obj_loss', 'train/cls_loss',
                    'metrics/precision', 'metrics/recall', 'metrics/mAP_0.5', 'metrics/F1',
                    'val/giou_loss', 'val/obj_loss', 'val/cls_loss']
            for x, tag in zip(list(mloss[:-1]) + list(results), tags):
                tb_writer.add_scalar(tag, x, epoch)

        # Update best mAP
        fi = fitness(np.array(results).reshape(1, -1))  # fitness_i = weighted combination of [P, R, mAP, F1]
        if fi > best_fitness:
            best_fitness = fi

        # Save model
        save = (not opt.nosave) or (final_epoch and not opt.evolve)
        if save:
            with open(results_file, 'r') as f:  # create checkpoint
                ckpt = {'epoch': epoch,
                        'best_fitness': best_fitness,
                        'training_results': f.read(),
                        'model': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
                        'optimizer': None if final_epoch else optimizer.state_dict()}

            # Save last, best and delete
            torch.save(ckpt, last)
            if (best_fitness == fi) and not final_epoch:
                torch.save(ckpt, best)
            del ckpt

        # end epoch ----------------------------------------------------------------------------------------------------
    # end training

    n = opt.name
    if len(n):
        n = '_' + n if not n.isnumeric() else n
        fresults, flast, fbest = 'results%s.txt' % n, wdir + 'last%s.pt' % n, wdir + 'best%s.pt' % n
        for f1, f2 in zip([wdir + 'last.pt', wdir + 'best.pt', 'results.txt'], [flast, fbest, fresults]):
            if os.path.exists(f2):
                #os.rename(f1, f2)  # rename
                ispt = f2.endswith('.pt')  # is *.pt
                strip_optimizer(f2) if ispt else None  # strip optimizer
                os.system('gsutil cp %s gs://%s/weights' % (f2, opt.bucket)) if opt.bucket and ispt else None  # upload

    print('%g epochs completed in %.3f hours.\n' % (epoch - start_epoch + 1, (time.time() - t0) / 3600))
    dist.destroy_process_group() if torch.cuda.device_count() > 1 else None
    torch.cuda.empty_cache()
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=300)  # 500200 batches at bs 16, 117263 COCO images = 273 epochs
    parser.add_argument('--batch-size', type=int, default=16)  # effective bs = batch_size * accumulate = 16 * 4 = 64
    parser.add_argument('--data', type=str, default='data/coco2014.data', help='*.data path')
    parser.add_argument('--multi-scale', action='store_true', help='adjust (67%% - 150%%) img_size every 10 batches')
    parser.add_argument('--img-size', nargs='+', type=int, default=[320, 640], help='[min_train, max-train, test]')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--resume', action='store_true', help='resume training from last.pt')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--notest', action='store_true', help='only test final epoch')
    parser.add_argument('--evolve', action='store_true', help='evolve hyperparameters')
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
    parser.add_argument('--cache-images', action='store_true', help='cache images for faster training')
    parser.add_argument('--name', default='', help='renames results.txt to results_name.txt if supplied')
    parser.add_argument('--device', default='', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--adam', action='store_true', help='use adam optimizer')
    parser.add_argument('--single-cls', action='store_true', help='train as single-class dataset')
    parser.add_argument('--freeze-layers', action='store_true', help='Freeze non-output layers')
    parser.add_argument('--adaptive', action='store_true', help='train adaptive model')
    parser.add_argument('--model', type=str, default='model.args', help='File for the model configurations')
    parser.add_argument('--cluster_idx', type=int, default=0, help='Cluster Index to be trained')
    parser.add_argument('--train_bc_only', action='store_true', help='train controller only')
    parser.add_argument('--prune', action='store_true', help='train pruned model')

    opt = parser.parse_args()

    if opt.name:
        last += 'last' + opt.name + '.pt'
        best += 'best' + opt.name + '.pt'
        results_file = 'results'  + opt.name + '.txt'

    else:
        last += 'last.pt'
        best += 'best.pt'
        results_file = 'results.txt'

    opt.data = check_file(opt.data)  # check file
    opt.model = check_file(opt.model)  # check file

    model_args = parse_model_args(opt.model)

    if opt.adaptive or opt.train_bc_only:
        opt.clusters = check_file(model_args['clusters'])
        opt.backbone_cfg = check_file(model_args['backbone_cfg'])
        opt.backbone_weights = check_file(model_args['backbone_weights'])
        opt.branches_cfg = [check_file(f) for f in model_args['branches_cfg']]
        if 'branches_weights' in model_args:
            opt.branches_weights = [check_file(f) for f in model_args['branches_weights']]
        else:
            opt.branches_weights = None
        opt.branch_controller_cfg = check_file(model_args['branch_controller_cfg'])
        if 'branch_controller_weights' in model_args:
            opt.branch_controller_weights = check_file(model_args['branch_controller_weights'])
        else:
            opt.branch_controller_weights = None

    else:
        opt.cfg = check_file(model_args['cfg'])  # check file
        opt.weights = None
        if 'weights' in model_args:
            opt.weights = check_file(model_args['weights'])  # check file
            opt.weights = last if opt.resume and not opt.weights else opt.weights
    
    opt.img_size.extend([opt.img_size[-1]] * (3 - len(opt.img_size)))  # extend to 3 sizes (min, max, test)
    device = torch_utils.select_device(opt.device, apex=mixed_precision, batch_size=opt.batch_size)

    if device.type == 'cpu':
        mixed_precision = False

    # scale hyp['obj'] by img_size (evolved at 320)
    # hyp['obj'] *= opt.img_size[0] / 320.

    tb_writer = None
    if opt.train_bc_only:
        opt.name = opt.name + "_controller"
        tb_writer = SummaryWriter(comment=opt.name)
        train_branch_controller()  # train normally
        exit()
    if opt.adaptive:
        name_prefix = opt.name + str(opt.cluster_idx)
        # train_branch_controller()
        
        opt.cfg = opt.branches_cfg[opt.cluster_idx]
        if opt.branches_weights:
            opt.weights = opt.branches_weights[opt.cluster_idx]
        else:
            opt.weights = ""
        print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')
        tb_writer = SummaryWriter(comment=opt.name)
        train(hyp)  # train normally


    else:
        print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')
        tb_writer = SummaryWriter(comment=opt.name)
        train(hyp)  # train normally
