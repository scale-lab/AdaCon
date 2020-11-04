from __future__ import division

from models import *
from utils.utils import *

from terminaltables import AsciiTable

import os
import sys
import time
import datetime
import argparse

import torch, torch.nn as nn

from torch.utils.data import DataLoader, Sampler
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim
import itertools 

from ptflops import get_model_complexity_info

def Log2(x):
    if x == 0:
        return False
 
    return (math.log10(x) /
            math.log10(2))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="cfg/yolov3.cfg", help="path to model definition file")
    parser.add_argument("--data", type=str, default="data/coco2014.data", help="path to data config file")
    parser.add_argument("--clusters_path", type=str, default="clusters.data", help="clusters file path")
    parser.add_argument("--ckpt_prefix", type=str, default="", help="pre for checkpoints files")

    opt = parser.parse_args()
    print(opt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Get data configuration
    data_config = parse_data_cfg(opt.data)
    train_path = data_config["train"]
    valid_path = data_config["valid"]
    class_names = load_classes(data_config["names"])
    num_classes = int(data_config["classes"])


    # Read clusters file 
    clusters = parse_clusters_config(opt.clusters_path)
    num_branches = len(clusters)
    num_classes_per_branch = np.asarray([len(cluster) for cluster in clusters])
    weight_classes_per_branch = 1/sum(num_classes_per_branch) *  num_classes_per_branch

    # Find Params and MAC for backbone
    backbone_path = "config/adayolo_backbone.cfg"
    model = Backbone(backbone_path).to(device)

    # print("MODEL CREATED")
    with torch.cuda.device(0):
        macs, params = get_model_complexity_info(model, (3, 416, 416), print_per_layer_stat = False)
        mac_backbone = float(macs.split(" ")[0])*(10**9)
        params_backbone = float(params.split(" ")[0])*(10**6)
    print("BACKBONE", macs, params)


    # Read Template branch Config file (Original Tiny YOLO)
    template_path = "cfg/yolov3.cfg"

    model = Darknet(template_path).to(device)

    # print("MODEL CREATED")
    with torch.cuda.device(0):
        macs, params = get_model_complexity_info(model, (3, 416, 416), print_per_layer_stat = False)
        mac_baseline = float(macs.split(" ")[0])*(10**9)
        params_baseline = float(params.split(" ")[0])*(10**6)
    print("Template", macs, params)


    param_limit = (params_baseline - params_backbone) * 0.9 # Allow at lease 5% smaller size
    mac_limit = (mac_baseline - mac_backbone) * 0.75 # Allow at least 25% less operations

    # weight_classes_per_branch = np.asarray(weight_classes_per_branch)
    param_limit_per_branch = param_limit*weight_classes_per_branch
    mac_limit_per_branch = mac_limit*weight_classes_per_branch

    print(param_limit_per_branch, mac_limit_per_branch)


    backbone_limit = 75
    output_path = "generated_archs/"
    file = open(template_path, 'r')
    lines = file.read().split('\n')
    lines = [x for x in lines if x and not x.startswith('#')]
    lines = [x.rstrip().lstrip() for x in lines] # get rid of fringe whitespaces
    module_defs = []

    search_space = []
    for line in lines:
        if line.startswith('['): # This marks the start of a new block
            module_defs.append({})
            module_defs[-1]['type'] = line[1:-1].rstrip()
            if module_defs[-1]['type'] == 'convolutional':
                module_defs[-1]['batch_normalize'] = 0
        else:
            key, value = line.split("=")
            value = value.strip()
            module_defs[-1][key.rstrip()] = value.strip()
            if module_defs[-1]['type'] == 'convolutional':
                if len(module_defs) > backbone_limit+1 and key == "filters": # Net
                    start_filter_size = int(value)
                    if math.ceil(Log2(start_filter_size)) == math.floor(Log2(start_filter_size)):
                        search_space_per_layer = []
                        for i in range(1,3):
                            search_space_per_layer.append(int(start_filter_size/(2**i)))
                        search_space.append(search_space_per_layer)
    
    print(search_space)
    search_space = list(itertools.product(*search_space))

    for bnch_num in range(num_branches):
        print(bnch_num)
        num_classes = num_classes_per_branch[bnch_num]

        # Write all found architectures

        for i, param in enumerate(search_space):
            if np.average(param) < 70:
                continue
            print("Writing file with params ", param)
            file_name = "arch_bnch" + str(bnch_num) + "_" +  str(i) + ".cfg"
            new_module_defs = module_defs.copy()

            file = open(output_path+file_name, 'w')
            change_filt = False
            filt_idx = 0
            for idx, module in enumerate(new_module_defs):
                # print(module)
                for key in module:
                    if key == "type":
                        if module[key] == "convolutional" and idx > backbone_limit:
                            change_filt = True                        
                        file.write("["+module['type']+"]")
                        file.write("\n")

                    else:
                        if change_filt == True and key == "filters":
                            if math.ceil(Log2(int(module[key]))) == math.floor(Log2(int(module[key]))):
                                # print(idx, i, param, "Changing filt from", module[key], param[filt_idx])
                                module[key] = param[filt_idx]
                                filt_idx += 1
                                change_filt = False
                            else:
                                module[key] = 3*(num_classes + 5)
                        if key == "classes":
                            module[key] = num_classes
                                
                        file.write (key +"=" + str(module[key]))
                        file.write("\n")

                file.write("\n")
                file.write("\n")

            file.close()

            model = Darknet(output_path+file_name).to(device)
            # print("MODEL CREATED")
            with torch.cuda.device(0):
                macs, params = get_model_complexity_info(model, (3, 416, 416), print_per_layer_stat = False)
                macs = float(macs.split(" ")[0])*(10**9)
                params = float(params.split(" ")[0])*(10**6)
                print(macs, mac_backbone + mac_limit_per_branch[0], params, params_backbone + param_limit_per_branch[bnch_num])

            if (macs > mac_backbone + mac_limit_per_branch[0]) or (params > params_backbone + param_limit_per_branch[bnch_num]):
                    print(file_name, "REJECTED")
                    os.remove(output_path+file_name)


    # Generate all possible architectures
    # Dismiss the architectures passing the # parameters and MACs limit
    # Train each of the remaining architectures for 5 epochs snd choose the best
    # Number of expected architectures 
    # 3 kernel size options per layer for 4 layers per branch
    # 3^4 = 81 options per branch 
    # Assuming 4 branches ==> 81^4 = 43M ???





    '''


    # Initiate model
    model = AdaptiveYOLO(opt.model_def).to(device)
    count_parameters(model)
    model.num_all_classes = num_classes
    
    ############## READ Clusters file and Create mapping ##########
    print(len(clusters))
    class_to_cluster_list = []

    ## create the class-cluster map to be used for labels in split training
    for cluster in clusters:
        class_to_cluster = {}
        cluster_to_class = {}
        for i, element in enumerate(cluster):
            class_to_cluster[element] = i
            cluster_to_class[i] = element

        class_to_cluster_list.append(class_to_cluster)

    ## Set the clusters and cluster mapping for the model
    model.mode_dicts_class_to_cluster = class_to_cluster_list
    model.mode_classes_list = clusters

    model.apply(weights_init_normal)

    # If specified we start from checkpoint
    if opt.pretrained_weights:
        if opt.pretrained_weights.endswith(".pth"):
            model.load_state_dict(torch.load(opt.pretrained_weights))
        else:
            model.load_darknet_weights(opt.pretrained_weights,opt.frozen_pretrained_layers)

    # Freeze the loaded layers
    for i, (name, param) in enumerate(model.named_parameters()):
        if i <= opt.frozen_pretrained_layers:
            print("Freeze ", name, " ", i)
            param.requires_grad = False
    
    optimizer = torch.optim.Adam(model.parameters())

    metrics = [
        "grid_size",
        "loss",
        "x",
        "y",
        "w",
        "h",
        "conf",
        "cls",
        "cls_acc",
        "recall50",
        "recall75",
        "precision",
        "conf_obj",
        "conf_noobj",
    ]

    with torch.cuda.device(0):
        macs, params = get_model_complexity_info(model, (3, 416, 416), as_strings=True, print_per_layer_stat=True, verbose=True)
        print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        print('{:<30}  {:<8}'.format('Number of parameters: ', params))

    #### Alternate between clusters at each epoch
    mode_i = 0
    best_model = model
    best_map = 0

    for epoch in range(opt.epochs):
        model.modes = [mode_i]
        dataset = ListDataset(train_path, augment=True, multiscale=opt.multiscale_training)
        
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=opt.n_cpu,
            pin_memory=True,
            collate_fn=dataset.collate_fn,
        )

        model.train()
        start_time = time.time()
        for batch_i, (_, imgs, targets) in enumerate(dataloader):
            batches_done = len(dataloader) * epoch + batch_i

            imgs = Variable(imgs.to(device))
            targets = Variable(targets.to(device), requires_grad=False)

            loss, outputs = model(imgs, targets)

            if outputs == None:
                continue
            loss.backward()

            print(loss, ">>>>>>>>>>>>>>>>>>")
            # Accumulates gradient before each step
            optimizer.step()
            optimizer.zero_grad()

            # ----------------
            #   Log progress
            # ----------------

            log_str = "\n---- [Cluster %d, Epoch %d/%d, Batch %d/%d] ----\n" % (mode_i, epoch, opt.epochs, batch_i, len(dataloader))

            metric_table = [["Metrics", *[f"YOLO Layer {i}" for i in range(len(model.yolo_layers))]]]

            # Log metrics at each YOLO layer
            for i, metric in enumerate(metrics):
                formats = {m: "%.6f" for m in metrics}
                formats["grid_size"] = "%2d"
                formats["cls_acc"] = "%.2f%%"
                row_metrics = [formats[metric] % yolo.metrics.get(metric, 0) for yolo in model.yolo_layers]
                metric_table += [[metric, *row_metrics]]

                # Tensorboard logging
                tensorboard_log = []
                for j, yolo in enumerate(model.yolo_layers):
                    for name, metric in yolo.metrics.items():
                        if name != "grid_size":
                            tensorboard_log += [(f"{name}_{j+1}", metric)]
                tensorboard_log += [("loss", loss.item())]
                logger.list_of_scalars_summary(tensorboard_log, batches_done)

            log_str += AsciiTable(metric_table).table
            log_str += f"\nTotal loss {loss.item()}"

            # Determine approximate time left for epoch
            epoch_batches_left = len(dataloader) - (batch_i + 1)
            time_left = datetime.timedelta(seconds=epoch_batches_left * (time.time() - start_time) / (batch_i + 1))
            log_str += f"\n---- ETA {time_left}"

            print(log_str)

            model.seen += imgs.size(0)

        torch.save(model.state_dict(), f"checkpoints/%s_yolov3_ckpt_clus%d_%d.pth" %(opt.ckpt_prefix, mode_i, epoch))

        print(f"\n---- Evaluating Model on Cluster ----", mode_i)
        # Evaluate the model on the validation set
        precision, recall, AP, f1, ap_class = evaluate(
            model,
            path=valid_path,
            iou_thres=0.3,
            conf_thres=0.3,
            nms_thres=0.3,
            img_size=opt.img_size,
            batch_size=1,
            max_bound=True,
        )
        evaluation_metrics = [
            ("val_precision", precision.mean()),
            ("val_recall", recall.mean()),
            ("val_mAP", AP.mean()),
            ("val_f1", f1.mean()),
        ]
        logger.list_of_scalars_summary(evaluation_metrics, epoch)

        # Print class APs and mAP
        ap_table = [["Index", "Class name", "AP"]]
        for i, c in enumerate(ap_class):
            ap_table += [[c, class_names[c], "%.5f" % AP[i]]]
        print(AsciiTable(ap_table).table)
        print(f"---- mAP {AP.mean()}")
        if AP.mean() > best_map:
            best_map = AP.mean()
            best_model = model

        mode_i = (mode_i + 1) % len(clusters)

    print("Saving best model of mAP", best_map)

    best_model.save_darknet_weights("weights/%s_yolov3_ada.weights" % opt.ckpt_prefix)
    torch.save(best_model.state_dict(), f"checkpoints/%s_yolov3_ada.pth" % opt.ckpt_prefix)
    '''
