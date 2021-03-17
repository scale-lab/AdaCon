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

    # Read Template branch Config file (Original Tiny YOLO)
    template_path = "cfg/yolov3.cfg"

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
    # Choose 4x compressed for the first 6 layers and 2x compressed for the rest of the layers
    
    param = []
    for i, l in enumerate(search_space):
        if i < 6:
            param.append(l[1])
        else:
            param.append(l[0])

    for bnch_num in range(num_branches):
        print(bnch_num)
        num_classes = num_classes_per_branch[bnch_num]

        # Write all found architectures

        print("Writing file with params ", param)
        file_name = "arch_bnch" + str(bnch_num+1) + "_" +  str(num_branches) + "arch2.cfg"
        new_module_defs = module_defs.copy()

        file = open(output_path+file_name, 'w')
        change_filt = False
        filt_idx = 0
        for idx, module in enumerate(new_module_defs):
            if idx > 0 and idx <= backbone_limit:
                continue
            for key in module:
                if key == "type":
                    if module[key] == "convolutional" and idx > backbone_limit:
                        change_filt = True                        
                    file.write("["+module['type']+"]")
                    file.write("\n")

                else:
                    if change_filt == True and key == "filters":
                        if math.ceil(Log2(int(module[key]))) == math.floor(Log2(int(module[key]))):
                            module[key] = param[filt_idx]
                            filt_idx += 1
                            change_filt = False
                        else:
                            module[key] = 3*(num_classes + 5)
                    elif key == "classes":
                        module[key] = num_classes
                            
                    file.write (key +"=" + str(module[key]))
                    file.write("\n")

                    if key == "layers" and len(module[key].split(",")) > 1:
                        size = new_module_defs[int(module[key].split(",")[-1])]['filters']
                        file.write ("filters=" + str(size))
                        file.write("\n")
            file.write("\n")
            file.write("\n")

        file.close()

            # model = Darknet(output_path+file_name).to(device)
            # # print("MODEL CREATED")
            # with torch.cuda.device(0):
            #     macs, params = get_model_complexity_info(model, (3, 416, 416), print_per_layer_stat = False)
            #     macs = float(macs.split(" ")[0])*(10**9)
            #     params = float(params.split(" ")[0])*(10**6)
            #     print(macs, mac_backbone + mac_limit_per_branch[0], params, params_backbone + param_limit_per_branch[bnch_num])

            # if (macs > mac_backbone + mac_limit_per_branch[0]) or (params > params_backbone + param_limit_per_branch[bnch_num]):
            #         print(file_name, "REJECTED")
            #         os.remove(output_path+file_name)
