#!/bin/bash

# Get yolov3 weights (used as backbone)
wget -c https://pjreddie.com/media/files/yolov3.weights -P weights
# wget -c https://pjreddie.com/media/files/yolov3-tiny.weights
# wget -c https://pjreddie.com/media/files/yolov3-spp.weights

# Get AdaCon branch weights 
wget -c http://scale.engin.brown.edu/tools/AdaCon/adacon_branch0v0.pt -P weights
wget -c http://scale.engin.brown.edu/tools/AdaCon/adacon_branch1v0.pt -P weights
wget -c http://scale.engin.brown.edu/tools/AdaCon/adacon_branch2v0.pt -P weights
wget -c http://scale.engin.brown.edu/tools/AdaCon/adacon_branch3v0.pt -P weights

# Get AdaCon Branch Controller
wget -c http://scale.engin.brown.edu/tools/AdaCon/adacon_branch_controller.pt -P weights
