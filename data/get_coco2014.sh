#!/bin/bash
# Zip coco folder
# zip -r coco.zip coco
# tar -czvf coco.tar.gz coco

# Download labels (Original files are taken from the authors of https://github.com/ultralytics/yolov3 
# but we uploaded them to our servers to add some modifications)
wget http://scale.engin.brown.edu/tools/AdaCon/coco2014labels.zip
unzip coco2014labels.zip
rm coco2014labels.zip

echo 'Downloaded Labels'
echo 'Unzipping Labels ... (takes ~ 30mins)'

# Unzip labels
unzip ${filename} # for coco.zip
# tar -xzf ${filename} # for coco.tar.gz
rm ${filename}

echo 'Unzipped Labels'

# Download and unzip images
cd coco/images
f="train2014.zip" && curl http://images.cocodataset.org/zips/$f -o $f && unzip -q $f && rm $f
f="val2014.zip" && curl http://images.cocodataset.org/zips/$f -o $f && unzip -q $f && rm $f

# cd out
cd ../..
