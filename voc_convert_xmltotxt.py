import xml.etree.ElementTree as ET
import os
import json


def write_train_images_set():

    ## WRITE TRAIN IMAGES SET IN A TEXT FILE
    images_dir = '/home/marina/OD/Adaptive_YOLO/data/voc/images'
    image_paths_file = '/home/marina/OD/Adaptive_YOLO/data/voc/train_set.txt'
    with open('../Adaptive_YOLO/data/pascal/VOCdevkit/TRAIN_images.json') as f:
        data = json.load(f)
        new_paths = []
        for f in data:
            np = os.path.join(images_dir,os.path.basename(f))
            exist = os.path.exists(np)
            if not exist:
                print(np, "does not exist")
            else:
                new_paths.append(np + "\n")
                print(np)
        print(len(new_paths))
        f = open(image_paths_file, 'w')
        f.seek(0,1)
        f.writelines(new_paths)
        f.close()


def write_val_images_set():
    ## WRITE VAL IMAGES SET IN A TEXT FILE
    image_paths_file = '/home/marina/OD/Adaptive_YOLO/data/voc/val_set.txt'
    with open('../Adaptive_YOLO/data/pascal/VOCdevkit/TEST_images.json') as f:
        data = json.load(f)
        new_paths = []
        for f in data:
            np = os.path.join(images_dir,os.path.basename(f))
            exist = os.path.exists(np)
            if not exist:
                print(np, "does not exist")
            else:
                new_paths.append(np + "\n")
                print(np)
        print(len(new_paths))
        f = open(image_paths_file, 'w')
        f.seek(0,1)
        f.writelines(new_paths)
        f.close()


def xml_to_txt():    
    ## WRITE LABELS IN TXT FORMAT (CONVERT FROM XML)
    directory07 = '../Adaptive_YOLO/data/pascal/VOCdevkit/VOC2012/Annotations'
    write_dir = '/home/marina/OD/Adaptive_YOLO/data/voc/labels'


    voc_labels = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
              'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
 
    for filename in os.listdir(directory07):
        if filename.endswith('.xml'):
            tree = ET.parse(os.path.join(directory07, filename))
            root = tree.getroot()
            # Extract file name
            f_name = root.find('filename').text
            print(f_name)
            # Extract the image dimentions
            size = root.find('size')
            width = int(size.find('width').text)
            height = int(size.find('height').text)

            f_objects = []
            for obj in root.findall('object'):
                # Extract class name
                name = obj.find('name').text
                name = voc_labels.index(name)
                if name == -1:
                    print("NAME NOT FOUND")
                name = str(name)

                # Extract the object bounding box 
                box = obj.find('bndbox')
                xmin = float(box.find('xmin').text)
                ymin = float(box.find('ymin').text)
                xmax = float(box.find('xmax').text)
                ymax = float(box.find('ymax').text)
                
                xmax = xmax - xmin
                ymax = ymax - ymin
                xmin = xmin + xmax/2
                ymin = ymin + ymax/2
                xmin = '%.3f'%(xmin/width)
                xmax = '%.3f'%(xmax/width)
                ymin = '%.3f'%(ymin/height)
                ymax = '%.3f'%(ymax/height)
            
                f_objects.append([name, xmin, ymin, xmax, ymax])

            with open(os.path.join(write_dir,f_name.rstrip('.jpg')+".txt"), 'w') as f:
                for line in f_objects:
                    for param in line:
                        f.write(param + ' ')

                    f.write('\n')
xml_to_txt()
