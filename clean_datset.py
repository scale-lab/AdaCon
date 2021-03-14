import os
import sys
import argparse
from utils.parse_config import parse_data_cfg

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/coco2014.data", help="path to data config file")
    parser.add_argument("--output_name", type=str, default="modified", help="name of output modified files")

    opt = parser.parse_args()

    data_config = parse_data_cfg(opt.data)
    train_path = data_config["train"]
    valid_path = data_config["valid"]

    dir_name = os.path.dirname(train_path)

    for data_path in [train_path, valid_path]:
        shapes_path = data_path.replace(".txt", ".shapes")

        img_files = None
        with open(data_path, "r") as file:
            img_files = file.readlines()

        label_files = [
            os.path.join(dir_name, path.rstrip().replace("./", ""). \
                replace("images", "labels").replace(".png", ".txt").replace(".jpg", ".txt"))
            for path in img_files
        ]

        shapes = []
        with open(shapes_path, "r") as file:
            shapes = file.readlines()
        
        files_to_keep = []
        shapes_to_keep = []
        count = 0
        for i, f in enumerate(label_files):
            f = f.rstrip()
            if os.path.exists(f):
                files_to_keep.append(img_files[i])
                shapes_to_keep.append(shapes[i])
            else:
                count += 1
        print(f'Removed {count} out of {len(label_files)}')

        with open(data_path, 'w') as filehandle:
            filehandle.writelines("%s\n" % img.rstrip() for img in files_to_keep)

        with open(shapes_path, 'w') as filehandle:
            filehandle.writelines("%s\n" % shape.rstrip() for shape in shapes_to_keep)