import argparse

from retinanet.adacon_model import retinanet_resnet50_fpn, adacon_retinanet_resnet50_fpn
from utils.datasets import *
from utils.utils import *

ONNX_EXPORT = False 
random.seed(123)

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
    

def coco80_to_coco91_class():  # converts 80-index (val2014) to 91-index (paper)
    # https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34,
         35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
         64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]
    return x

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

def main(args, save_img=False):
    num_classes = 91
    coco80to91 = coco80_to_coco91_class()
    #device = torch_utils.select_device(device='cpu' if ONNX_EXPORT else args.device)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    min_size, max_size = get_image_size_range(args.img_size)
    if args.adaptive:
        clusters = parse_clusters_config(args.clusters)
        active_branch = args.active_branch
        model = adacon_retinanet_resnet50_fpn(clusters=clusters, active_branch=active_branch, num_branches=len(clusters),
                            num_classes=num_classes, trainable_backbone_layers=0,
                            pretrained=args.pretrained, pretrained_backbone=args.pretrained_backbone, branches_weights=args.branches,
                            backbone_weights=args.backbone_weights, bc_weights=args.branch_controller,
                            min_size=min_size, max_size=max_size, deploy=args.deploy)
        freeze_adacon_retinanet_all_non_active_layers(model, active_branch, False)

        if args.oracle:
            model.oracle = True
        if args.single:
            model.singleb = True
        if args.multi:
            model.multib = True
        model.bc_thres = args.bc_thres        
    else:
        model = retinanet_resnet50_fpn(num_classes=num_classes, trainable_backbone_layers=0,
                            pretrained=args.pretrained, pretrained_backbone=args.pretrained_backbone, 
                            backbone_weights="retinanet_coco_backbone.pt", head_weights="retinanet_coco_head.pt",
                            min_size=min_size, max_size=max_size)
    model.eval()
    model.to(device)

    imgsz = (320, 192) if ONNX_EXPORT else args.img_size  # (320, 192) or (416, 256) or (608, 352) for (height, width)
    out, source, view_img, save_txt = args.output, args.source, args.view_img, args.save_txt
    webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

    # Initialize
    # device = torch_utils.select_device(device='cpu' if ONNX_EXPORT else opt.device)
    if not os.path.exists(out):
        #     shutil.rmtree(out)  # delete output folder
        os.makedirs(out)  # make new output folder

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        torch.backends.cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        save_img = True
        dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names = load_classes(args.names)
    nc = len(names)
    print("nc ", nc)
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # Run inference
    t0 = time.time()
    warmup_frames = 10 
    for idx, (path, img, im0s, vid_cap) in enumerate(dataset):
        if idx == 50:
            break
        img = torch.from_numpy(img).to(device)
        img = img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = torch_utils.time_synchronized()
        outputs = model(img)

        t2 = torch_utils.time_synchronized()

        outputs = [{k: v.to("cpu") for k, v in t.items()} for t in outputs]

        # Process detections
        for i, output in enumerate(outputs):  # detections for image i
            box = output["boxes"]
            score = output["scores"]
            label = output["labels"]

            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s

            save_path = str(Path(out) / Path(p).name)
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  #  normalization gain whwh
            if box is not None and len(box):
                # Rescale boxes from imgsz to im0 size
                box = scale_coords(img.shape[2:], box, im0.shape).round()

                # Print results
                for c in label.unique():
                    n = (label == c).sum()  # detections per class
                    # s += '%g %ss, ' % (n, names[int(c)])  # add to string

                # Write results
                for l, b, s in zip(label, box, score):
                    if s < 0.5:
                        continue
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(b).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        with open(save_path[:save_path.rfind('.')] + '.txt', 'a') as file:
                            file.write(('%g ' * 5 + '\n') % ("cls", *b))  # label format

                    if save_img or view_img:  # Add bbox to image
                        lbl = '%s %.2f' % (names[coco80to91.index(l)], s)
                        plot_one_box(b, im0, label=lbl, color=colors[int(coco80to91.index(l))])

            # Print time (inference + NMS)
            print('%sDone. (%.3fs)' % (s, t2 - t1))

            # Stream results
            if view_img:
                cv2.imshow(p, im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'images':
                    cv2.imwrite(save_path, im0)
                else:
                    if idx == warmup_frames:
                        t0 = time.time()
                    elif idx == warmup_frames + 10:
                        correct_fps = 10/(time.time() - t0)
                    elif idx > warmup_frames + 10:
                        if vid_path != save_path:  # new video
                            vid_path = save_path
                            if isinstance(vid_writer, cv2.VideoWriter):
                                vid_writer.release()  # release previous video writer
                            # fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            print(w, h)
                            vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*args.fourcc), correct_fps, (w, h))
                        cv2.putText(im0, 
                            str("FPS = {:.3f}".format(idx/(time.time() - t0))), 
                            (250, 150), 
                            cv2.FONT_HERSHEY_SIMPLEX, 2, 
                            (255, 255, 255), 
                            4)
                        vid_writer.write(im0)

    if save_txt or save_img:
        print('Results saved to %s' % os.getcwd() + os.sep + out)
        if platform == 'darwin':  # MacOS
            os.system('open ' + save_path)

    print('Done. (%.3fs)' % (time.time() - t0))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description=__doc__)
    parser.add_argument('--names', type=str, default='data/coco.names', help='*.names path')
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
    parser.add_argument('--source', type=str, default='data/samples', help='source')  # input file/folder, 0 for webcam
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
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
    parser.add_argument('--output', type=str, default='output', help='output folder')  # output folder

    args = parser.parse_args()

    main(args)
