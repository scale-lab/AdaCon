import torchvision.transforms as transforms
import torchvision
import torch 
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.utils import *
from torch.utils.tensorboard import SummaryWriter
from utils.datasets_native import COCODataset
from test_retinanet import evaluate_coco
from model.retinanet import adacon_retinanet_resnet50_fpn
from utils.parse_config import parse_clusters_config

if __name__ == '__main__':
    train_ann = "data/coco/annotations/instances_train2014.json"
    train_root = "data/coco/images/train2014/"

    val_ann = "data/coco/annotations/instances_val2017.json"
    val_root = "data/coco/images/val2017/"

    weights = None

    batch_size = 16

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = COCODataset(root=train_root,
                            annFile=train_ann,
                            transform=transforms.ToTensor(),
                            train=True)
    train_dataloader = DataLoader(train_dataset,
                                batch_size=batch_size,
                                num_workers=min(batch_size,8),
                                pin_memory=True,
                                collate_fn=train_dataset.collate_fn)

    val_dataset = COCODataset(root=val_root,
                        annFile=val_ann,
                        transform=transforms.ToTensor())

    val_dataloader = DataLoader(val_dataset,
                                batch_size=1,
                                num_workers=min(1,8),
                                pin_memory=True,
                                collate_fn=val_dataset.collate_fn)

    # model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    clusters = parse_clusters_config('clusters.data')
    model = adacon_retinanet_resnet50_fpn(clusters, active_branch=0, num_branches=len(clusters), pretrained_backbone=True)

    model.train()
    count_parameters(model)
    model.to(device)

    if weights:
        model.load_state_dict(torch.load(weights))

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    tb_writer = SummaryWriter(comment="RetinaNet")

    for epoch in range(10):
        model.train()
        mloss = torch.zeros(3).to(device)
        print("Epoch ", epoch, len(train_dataloader))
        for batch_i, (imgs, _, reg_targets, cls_targets, _, _) in enumerate(tqdm(train_dataloader)):
            # try:
            imgs = imgs.to(device)
            if len(reg_targets) < len(imgs):
                continue
            targets = [{'boxes': reg_target.to(device), 'labels': cls_target.to(device)} for reg_target, cls_target in zip(reg_targets, cls_targets)]
            
            optimizer.zero_grad()

            loss = model(imgs, targets)
            total_loss = loss['classification'] + loss['bbox_regression']
            total_loss.backward()
            if batch_i % 100 == 0:
                print(total_loss)
            
            # temp = [loss['classification'], loss['bbox_regression'], total_loss]
            # mloss = [(mloss[i]*batch_i + temp[i])/ (batch_i + 1) for i in range(len(temp))]
            optimizer.step()
            # except:
            #     print("Exception")
            #     continue

        torch.save(model.state_dict(), str(epoch) + "retinanet.pt")

        results = evaluate_coco(val_dataloader, model, detection_threshold=0.6)

        if tb_writer:
            tags = ['train/cls_loss', 'train/reg_loss', 'train/total_loss',
            'metrics/mAP', 'metrics/mAP.5', 'metrics/mAP.75', 
            'metrics/mAPs', 'metrics/mAPm', 'metrics/mAPl']
            for x, tag in zip(list(mloss) + list(results[0:6]), tags):
                tb_writer.add_scalar(tag, x, epoch)
