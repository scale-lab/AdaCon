from utils.google_utils import *
from utils.layers import *
from utils.parse_config import *
import utils.extras as extras
from models import Darknet, Backbone, BranchController
from enum import Enum

class AdaConMode(Enum):
    multi_branch = 1
    single_branch = 2
    oracle = 3

class AdaConYolo(nn.Module):
    def __init__(self, model_args, img_size=(416,416), exec_mode=1, multi_branch_thres=0.1):
        super(AdaConYolo, self).__init__()
        model_args = parse_model_args(model_args)

        self.num_classes = int(model_args['num_classes'])
        clusters_file = model_args['clusters']

        self.clusters = parse_clusters_config(clusters_file)
        self.class_to_cluster_list = get_class_to_cluster_map(self.clusters)

        self.exec_mode = exec_mode
        self.multi_branch_thres = multi_branch_thres

        backbone_cfg = model_args['backbone_cfg']
        backbone_weights = model_args['backbone_weights']

        branch_controller_cfg = model_args['branch_controller_cfg']
        branch_controller_weights = None
        if 'branch_controller_weights' in model_args:
            branch_controller_weights = model_args['branch_controller_weights']

        branches_cfg = model_args['branches_cfg']
        branches_weights = None
        if 'branches_weights' in model_args:
            branches_weights = [check_file(f) for f in model_args['branches_weights']]

        self.backbone = Backbone(backbone_cfg)
        self.backbone.load_darknet_weights(backbone_weights, 100)

        self.branches = nn.ModuleList()
        for i, cfg in enumerate(branches_cfg):
            branch = Darknet(cfg, img_size)
            if branches_weights:
                branch.load_state_dict(torch.load(branches_weights[i])['model'])
            self.branches.append(branch)
        
        self.branch_controller = BranchController(branch_controller_cfg, len(self.clusters))
        if branch_controller_weights:
            self.branch_controller.load_state_dict(torch.load(branch_controller_weights))

    def forward(self, x):
        if self.training:
            return self._forward_training(x)
        else:
            return self._forward_testing(x)

    def _forward_training(self, x):
        back_out = self.backbone(x)
        preds = []
        for cluster_idx, branch in enumerate(self.branches):
            branch_out = branch(back_out, out=self.backbone.layer_outputs)

            preds.append(branch_out)

        self.backbone.layer_outputs = []
        
        return preds, self.branch_controller(back_out)

    def _forward_testing(self, x):
        back_out = self.backbone(x)

        if self.exec_mode == AdaConMode.single_branch.value:
            active_branches = [torch.argmax(self.branch_controller(back_out, []))]
        
        elif self.exec_mode == AdaConMode.multi_branch.value:
            class_out = self.branch_controller(back_out, [])
            active_branches = torch.where(class_out > self.multi_branch_thres)[1]
        elif self.exec_mode == AdaConMode.oracle.value:
            active_branches = np.arange(len(self.branches))

        preds = []
        for cluster_idx, branch in enumerate(self.branches):
            if cluster_idx not in active_branches:
                continue
            branch_out, _ = branch(back_out, out=self.backbone.layer_outputs)

            full_detection = torch.zeros(branch_out.shape[0], branch_out.shape[1], self.num_classes+5, device=x.get_device())
            full_detection[:, :, 0:5] = branch_out[:, :, 0:5]

            new_indices = [5 + k for k in self.clusters[cluster_idx]]
            full_detection[:, :, new_indices] = branch_out[:, :, 5:]
            preds.append(full_detection)
            
        self.backbone.layer_outputs = []
    
        return torch.cat(preds, 1)

    def backward(self, losses):
        for module in self.branches:
            for param in module.parameters():
                param.require_grad = False

        mean_losses = sum(losses)/len(losses)
        mean_losses.backward(retain_graph=True)

        for param in self.backbone.parameters():
                param.require_grad = False

        for i, loss in enumerate(losses):
            for param in self.branches[i].parameters():
                param.require_grad = True

            if i < len(losses) - 1:
                loss.backward(retain_graph=True)

                for param in self.branches[i].parameters():
                    param.require_grad = False
            else:
                loss.backward()
        
        for param in self.backbone.parameters():
                param.require_grad = True

        for module in self.branches:
            for param in module.parameters():
                param.require_grad = True