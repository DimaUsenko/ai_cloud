from torchvision.models.detection.faster_rcnn import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
import torch


class FRCNNObjectDetector(FasterRCNN):
    def __init__(self, pretrained_weights_path=None, num_classes=8, **kwargs):
        if pretrained_weights_path is None:
            backbone = resnet_fpn_backbone('resnet50', True)
            super(FRCNNObjectDetector, self).__init__(backbone, num_classes, **kwargs)
        else:
            backbone = resnet_fpn_backbone('resnet50', False)
            super(FRCNNObjectDetector, self).__init__(backbone, num_classes, **kwargs)
            self.load_state_dict(torch.load(pretrained_weights_path))
