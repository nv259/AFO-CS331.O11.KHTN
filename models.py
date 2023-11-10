import numpy as np
import torch

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.retinanet import RetinaNetClassificationHead
from torchvision.models.detection.ssd import SSDHead

from functools import partial


def load_checkpoint(model, path_to_checkpoint, device=torch.device('cuda')):
    model.to(device)
    checkpoint = torch.load(path_to_checkpoint)
    model.load_state_dict(checkpoint["model_state_dict"])
    # epoch = checkpoint["epoch"]
    # best_val_loss = checkpoint["val_loss"]
    
    return model


def createFaster_RCNN(num_classes=3):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights = "DEFAULT")
    num_classes = num_classes
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        
    return model 


def createRetinaNet(num_classes=3):
    model = torchvision.models.detection.retinanet_resnet50_fpn(weights="DEFAULT")
    num_anchors = model.head.classification_head.num_anchors

    model.head.classification_head = RetinaNetClassificationHead(
        in_channels=256,
        num_anchors=num_anchors,
        num_classes=num_classes,
        norm_layer=partial(torch.nn.GroupNorm, 32)
    )

    return model 


def createSSD300(num_classes=3):
    model = torchvision.models.detection.ssd300_vgg16(weights="DEFAULT", trainable_backbone_layers=2)
    model.head = SSDHead(in_channels = [512, 1024, 512, 256, 256, 256],
                        num_anchors = model.anchor_generator.num_anchors_per_location(),
                        num_classes=num_classes)

    return model