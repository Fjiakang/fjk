import torch.nn as nn
from .yolox_head import YOLOXHead
from .yolox_backbone import YOLOPAFPN


class YOLOX(nn.Module):
    def __init__(self, args, backbone=None, head=None):
        super().__init__()
        self.backbone = YOLOPAFPN(args) if backbone == None else head

        self.head = YOLOXHead(args) if head == None else head

    def forward(self, x):
        fpn_outs = self.backbone(x)
        cls_output, reg_output, obj_output = self.head(fpn_outs)
        return {"cls_output": cls_output, "reg_output": reg_output, "obj_output": obj_output}
