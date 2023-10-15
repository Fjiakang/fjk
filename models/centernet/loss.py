import torch
import torch.nn as nn
import torch.nn.functional as F
from libs.utils.centernet_aug import _transpose_and_gather_feat, _sigmoid


class FocalLoss(nn.Module):
    """nn.Module warpper for focal loss"""

    def __init__(self):
        super().__init__()
        self.neg_loss = _neg_loss

    def forward(self, out, target):
        return self.neg_loss(out, target)


class RegL1Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, output, mask, ind, target):
        pred = _transpose_and_gather_feat(output, ind)
        mask = mask.unsqueeze(2).expand_as(pred).float()
        # loss = F.l1_loss(pred * mask, target * mask, reduction='elementwise_mean')
        loss = F.l1_loss(pred * mask, target * mask, size_average=False)
        loss = loss / (mask.sum() + 1e-4)
        return loss


def _neg_loss(pred, gt):

    """
    Args:
        pred : (batch x c x h x w)
        gt : (batch x c x h x w)
    Returns:
        Modified focal loss.
    Reference:
        This code is base on
        CornerNet (https://github.com/princeton-vl/CornerNet)
        Copyright (c) 2018, University of Micshigan
    """

    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()

    neg_weights = torch.pow(1 - gt, 4)

    loss = 0

    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos
    return loss


class CtdetLoss(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.crit = FocalLoss()
        self.crit_reg = RegL1Loss()
        self.crit_wh = self.crit_reg
        self.args = args

    def forward(self, outputs, target):
        hm_loss, wh_loss, off_loss = 0, 0, 0
        for s in range(self.args.num_stacks):
            output = outputs[s]
            output["hm"] = _sigmoid(output["hm"])

            hm_loss += self.crit(output["hm"], target["hm"]) / self.args.num_stacks

            wh_loss += self.crit_reg(output["wh"], target["reg_mask"], target["ind"], target["wh"]) / self.args.num_stacks
            off_loss += self.crit_reg(output["reg"], target["reg_mask"], target["ind"], target["reg"]) / self.args.num_stacks

        loss = self.args.hm_weight * hm_loss + self.args.wh_weight * wh_loss + self.args.off_weight * off_loss
        loss_stats = {"loss": loss, "hm_loss": hm_loss, "wh_loss": wh_loss, "off_loss": off_loss}

        return loss, loss_stats
