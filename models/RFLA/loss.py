import torch
from .head import distance2bbox, multi_apply

import torch.nn as nn
import numpy as np
from mmdet.core import build_assigner
import torch.distributed as dist
from mmdet.models import HEADS, build_loss
INF = 1e8

class LOSS(nn.Module):
    def __init__(self,args):
        super().__init__()
        # self.center_prior = CenterPrior(
        #     soft_prior=True,
        #     num_classes=10,
        self.strides=(8, 16, 32, 64, 128)
        self.regress_ranges=((-1, 64), (64, 128), (128, 256), (256, 512),
                                 (512, INF))
        self.num_classes = args.num_classes+1
        self.fpn_layer = 'p3'
        self.fraction  = 1/2
        assigner=dict(
            type='HieAssigner',
             ignore_iof_thr=-1,
             gpu_assign_thr=256,
             iou_calculator=dict(type='BboxDistanceMetric'),
             assign_metric='kl',
             topk=[3,1],
             ratio=0.9)
        loss_cls=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0)
        loss_centerness=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0)
        loss_bbox=dict(type='DIoULoss', loss_weight=1.0)
        self.loss_centerness = build_loss(loss_centerness)
        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)
        self.norm_on_bbox=False
        self.assigner = build_assigner(assigner)
# def loss(self,
#              cls_scores,
#              bbox_preds,
#              centernesses,
#              gt_bboxes,
#              gt_labels,
#              img_metas,
#              gt_bboxes_ignore=None):
        """Compute loss of the head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level,
                each is a 4D-tensor, the channel number is
                num_points * num_classes.
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level, each is a 4D-tensor, the channel number is
                num_points * 4.
            centernesses (list[Tensor]): centerness for each scale level, each
                is a 4D-tensor, the channel number is num_points * 1.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
    def get_targets(self, points, gt_bboxes_list, gt_labels_list):
        """Compute regression, classification and centerness targets for points
        in multiple images.

        Args:
            points (list[Tensor]): Points of each fpn level, each has shape
                (num_points, 2).
            gt_bboxes_list (list[Tensor]): Ground truth bboxes of each image,
                each has shape (num_gt, 4).
            gt_labels_list (list[Tensor]): Ground truth labels of each box,
                each has shape (num_gt,).

        Returns:
            tuple:
                concat_lvl_labels (list[Tensor]): Labels of each level. \
                concat_lvl_bbox_targets (list[Tensor]): BBox targets of each \
                    level.
        """
        assert len(points) == len(self.regress_ranges)
        
        # convert points to rf according to its layer
        rfields = []
        trfs = gen_trf()
        for num in range(len(points)):
            rfield=[]
            if self.fpn_layer == 'p3':
                rfnum = num +1
            else:
                rfnum = num

            if rfnum == 0:
                rf = trfs[0]*self.fraction
            elif rfnum == 1:
                rf = trfs[1]*self.fraction
            elif rfnum == 2:
                rf = trfs[2]*self.fraction
            elif rfnum == 3:
                rf = trfs[3]*self.fraction 
            elif rfnum == 4:
                rf = trfs[4]*self.fraction
            else:
                rf = trfs[5]*self.fraction
                      
            point = points[num]
            px1 = point[...,0] - rf/2
            py1 = point[...,1] - rf/2
            px2 = point[...,0] + rf/2
            py2 = point[...,1] + rf/2
            rfield = torch.cat((px1[...,None], py1[...,None]), dim=1)
            rfield = torch.cat((rfield, px2[...,None]), dim=1)
            rfield = torch.cat((rfield, py2[...,None]), dim=1)
            rfields.append(rfield)
        rfields = torch.cat(rfields, dim=0)
        num_levels = len(points)
        # expand regress ranges to align with points
        expanded_regress_ranges = [
            points[i].new_tensor(self.regress_ranges[i])[None].expand_as(  #[[-1,64],[-1,64]] num*2*2
                points[i]) for i in range(num_levels)
        ]
        # concat all levels points and regress ranges
        concat_regress_ranges = torch.cat(expanded_regress_ranges, dim=0) 
        concat_points = torch.cat(points, dim=0)

        # the number of points per img, per lvl
        num_points = [center.size(0) for center in points]

        # get labels and bbox_targets of each image
        labels_list, bbox_targets_list = multi_apply(
            self._get_target_single,
           
            gt_bboxes_list,
            gt_labels_list,
            points=concat_points,
            rfields=rfields, 
            regress_ranges=concat_regress_ranges,
            num_points_per_lvl=num_points)

        # split to per img, per level
        labels_list = [labels.split(num_points, 0) for labels in labels_list]
        bbox_targets_list = [
            bbox_targets.split(num_points, 0)
            for bbox_targets in bbox_targets_list
        ]

        # concat per level image
        concat_lvl_labels = []
        concat_lvl_bbox_targets = []
        for i in range(num_levels):
            concat_lvl_labels.append(
                torch.cat([labels[i] for labels in labels_list]))
            bbox_targets = torch.cat(
                [bbox_targets[i] for bbox_targets in bbox_targets_list])
            if self.norm_on_bbox:
                bbox_targets = bbox_targets / self.strides[i]
            concat_lvl_bbox_targets.append(bbox_targets)
        return concat_lvl_labels, concat_lvl_bbox_targets

    def _get_target_single(self, gt_bboxes, gt_labels, points, rfields, regress_ranges,
                           num_points_per_lvl):
        """Compute regression and classification targets for a single image."""
        num_points = points.size(0)
        num_gts = gt_labels.size(0)
        gt_bboxes = gt_bboxes.cuda()
        gt_labels = gt_labels.cuda()
        gt_ori = gt_bboxes
        if num_gts == 0:
            return gt_labels.new_full((num_points,), self.num_classes), \
                   gt_bboxes.new_zeros((num_points, 4))

        areas = (gt_bboxes[:, 2] - gt_bboxes[:, 0]) * (
            gt_bboxes[:, 3] - gt_bboxes[:, 1])
        # TODO: figure out why these two are different
        # areas = areas[None].expand(num_points, num_gts)
        areas = areas[None].repeat(num_points, 1) # points, gts
        regress_ranges = regress_ranges[:, None, :].expand(
            num_points, num_gts, 2)
        gt_bboxes = gt_bboxes[None].expand(num_points, num_gts, 4)
        xs, ys = points[:, 0], points[:, 1]
        xs = xs[:, None].expand(num_points, num_gts)
        ys = ys[:, None].expand(num_points, num_gts)

        left = xs - gt_bboxes[..., 0] #numpoints, num_gt
        right = gt_bboxes[..., 2] - xs
        top = ys - gt_bboxes[..., 1]
        bottom = gt_bboxes[..., 3] - ys
        bbox_targets = torch.stack((left, top, right, bottom), -1) # numpoints, num_gt, 4

        
        assign_result = self.assigner.assign(rfields, gt_ori, gt_bboxes_ignore=None)
        inds = assign_result.gt_inds #num_points
        inds = inds - 1

        inds_mask = inds[...,None].repeat(1, num_gts) #num_points, num_gts
        point_mask = torch.arange(num_gts).repeat(num_points, 1).cuda() # num_points, num_gts
        assigned_mask = (inds_mask == point_mask)
        areas[assigned_mask == False] = INF
        min_area, min_area_inds = areas.min(dim=1)


        labels = gt_labels[min_area_inds]
        labels[min_area == INF] = self.num_classes  # set as BG
        bbox_targets = bbox_targets[range(num_points), min_area_inds]


        return labels, bbox_targets
    def forward(self,preds,targets):
        
        cls_scores, bbox_preds,centernesses= preds
        gt_bboxes, gt_labels, img_metas = targets
        assert len(cls_scores) == len(bbox_preds) == len(centernesses)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        all_level_points = get_points(self,featmap_sizes, bbox_preds[0].dtype,
                                           bbox_preds[0].device)
        labels, bbox_targets = self.get_targets(all_level_points, gt_bboxes,
                                                gt_labels)

        num_imgs = cls_scores[0].size(0)
        # flatten cls_scores, bbox_preds and centerness
        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(-1, self.num_classes)
            for cls_score in cls_scores
        ]
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
            for bbox_pred in bbox_preds
        ]
        flatten_centerness = [
            centerness.permute(0, 2, 3, 1).reshape(-1)
            for centerness in centernesses
        ]
        flatten_cls_scores = torch.cat(flatten_cls_scores)
        flatten_bbox_preds = torch.cat(flatten_bbox_preds)
        flatten_centerness = torch.cat(flatten_centerness)
        flatten_labels = torch.cat(labels)
        flatten_bbox_targets = torch.cat(bbox_targets)
        # repeat points to align with bbox_preds
        flatten_points = torch.cat(
            [points.repeat(num_imgs, 1) for points in all_level_points])

        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        bg_class_ind = self.num_classes
        pos_inds = ((flatten_labels >= 0)
                    & (flatten_labels < bg_class_ind)).nonzero().reshape(-1)
        num_pos = torch.tensor(
            len(pos_inds), dtype=torch.float, device=bbox_preds[0].device)
        num_pos = max(reduce_mean(num_pos), 1.0)
        loss_cls = self.loss_cls(
            flatten_cls_scores.cpu(), flatten_labels.cpu().to(torch.int64), avg_factor=num_pos)

        pos_bbox_preds = flatten_bbox_preds[pos_inds]
        pos_centerness = flatten_centerness[pos_inds]
        pos_bbox_targets = flatten_bbox_targets[pos_inds]
        pos_centerness_targets = self.centerness_target(pos_bbox_targets)
        # centerness weighted iou loss
        centerness_denorm = max(
            reduce_mean(pos_centerness_targets.sum().detach()), 1e-6)

        if len(pos_inds) > 0:
            pos_points = flatten_points[pos_inds]
            pos_decoded_bbox_preds = distance2bbox(pos_points, pos_bbox_preds)
            pos_decoded_target_preds = distance2bbox(pos_points,
                                                     pos_bbox_targets)
            loss_bbox = self.loss_bbox(
                pos_decoded_bbox_preds,
                pos_decoded_target_preds,
                weight=pos_centerness_targets,
                avg_factor=centerness_denorm)
            loss_centerness = self.loss_centerness(
                pos_centerness, pos_centerness_targets, avg_factor=num_pos)
        else:
            loss_bbox = pos_bbox_preds.sum()
            loss_centerness = pos_centerness.sum()

        return loss_cls,loss_bbox,loss_centerness
    def centerness_target(self, pos_bbox_targets):
        """Compute centerness targets.

        Args:
            pos_bbox_targets (Tensor): BBox targets of positive bboxes in shape
                (num_pos, 4)

        Returns:
            Tensor: Centerness target.
        """
        # only calculate pos centerness targets, otherwise there may be nan
        left_right = pos_bbox_targets[:, [0, 2]]
        top_bottom = pos_bbox_targets[:, [1, 3]]
        if len(left_right) == 0:
            centerness_targets = left_right[..., 0]
        else:
            centerness_targets = (
                left_right.min(dim=-1)[0].clamp(min=0.01) / left_right.max(dim=-1)[0]) * (
                    top_bottom.min(dim=-1)[0].clamp(min=0.01) / top_bottom.max(dim=-1)[0])
        return torch.sqrt(centerness_targets)
def reduce_mean(tensor):
    """"Obtain the mean of tensor on different GPUs."""
    if not (dist.is_available() and dist.is_initialized()):
        return tensor
    tensor = tensor.clone()
    dist.all_reduce(tensor.div_(dist.get_world_size()), op=dist.ReduceOp.SUM)
    return tensor

def get_points(self, featmap_sizes, dtype, device, flatten=False):
        """Get points according to feature map sizes.

        Args:
            featmap_sizes (list[tuple]): Multi-level feature map sizes.
            dtype (torch.dtype): Type of points.
            device (torch.device): Device of points.

        Returns:
            tuple: points of each image.
        """
        mlvl_points = []
        for i in range(len(featmap_sizes)):
            mlvl_points.append(
                _get_points_single(featmap_sizes[i], self.strides[i],
                                        dtype, device, flatten))
        return mlvl_points
def gen_trf():
        '''
        Calculate the theoretical receptive field from P2-p7 of a standard ResNet-50-FPN.
        # ref: https://distill.pub/2019/computing-receptive-fields/
        '''

        j_i = [1]
        for i in range(7):
            j = j_i[i]*2
            j_i.append(j)

        r0 = 1
        r1 = r0 + (7-1)*j_i[0]
        
        r2 = r1 + (3-1)*j_i[1]
        trf_p2 = r2 + (3-1)*j_i[2]*3

        r3 = trf_p2 + (3-1)*j_i[2]
        trf_p3 = r3 + (3-1)*j_i[3]*3

        r4 = trf_p3 + (3-1)*j_i[3]
        trf_p4 = r4 + (3-1)*j_i[4]*5

        r5 = trf_p4 + (3-1)*j_i[4]
        trf_p5 = r5 + (3-1)*j_i[5]*2
 
        trf_p6 = trf_p5 + (3-1)*j_i[6]

        trf_p7 = trf_p6 + (3-1)*j_i[7]

        trfs = [trf_p2, trf_p3, trf_p4, trf_p5, trf_p6, trf_p7]

        return trfs
# def get_points_single(featmap_size,stride,dtype,device,flatten=False):
#         """Get points of a single scale level."""
#         h, w = featmap_size
#         # First create Range with the default dtype, than convert to
#         # target `dtype` for onnx exporting.
#         x_range = torch.arange(w, device=device).to(dtype)
#         y_range = torch.arange(h, device=device).to(dtype)
#         y, x = torch.meshgrid(y_range, x_range)
#         if flatten:
#             y = y.flatten()
#             x = x.flatten()
#         return y, x
def _get_points_single(
                           featmap_size,
                           stride,
                           dtype,
                           device,
                           flatten=False):
        """Get points according to feature map sizes."""
        h, w = featmap_size
        # First create Range with the default dtype, than convert to
        # target `dtype` for onnx exporting.
        x_range = torch.arange(w, device=device).to(dtype)
        y_range = torch.arange(h, device=device).to(dtype)
        y, x = torch.meshgrid(y_range, x_range)
        if flatten:
            y = y.flatten()
            x = x.flatten()
        #y, x = super().get_points_single(featmap_size, stride, dtype, device)
        points = torch.stack((x.reshape(-1) * stride, y.reshape(-1) * stride),
                             dim=-1) + stride // 2
        return points