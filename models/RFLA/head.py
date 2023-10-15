import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Scale
from mmcv.runner import force_fp32
import os
# from mmdet.core import distance2bbox, multi_apply, multiclass_nms, reduce_mean, build_assigner
# from ..builder import HEADS, build_loss
# from .anchor_free_head import AnchorFreeHead
from functools import partial
INF = 1e8
from mmcv.cnn import ConvModule

#@HEADS.register_module()
class RFLAHead(nn.Module):
    """Anchor-free head used in `FCOS <https://arxiv.org/abs/1904.01355>`_.

    The FCOS head does not use anchor boxes. Instead bounding boxes are
    predicted at each pixel and a centerness measure is used to suppress
    low-quality predictions.
    Here norm_on_bbox, centerness_on_reg, dcn_on_last_conv are training
    tricks used in official repo, which will bring remarkable mAP gains
    of up to 4.9. Please see https://github.com/tianzhi0549/FCOS for
    more detail.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        strides (list[int] | list[tuple[int, int]]): Strides of points
            in multiple feature levels. Default: (4, 8, 16, 32, 64).
        regress_ranges (tuple[tuple[int, int]]): Regress range of multiple
            level points.
        center_sampling (bool): If true, use center sampling. Default: False.
        center_sample_radius (float): Radius of center sampling. Default: 1.5.
        norm_on_bbox (bool): If true, normalize the regression targets
            with FPN strides. Default: False.
        centerness_on_reg (bool): If true, position centerness on the
            regress branch. Please refer to https://github.com/tianzhi0549/FCOS/issues/89#issuecomment-516877042.
            Default: False.
        conv_bias (bool | str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias of conv will be set as True if `norm_cfg` is None, otherwise
            False. Default: "auto".
        loss_cls (dict): Config of classification loss.
        loss_bbox (dict): Config of localization loss.
        loss_centerness (dict): Config of centerness loss.
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default: norm_cfg=dict(type='GN', num_groups=32, requires_grad=True).
        init_cfg (dict or list[dict], optional): Initialization config dict.

    Example:
        >>> self = FCOSHead(11, 7)
        >>> feats = [torch.rand(1, 7, s, s) for s in [4, 8, 16, 32, 64]]
        >>> cls_score, bbox_pred, centerness = self.forward(feats)
        >>> assert len(cls_score) == len(self.scales)
    """  # noqa: E501
    def __init__(self, in_channel,class_num,GN=True,cnt_on_reg=True,prior=0.01):
    # def __init__(self,
    #              num_classes,
    #              in_channels,
    #              regress_ranges=((-1, 64), (64, 128), (128, 256), (256, 512),
    #                              (512, INF)),
    #              center_sampling=False,
    #              center_sample_radius=1.5,
    #              norm_on_bbox=False,
    #              centerness_on_reg=False,
    #              fpn_layer = 'p3',
    #              fraction= 1/3,
    #              loss_cls=dict(
    #                  type='FocalLoss',
    #                  use_sigmoid=True,
    #                  gamma=2.0,
    #                  alpha=0.25,
    #                  loss_weight=1.0),
    #              loss_bbox=dict(type='IoULoss', loss_weight=1.0),
    #              loss_centerness=dict(
    #                  type='CrossEntropyLoss',
    #                  use_sigmoid=True,
    #                  loss_weight=1.0),
    #              norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
    #              init_cfg=dict(
    #                  type='Normal',
    #                  layer='Conv2d',
    #                  std=0.01,
    #                  override=dict(
    #                      type='Normal',
    #                      name='conv_cls',
    #                      std=0.01,
    #                      bias_prob=0.01)),
    #              **kwargs):
        super(RFLAHead,self).__init__()
        self.regress_ranges = ((-1, 64), (64, 128), (128, 256), (256, 512),
                                 (512, INF))
        self.cls_out_channels = class_num
        self.center_sampling =False
        self.center_sample_radius = 1.5
        self.norm_on_bbox = False
        self.in_channels = in_channel
        self.centerness_on_reg = False
        
        self.stacked_convs = 4
        
        self.feat_channels = 256
        self.norm_cfg = None
        self.dcn_on_last_conv  = False
        conv_bias='auto'
        self.conv_cfg  = None
        self.strides = (4, 8, 16, 32, 64)
        self.conv_centerness = nn.Conv2d(self.feat_channels, 1, 3, padding=1)
        self.scales = nn.ModuleList([Scale(1.0) for _ in self.strides])
        self.cls_convs = nn.ModuleList()
        assert conv_bias == 'auto' or isinstance(conv_bias, bool)
        self.conv_bias = conv_bias
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            if self.dcn_on_last_conv and i == self.stacked_convs - 1:
                conv_cfg = dict(type='DCNv2')
            else:
                conv_cfg = self.conv_cfg
            self.cls_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=self.conv_bias))
            self.reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            if self.dcn_on_last_conv and i == self.stacked_convs - 1:
                conv_cfg = dict(type='DCNv2')
            else:
                conv_cfg = self.conv_cfg
            self.reg_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=self.conv_bias))
        self.conv_cls = nn.Conv2d(
            self.feat_channels, self.cls_out_channels, 3, padding=1)
        self.conv_reg = nn.Conv2d(self.feat_channels, 4, 3, padding=1)    
        # super().__init__(
        #     num_classes,
        #     in_channels,
        #     loss_cls=loss_cls,
        #     loss_bbox=loss_bbox,
        #     norm_cfg=norm_cfg,
        #     init_cfg=init_cfg,
        #     **kwargs)
        # self.loss_centerness = build_loss(loss_centerness)
        # self.assigner = build_assigner(self.train_cfg.assigner)
    def init_layers(self):
        """Initialize layers of the head."""
        _init_cls_convs()
        _init_reg_convs()
        _init_predictor()
    def _init_layers(self):
        """Initialize layers of the head."""
        super().init_layers()
        self.conv_centerness = nn.Conv2d(self.feat_channels, 1, 3, padding=1)
        self.scales = nn.ModuleList([Scale(1.0) for _ in self.strides])

    def forward(self, feats):
        """Forward features from the upstream network.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple:
                cls_scores (list[Tensor]): Box scores for each scale level, \
                    each is a 4D-tensor, the channel number is \
                    num_points * num_classes.
                bbox_preds (list[Tensor]): Box energies / deltas for each \
                    scale level, each is a 4D-tensor, the channel number is \
                    num_points * 4.
                centernesses (list[Tensor]): centerness for each scale level, \
                    each is a 4D-tensor, the channel number is num_points * 1.
        """
        cls_logits=[]
        cnt_logits=[]
        reg_preds=[]
        for index,P in enumerate(feats):
            
            cls_score, bbox_pred, ctn= self.forward_single(P, self.scales[index],self.strides[index])
            cls_logits.append(cls_score)
            
            cnt_logits.append(ctn)
            
            reg_preds.append(bbox_pred)
        return cls_logits, reg_preds, cnt_logits

    def forward_single(self, x, scale, stride):
        """Forward features of a single scale level.

        Args:
            x (Tensor): FPN feature maps of the specified stride.
            scale (:obj: `mmcv.cnn.Scale`): Learnable scale module to resize
                the bbox prediction.
            stride (int): The corresponding stride for feature maps, only
                used to normalize the bbox prediction when self.norm_on_bbox
                is True.

        Returns:
            tuple: scores for each class, bbox predictions and centerness \
                predictions of input feature maps.
        """
        cls_feat = x
        reg_feat = x

        for cls_layer in self.cls_convs:
            cls_feat = cls_layer(cls_feat)
        cls_score = self.conv_cls(cls_feat)

        for reg_layer in self.reg_convs:
            reg_feat = reg_layer(reg_feat)
        bbox_pred = self.conv_reg(reg_feat)
        if self.centerness_on_reg:
            centerness = self.conv_centerness(reg_feat)
        else:
            centerness = self.conv_centerness(cls_feat)
        # scale the bbox_pred of different level
        # float to avoid overflow when enabling FP16
        bbox_pred = scale(bbox_pred).float()
        if self.norm_on_bbox:
            bbox_pred = F.relu(bbox_pred)
            if not self.training:
                bbox_pred *= stride
        else:
            bbox_pred = bbox_pred.exp()
        return cls_score, bbox_pred, centerness

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'centernesses'))
    # def loss(self,
    #          cls_scores,
    #          bbox_preds,
    #          centernesses,
    #          gt_bboxes,
    #          gt_labels,
    #          img_metas,
    #          gt_bboxes_ignore=None):
    #     """Compute loss of the head.

    #     Args:
    #         cls_scores (list[Tensor]): Box scores for each scale level,
    #             each is a 4D-tensor, the channel number is
    #             num_points * num_classes.
    #         bbox_preds (list[Tensor]): Box energies / deltas for each scale
    #             level, each is a 4D-tensor, the channel number is
    #             num_points * 4.
    #         centernesses (list[Tensor]): centerness for each scale level, each
    #             is a 4D-tensor, the channel number is num_points * 1.
    #         gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
    #             shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
    #         gt_labels (list[Tensor]): class indices corresponding to each box
    #         img_metas (list[dict]): Meta information of each image, e.g.,
    #             image size, scaling factor, etc.
    #         gt_bboxes_ignore (None | list[Tensor]): specify which bounding
    #             boxes can be ignored when computing the loss.

    #     Returns:
    #         dict[str, Tensor]: A dictionary of loss components.
    #     """
    #     assert len(cls_scores) == len(bbox_preds) == len(centernesses)
    #     featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
    #     all_level_points = self.get_points(featmap_sizes, bbox_preds[0].dtype,
    #                                        bbox_preds[0].device)
    #     labels, bbox_targets = self.get_targets(all_level_points, gt_bboxes,
    #                                             gt_labels)

    #     num_imgs = cls_scores[0].size(0)
    #     # flatten cls_scores, bbox_preds and centerness
    #     flatten_cls_scores = [
    #         cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
    #         for cls_score in cls_scores
    #     ]
    #     flatten_bbox_preds = [
    #         bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
    #         for bbox_pred in bbox_preds
    #     ]
    #     flatten_centerness = [
    #         centerness.permute(0, 2, 3, 1).reshape(-1)
    #         for centerness in centernesses
    #     ]
    #     flatten_cls_scores = torch.cat(flatten_cls_scores)
    #     flatten_bbox_preds = torch.cat(flatten_bbox_preds)
    #     flatten_centerness = torch.cat(flatten_centerness)
    #     flatten_labels = torch.cat(labels)
    #     flatten_bbox_targets = torch.cat(bbox_targets)
    #     # repeat points to align with bbox_preds
    #     flatten_points = torch.cat(
    #         [points.repeat(num_imgs, 1) for points in all_level_points])

    #     # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
    #     bg_class_ind = self.num_classes
    #     pos_inds = ((flatten_labels >= 0)
    #                 & (flatten_labels < bg_class_ind)).nonzero().reshape(-1)
    #     num_pos = torch.tensor(
    #         len(pos_inds), dtype=torch.float, device=bbox_preds[0].device)
    #     num_pos = max(reduce_mean(num_pos), 1.0)
    #     loss_cls = self.loss_cls(
    #         flatten_cls_scores, flatten_labels, avg_factor=num_pos)

    #     pos_bbox_preds = flatten_bbox_preds[pos_inds]
    #     pos_centerness = flatten_centerness[pos_inds]
    #     pos_bbox_targets = flatten_bbox_targets[pos_inds]
    #     pos_centerness_targets = self.centerness_target(pos_bbox_targets)
    #     # centerness weighted iou loss
    #     centerness_denorm = max(
    #         reduce_mean(pos_centerness_targets.sum().detach()), 1e-6)

    #     if len(pos_inds) > 0:
    #         pos_points = flatten_points[pos_inds]
    #         pos_decoded_bbox_preds = distance2bbox(pos_points, pos_bbox_preds)
    #         pos_decoded_target_preds = distance2bbox(pos_points,
    #                                                  pos_bbox_targets)
    #         loss_bbox = self.loss_bbox(
    #             pos_decoded_bbox_preds,
    #             pos_decoded_target_preds,
    #             weight=pos_centerness_targets,
    #             avg_factor=centerness_denorm)
    #         loss_centerness = self.loss_centerness(
    #             pos_centerness, pos_centerness_targets, avg_factor=num_pos)
    #     else:
    #         loss_bbox = pos_bbox_preds.sum()
    #         loss_centerness = pos_centerness.sum()

    #     return dict(
    #         loss_cls=loss_cls,
    #         loss_bbox=loss_bbox,
    #         loss_centerness=loss_centerness)

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'centernesses'))
    def get_bboxes(self,
                   cls_scores,
                   bbox_preds,
                   centernesses,
                   img_metas,
                   cfg=None,
                   rescale=False,
                   with_nms=True):
        """Transform network output for a batch into bbox predictions.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                with shape (N, num_points * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_points * 4, H, W).
            centernesses (list[Tensor]): Centerness for each scale level with
                shape (N, num_points * 1, H, W).
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            cfg (mmcv.Config | None): Test / postprocessing configuration,
                if None, test_cfg would be used. Default: None.
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before return boxes.
                Default: True.

        Returns:
            list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is an (n, 5) tensor, where 5 represent
                (tl_x, tl_y, br_x, br_y, score) and the score between 0 and 1.
                The shape of the second tensor in the tuple is (n,), and
                each element represents the class label of the corresponding
                box.
        """
        assert len(cls_scores) == len(bbox_preds)
        num_levels = len(cls_scores)

        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        mlvl_points = self.get_points(featmap_sizes, bbox_preds[0].dtype,
                                      bbox_preds[0].device)

        cls_score_list = [cls_scores[i].detach() for i in range(num_levels)]
        bbox_pred_list = [bbox_preds[i].detach() for i in range(num_levels)]
        centerness_pred_list = [
            centernesses[i].detach() for i in range(num_levels)
        ]
        if torch.onnx.is_in_onnx_export():
            assert len(
                img_metas
            ) == 1, 'Only support one input image while in exporting to ONNX'
            img_shapes = img_metas[0]['img_shape_for_onnx']
        else:
            img_shapes = [
                img_metas[i]['img_shape']
                for i in range(cls_scores[0].shape[0])
            ]
        scale_factors = [
            img_metas[i]['scale_factor'] for i in range(cls_scores[0].shape[0])
        ]
        result_list = self._get_bboxes(cls_score_list, bbox_pred_list,
                                       centerness_pred_list, mlvl_points,
                                       img_shapes, scale_factors, cfg, rescale,
                                       with_nms)
        return result_list

    def _get_bboxes(self,
                    cls_scores,
                    bbox_preds,
                    centernesses,
                    mlvl_points,
                    img_shapes,
                    scale_factors,
                    cfg,
                    rescale=False,
                    with_nms=True):
        """Transform outputs for a single batch item into bbox predictions.

        Args:
            cls_scores (list[Tensor]): Box scores for a single scale level
                with shape (N, num_points * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for a single scale
                level with shape (N, num_points * 4, H, W).
            centernesses (list[Tensor]): Centerness for a single scale level
                with shape (N, num_points, H, W).
            mlvl_points (list[Tensor]): Box reference for a single scale level
                with shape (num_total_points, 4).
            img_shapes (list[tuple[int]]): Shape of the input image,
                list[(height, width, 3)].
            scale_factors (list[ndarray]): Scale factor of the image arrange as
                (w_scale, h_scale, w_scale, h_scale).
            cfg (mmcv.Config | None): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before return boxes.
                Default: True.

        Returns:
            tuple(Tensor):
                det_bboxes (Tensor): BBox predictions in shape (n, 5), where
                    the first 4 columns are bounding box positions
                    (tl_x, tl_y, br_x, br_y) and the 5-th column is a score
                    between 0 and 1.
                det_labels (Tensor): A (n,) tensor where each item is the
                    predicted class label of the corresponding box.
        """
        cfg = self.test_cfg if cfg is None else cfg
        assert len(cls_scores) == len(bbox_preds) == len(mlvl_points)
        device = cls_scores[0].device
        batch_size = cls_scores[0].shape[0]
        # convert to tensor to keep tracing
        nms_pre_tensor = torch.tensor(
            cfg.get('nms_pre', -1), device=device, dtype=torch.long)
        mlvl_bboxes = []
        mlvl_scores = []
        mlvl_centerness = []
        for cls_score, bbox_pred, centerness, points in zip(
                cls_scores, bbox_preds, centernesses, mlvl_points):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            scores = cls_score.permute(0, 2, 3, 1).reshape(
                batch_size, -1, self.cls_out_channels).sigmoid()
            centerness = centerness.permute(0, 2, 3,
                                            1).reshape(batch_size,
                                                       -1).sigmoid()

            bbox_pred = bbox_pred.permute(0, 2, 3,
                                          1).reshape(batch_size, -1, 4)
            points = points.expand(batch_size, -1, 2)
            # Get top-k prediction
            
            nms_pre = get_k_for_topk(nms_pre_tensor, bbox_pred.shape[1])
            if nms_pre > 0:
                max_scores, _ = (scores * centerness[..., None]).max(-1)
                _, topk_inds = max_scores.topk(nms_pre)
                batch_inds = torch.arange(batch_size).view(
                    -1, 1).expand_as(topk_inds).long()
                # Avoid onnx2tensorrt issue in https://github.com/NVIDIA/TensorRT/issues/1134 # noqa: E501
                if torch.onnx.is_in_onnx_export():
                    transformed_inds = bbox_pred.shape[
                        1] * batch_inds + topk_inds
                    points = points.reshape(-1,
                                            2)[transformed_inds, :].reshape(
                                                batch_size, -1, 2)
                    bbox_pred = bbox_pred.reshape(
                        -1, 4)[transformed_inds, :].reshape(batch_size, -1, 4)
                    scores = scores.reshape(
                        -1, self.num_classes)[transformed_inds, :].reshape(
                            batch_size, -1, self.num_classes)
                    centerness = centerness.reshape(
                        -1, 1)[transformed_inds].reshape(batch_size, -1)
                else:
                    points = points[batch_inds, topk_inds, :]
                    bbox_pred = bbox_pred[batch_inds, topk_inds, :]
                    scores = scores[batch_inds, topk_inds, :]
                    centerness = centerness[batch_inds, topk_inds]

            bboxes = distance2bbox(points, bbox_pred, max_shape=img_shapes)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
            mlvl_centerness.append(centerness)

        batch_mlvl_bboxes = torch.cat(mlvl_bboxes, dim=1)
        if rescale:
            batch_mlvl_bboxes /= batch_mlvl_bboxes.new_tensor(
                scale_factors).unsqueeze(1)
        batch_mlvl_scores = torch.cat(mlvl_scores, dim=1)
        batch_mlvl_centerness = torch.cat(mlvl_centerness, dim=1)

        # Replace multiclass_nms with ONNX::NonMaxSuppression in deployment
        if torch.onnx.is_in_onnx_export() and with_nms:
            #from mmdet.core.export import add_dummy_nms_for_onnx
            batch_mlvl_scores = batch_mlvl_scores * (
                batch_mlvl_centerness.unsqueeze(2))
            max_output_boxes_per_class = cfg.nms.get(
                'max_output_boxes_per_class', 200)
            iou_threshold = cfg.nms.get('iou_threshold', 0.5)
            score_threshold = cfg.score_thr
            nms_pre = cfg.get('deploy_nms_pre', -1)
            return add_dummy_nms_for_onnx(batch_mlvl_bboxes, batch_mlvl_scores,
                                          max_output_boxes_per_class,
                                          iou_threshold, score_threshold,
                                          nms_pre, cfg.max_per_img)
        # remind that we set FG labels to [0, num_class-1] since mmdet v2.0
        # BG cat_id: num_class
        padding = batch_mlvl_scores.new_zeros(batch_size,
                                              batch_mlvl_scores.shape[1], 1)
        batch_mlvl_scores = torch.cat([batch_mlvl_scores, padding], dim=-1)

        if with_nms:
            det_results = []
            for (mlvl_bboxes, mlvl_scores,
                 mlvl_centerness) in zip(batch_mlvl_bboxes, batch_mlvl_scores,
                                         batch_mlvl_centerness):
                det_bbox, det_label = multiclass_nms(
                    mlvl_bboxes,
                    mlvl_scores,
                    cfg.score_thr,
                    cfg.nms,
                    cfg.max_per_img,
                    score_factors=mlvl_centerness)
                det_results.append(tuple([det_bbox, det_label]))
        else:
            det_results = [
                tuple(mlvl_bs)
                for mlvl_bs in zip(batch_mlvl_bboxes, batch_mlvl_scores,
                                   batch_mlvl_centerness)
            ]
        return det_results

    def _get_points_single(self,
                           featmap_size,
                           stride,
                           dtype,
                           device,
                           flatten=False):
        """Get points according to feature map sizes."""
        y, x = super()._get_points_single(featmap_size, stride, dtype, device)
        points = torch.stack((x.reshape(-1) * stride, y.reshape(-1) * stride),
                             dim=-1) + stride // 2
        return points

    
    

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

   
def get_k_for_topk(k, size):
    """Get k of TopK for onnx exporting.

    The K of TopK in TensorRT should not be a Tensor, while in ONNX Runtime
      it could be a Tensor.Due to dynamic shape feature, we have to decide
      whether to do TopK and what K it should be while exporting to ONNX.
    If returned K is less than zero, it means we do not have to do
      TopK operation.

    Args:
        k (int or Tensor): The set k value for nms from config file.
        size (Tensor or torch.Size): The number of elements of \
            TopK's input tensor
    Returns:
        tuple: (int or Tensor): The final K for TopK.
    """
    ret_k = -1
    if k <= 0 or size <= 0:
        return ret_k
    if torch.onnx.is_in_onnx_export():
        is_trt_backend = os.environ.get('ONNX_BACKEND') == 'MMCVTensorRT'
        if is_trt_backend:
            # TensorRT does not support dynamic K with TopK op
            if 0 < k < size:
                ret_k = k
        else:
            # Always keep topk op for dynamic input in onnx for ONNX Runtime
            ret_k = torch.where(k < size, k, size)
    elif k < size:
        ret_k = k
    else:
        # ret_k is -1
        pass
    return ret_k
def add_dummy_nms_for_onnx(boxes,
                           scores,
                           max_output_boxes_per_class=1000,
                           iou_threshold=0.5,
                           score_threshold=0.05,
                           pre_top_k=-1,
                           after_top_k=-1,
                           labels=None):
    """Create a dummy onnx::NonMaxSuppression op while exporting to ONNX.

    This function helps exporting to onnx with batch and multiclass NMS op.
    It only supports class-agnostic detection results. That is, the scores
    is of shape (N, num_bboxes, num_classes) and the boxes is of shape
    (N, num_boxes, 4).

    Args:
        boxes (Tensor): The bounding boxes of shape [N, num_boxes, 4]
        scores (Tensor): The detection scores of shape
            [N, num_boxes, num_classes]
        max_output_boxes_per_class (int): Maximum number of output
            boxes per class of nms. Defaults to 1000.
        iou_threshold (float): IOU threshold of nms. Defaults to 0.5
        score_threshold (float): score threshold of nms.
            Defaults to 0.05.
        pre_top_k (bool): Number of top K boxes to keep before nms.
            Defaults to -1.
        after_top_k (int): Number of top K boxes to keep after nms.
            Defaults to -1.
        labels (Tensor, optional): It not None, explicit labels would be used.
            Otherwise, labels would be automatically generated using
            num_classed. Defaults to None.

    Returns:
        tuple[Tensor, Tensor]: dets of shape [N, num_det, 5] and class labels
            of shape [N, num_det].
    """
    max_output_boxes_per_class = torch.LongTensor([max_output_boxes_per_class])
    iou_threshold = torch.tensor([iou_threshold], dtype=torch.float32)
    score_threshold = torch.tensor([score_threshold], dtype=torch.float32)
    batch_size = scores.shape[0]
    num_class = scores.shape[2]

    nms_pre = torch.tensor(pre_top_k, device=scores.device, dtype=torch.long)
    nms_pre = get_k_for_topk(nms_pre, boxes.shape[1])

    if nms_pre > 0:
        max_scores, _ = scores.max(-1)
        _, topk_inds = max_scores.topk(nms_pre)
        batch_inds = torch.arange(batch_size).view(
            -1, 1).expand_as(topk_inds).long()
        # Avoid onnx2tensorrt issue in https://github.com/NVIDIA/TensorRT/issues/1134 # noqa: E501
        transformed_inds = boxes.shape[1] * batch_inds + topk_inds
        boxes = boxes.reshape(-1, 4)[transformed_inds, :].reshape(
            batch_size, -1, 4)
        scores = scores.reshape(-1, num_class)[transformed_inds, :].reshape(
            batch_size, -1, num_class)
        if labels is not None:
            labels = labels.reshape(-1, 1)[transformed_inds].reshape(
                batch_size, -1)

    scores = scores.permute(0, 2, 1)
    num_box = boxes.shape[1]
    # turn off tracing to create a dummy output of nms
    state = torch._C._get_tracing_state()
    # dummy indices of nms's output
    num_fake_det = 2
    batch_inds = torch.randint(batch_size, (num_fake_det, 1))
    cls_inds = torch.randint(num_class, (num_fake_det, 1))
    box_inds = torch.randint(num_box, (num_fake_det, 1))
    indices = torch.cat([batch_inds, cls_inds, box_inds], dim=1)
    output = indices
    setattr(DummyONNXNMSop, 'output', output)

    # open tracing
    torch._C._set_tracing_state(state)
    selected_indices = DummyONNXNMSop.apply(boxes, scores,
                                            max_output_boxes_per_class,
                                            iou_threshold, score_threshold)

    batch_inds, cls_inds = selected_indices[:, 0], selected_indices[:, 1]
    box_inds = selected_indices[:, 2]
    if labels is None:
        labels = torch.arange(num_class, dtype=torch.long).to(scores.device)
        labels = labels.view(1, num_class, 1).expand_as(scores)
    scores = scores.reshape(-1, 1)
    boxes = boxes.reshape(batch_size, -1).repeat(1, num_class).reshape(-1, 4)
    pos_inds = (num_class * batch_inds + cls_inds) * num_box + box_inds
    mask = scores.new_zeros(scores.shape)
    # Avoid onnx2tensorrt issue in https://github.com/NVIDIA/TensorRT/issues/1134 # noqa: E501
    # PyTorch style code: mask[batch_inds, box_inds] += 1
    mask[pos_inds, :] += 1
    scores = scores * mask
    boxes = boxes * mask

    scores = scores.reshape(batch_size, -1)
    boxes = boxes.reshape(batch_size, -1, 4)
    labels = labels.reshape(batch_size, -1)

    nms_after = torch.tensor(
        after_top_k, device=scores.device, dtype=torch.long)
    nms_after = get_k_for_topk(nms_after, num_box * num_class)

    if nms_after > 0:
        _, topk_inds = scores.topk(nms_after)
        batch_inds = torch.arange(batch_size).view(-1, 1).expand_as(topk_inds)
        # Avoid onnx2tensorrt issue in https://github.com/NVIDIA/TensorRT/issues/1134 # noqa: E501
        transformed_inds = scores.shape[1] * batch_inds + topk_inds
        scores = scores.reshape(-1, 1)[transformed_inds, :].reshape(
            batch_size, -1)
        boxes = boxes.reshape(-1, 4)[transformed_inds, :].reshape(
            batch_size, -1, 4)
        labels = labels.reshape(-1, 1)[transformed_inds, :].reshape(
            batch_size, -1)

    scores = scores.unsqueeze(2)
    dets = torch.cat([boxes, scores], dim=2)
    return dets, labels
class DummyONNXNMSop(torch.autograd.Function):
    """DummyONNXNMSop.

    This class is only for creating onnx::NonMaxSuppression.
    """

    @staticmethod
    def forward(ctx, boxes, scores, max_output_boxes_per_class, iou_threshold,
                score_threshold):

        return DummyONNXNMSop.output

    @staticmethod
    def symbolic(g, boxes, scores, max_output_boxes_per_class, iou_threshold,
                 score_threshold):
        return g.op(
            'NonMaxSuppression',
            boxes,
            scores,
            max_output_boxes_per_class,
            iou_threshold,
            score_threshold,
            outputs=1)
def distance2bbox(points, distance, max_shape=None):
    """Decode distance prediction to bounding box.

    Args:
        points (Tensor): Shape (B, N, 2) or (N, 2).
        distance (Tensor): Distance from the given point to 4
            boundaries (left, top, right, bottom). Shape (B, N, 4) or (N, 4)
        max_shape (Sequence[int] or torch.Tensor or Sequence[
            Sequence[int]],optional): Maximum bounds for boxes, specifies
            (H, W, C) or (H, W). If priors shape is (B, N, 4), then
            the max_shape should be a Sequence[Sequence[int]]
            and the length of max_shape should also be B.

    Returns:
        Tensor: Boxes with shape (N, 4) or (B, N, 4)
    """
    x1 = points[..., 0] - distance[..., 0]
    y1 = points[..., 1] - distance[..., 1]
    x2 = points[..., 0] + distance[..., 2]
    y2 = points[..., 1] + distance[..., 3]

    bboxes = torch.stack([x1, y1, x2, y2], -1)

    if max_shape is not None:
        # clip bboxes with dynamic `min` and `max` for onnx
        if torch.onnx.is_in_onnx_export():
            #from mmdet.core.export import dynamic_clip_for_onnx
            x1, y1, x2, y2 = dynamic_clip_for_onnx(x1, y1, x2, y2, max_shape)
            bboxes = torch.stack([x1, y1, x2, y2], dim=-1)
            return bboxes
        if not isinstance(max_shape, torch.Tensor):
            max_shape = x1.new_tensor(max_shape)
        max_shape = max_shape[..., :2].type_as(x1)
        if max_shape.ndim == 2:
            assert bboxes.ndim == 3
            assert max_shape.size(0) == bboxes.size(0)

        min_xy = x1.new_tensor(0)
        max_xy = torch.cat([max_shape, max_shape],
                           dim=-1).flip(-1).unsqueeze(-2)
        bboxes = torch.where(bboxes < min_xy, min_xy, bboxes)
        bboxes = torch.where(bboxes > max_xy, max_xy, bboxes)

    return bboxes
def dynamic_clip_for_onnx(x1, y1, x2, y2, max_shape):
    """Clip boxes dynamically for onnx.

    Since torch.clamp cannot have dynamic `min` and `max`, we scale the
      boxes by 1/max_shape and clamp in the range [0, 1].

    Args:
        x1 (Tensor): The x1 for bounding boxes.
        y1 (Tensor): The y1 for bounding boxes.
        x2 (Tensor): The x2 for bounding boxes.
        y2 (Tensor): The y2 for bounding boxes.
        max_shape (Tensor or torch.Size): The (H,W) of original image.
    Returns:
        tuple(Tensor): The clipped x1, y1, x2, y2.
    """
    assert isinstance(
        max_shape,
        torch.Tensor), '`max_shape` should be tensor of (h,w) for onnx'

    # scale by 1/max_shape
    x1 = x1 / max_shape[1]
    y1 = y1 / max_shape[0]
    x2 = x2 / max_shape[1]
    y2 = y2 / max_shape[0]

    # clamp [0, 1]
    x1 = torch.clamp(x1, 0, 1)
    y1 = torch.clamp(y1, 0, 1)
    x2 = torch.clamp(x2, 0, 1)
    y2 = torch.clamp(y2, 0, 1)

    # scale back
    x1 = x1 * max_shape[1]
    y1 = y1 * max_shape[0]
    x2 = x2 * max_shape[1]
    y2 = y2 * max_shape[0]
    return x1, y1, x2, y2
def multiclass_nms(multi_bboxes,
                   multi_scores,
                   score_thr,
                   nms_cfg,
                   max_num=-1,
                   score_factors=None,
                   return_inds=False):
    """NMS for multi-class bboxes.

    Args:
        multi_bboxes (Tensor): shape (n, #class*4) or (n, 4)
        multi_scores (Tensor): shape (n, #class), where the last column
            contains scores of the background class, but this will be ignored.
        score_thr (float): bbox threshold, bboxes with scores lower than it
            will not be considered.
        nms_thr (float): NMS IoU threshold
        max_num (int, optional): if there are more than max_num bboxes after
            NMS, only top max_num will be kept. Default to -1.
        score_factors (Tensor, optional): The factors multiplied to scores
            before applying NMS. Default to None.
        return_inds (bool, optional): Whether return the indices of kept
            bboxes. Default to False.

    Returns:
        tuple: (dets, labels, indices (optional)), tensors of shape (k, 5),
            (k), and (k). Dets are boxes with scores. Labels are 0-based.
    """
    num_classes = multi_scores.size(1) - 1
    # exclude background category
    if multi_bboxes.shape[1] > 4:
        bboxes = multi_bboxes.view(multi_scores.size(0), -1, 4)
    else:
        bboxes = multi_bboxes[:, None].expand(
            multi_scores.size(0), num_classes, 4)

    scores = multi_scores[:, :-1]

    labels = torch.arange(num_classes, dtype=torch.long)
    labels = labels.view(1, -1).expand_as(scores)

    bboxes = bboxes.reshape(-1, 4)
    scores = scores.reshape(-1)
    labels = labels.reshape(-1)

    if not torch.onnx.is_in_onnx_export():
        # NonZero not supported  in TensorRT
        # remove low scoring boxes
        valid_mask = scores > score_thr
    # multiply score_factor after threshold to preserve more bboxes, improve
    # mAP by 1% for YOLOv3
    if score_factors is not None:
        # expand the shape to match original shape of score
        score_factors = score_factors.view(-1, 1).expand(
            multi_scores.size(0), num_classes)
        score_factors = score_factors.reshape(-1)
        scores = scores * score_factors

    if not torch.onnx.is_in_onnx_export():
        # NonZero not supported  in TensorRT
        inds = valid_mask.nonzero(as_tuple=False).squeeze(1)
        bboxes, scores, labels = bboxes[inds], scores[inds], labels[inds]
    else:
        # TensorRT NMS plugin has invalid output filled with -1
        # add dummy data to make detection output correct.
        bboxes = torch.cat([bboxes, bboxes.new_zeros(1, 4)], dim=0)
        scores = torch.cat([scores, scores.new_zeros(1)], dim=0)
        labels = torch.cat([labels, labels.new_zeros(1)], dim=0)

    if bboxes.numel() == 0:
        if torch.onnx.is_in_onnx_export():
            raise RuntimeError('[ONNX Error] Can not record NMS '
                               'as it has not been executed this time')
        dets = torch.cat([bboxes, scores[:, None]], -1)
        if return_inds:
            return dets, labels, inds
        else:
            return dets, labels

    dets, keep = batched_nms(bboxes, scores, labels, nms_cfg)

    if max_num > 0:
        dets = dets[:max_num]
        keep = keep[:max_num]

    if return_inds:
        return dets, labels[keep], keep
    else:
        return dets, labels[keep]
def batched_nms(boxes, scores, idxs, nms_cfg, class_agnostic=False):
    """Performs non-maximum suppression in a batched fashion.

    Modified from https://github.com/pytorch/vision/blob
    /505cd6957711af790211896d32b40291bea1bc21/torchvision/ops/boxes.py#L39.
    In order to perform NMS independently per class, we add an offset to all
    the boxes. The offset is dependent only on the class idx, and is large
    enough so that boxes from different classes do not overlap.

    Arguments:
        boxes (torch.Tensor): boxes in shape (N, 4).
        scores (torch.Tensor): scores in shape (N, ).
        idxs (torch.Tensor): each index value correspond to a bbox cluster,
            and NMS will not be applied between elements of different idxs,
            shape (N, ).
        nms_cfg (dict): specify nms type and other parameters like iou_thr.
            Possible keys includes the following.

            - iou_thr (float): IoU threshold used for NMS.
            - split_thr (float): threshold number of boxes. In some cases the
                number of boxes is large (e.g., 200k). To avoid OOM during
                training, the users could set `split_thr` to a small value.
                If the number of boxes is greater than the threshold, it will
                perform NMS on each group of boxes separately and sequentially.
                Defaults to 10000.
        class_agnostic (bool): if true, nms is class agnostic,
            i.e. IoU thresholding happens over all boxes,
            regardless of the predicted class.

    Returns:
        tuple: kept dets and indice.
    """
    nms_cfg_ = nms_cfg.copy()
    class_agnostic = nms_cfg_.pop('class_agnostic', class_agnostic)
    if class_agnostic:
        boxes_for_nms = boxes
    else:
        max_coordinate = boxes.max()
        offsets = idxs.to(boxes) * (max_coordinate + torch.tensor(1).to(boxes))
        boxes_for_nms = boxes + offsets[:, None]

    nms_type = nms_cfg_.pop('type', 'nms')
    nms_op = eval(nms_type)

    split_thr = nms_cfg_.pop('split_thr', 10000)
    if boxes_for_nms.shape[0] < split_thr:
        dets, keep = nms_op(boxes_for_nms, scores, **nms_cfg_)
        boxes = boxes[keep]
        scores = dets[:, -1]
    else:
        total_mask = scores.new_zeros(scores.size(), dtype=torch.bool)
        for id in torch.unique(idxs):
            mask = (idxs == id).nonzero(as_tuple=False).view(-1)
            dets, keep = nms_op(boxes_for_nms[mask], scores[mask], **nms_cfg_)
            total_mask[mask[keep]] = True

        keep = total_mask.nonzero(as_tuple=False).view(-1)
        keep = keep[scores[keep].argsort(descending=True)]
        boxes = boxes[keep]
        scores = scores[keep]

    return torch.cat([boxes, scores[:, None]], -1), keep
def multi_apply(func, *args, **kwargs):
    """Apply function to a list of arguments.

    Note:
        This function applies the ``func`` to multiple inputs and
        map the multiple outputs of the ``func`` into different
        list. Each list contains the same type of outputs corresponding
        to different inputs.

    Args:
        func (Function): A function that will be applied to a list of
            arguments

    Returns:
        tuple(list): A tuple containing multiple list, each list contains \
            a kind of returned results by the function
    """
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
    return tuple(map(list, zip(*map_results)))
def _init_cls_convs(self):
        """Initialize classification conv layers of the head."""
        self.cls_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            if self.dcn_on_last_conv and i == self.stacked_convs - 1:
                conv_cfg = dict(type='DCNv2')
            else:
                conv_cfg = self.conv_cfg
            self.cls_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=self.conv_bias))

def _init_reg_convs(self):
    """Initialize bbox regression conv layers of the head."""
    self.reg_convs = nn.ModuleList()
    for i in range(self.stacked_convs):
        chn = self.in_channels if i == 0 else self.feat_channels
        if self.dcn_on_last_conv and i == self.stacked_convs - 1:
            conv_cfg = dict(type='DCNv2')
        else:
            conv_cfg = self.conv_cfg
        self.reg_convs.append(
            ConvModule(
                chn,
                self.feat_channels,
                3,
                stride=1,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=self.norm_cfg,
                bias=self.conv_bias))

def _init_predictor(self):
    """Initialize predictor layers of the head."""
    self.conv_cls = nn.Conv2d(
        self.feat_channels, self.cls_out_channels, 3, padding=1)
    self.conv_reg = nn.Conv2d(self.feat_channels, 4, 3, padding=1)
