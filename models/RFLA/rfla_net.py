from .head import RFLAHead
from .rfla_neck import FPN
from .rfla_backbone import resnet50
import torch.nn as nn
from .loss import LOSS
import torch
import itertools

class RFLA(nn.Module):
    
    def __init__(self,args):
        super().__init__()
        self.fpn_out_channels = 256
        self.num_classes = args.num_classes+1
        self.cnt_on_reg=False
        self.prior=0.01
        self.use_GN_head=True
        self.pretrained=True
        self.freeze_stage_1=True
        self.freeze_bn=True
        
        
        self.backbone=resnet50(args,pretrained=True)
        self.fpn=FPN(self.fpn_out_channels,use_p5=True)
        self.head=RFLAHead(self.fpn_out_channels,self.num_classes,
                                self.use_GN_head,self.cnt_on_reg,self.prior)
        
    def train(self,mode=True):
        '''
        set module training mode, and frozen bn
        '''
        super().train(mode=True)
        def freeze_bn(module):
            if isinstance(module,nn.BatchNorm2d):
                module.eval()
            classname = module.__class__.__name__
            if classname.find('BatchNorm') != -1:
                for p in module.parameters(): p.requires_grad=False
        if self.freeze_bn:
            self.apply(freeze_bn)
            print("INFO===>success frozen BN")
        if self.freeze_stage_1:
            self.backbone.freeze_stages(1)
            print("INFO===>success frozen backbone stage1")

    def forward(self,x):
        '''
        Returns
        list [cls_logits,cnt_logits,reg_preds]  
        cls_logits  list contains five [batch_size,class_num,h,w]
        cnt_logits  list contains five [batch_size,1,h,w]
        reg_preds   list contains five [batch_size,4,h,w]
        '''
        C3,C4,C5=self.backbone(x)
        all_P=self.fpn([C3,C4,C5])
        cls_logits,reg_preds,cnt_logits=self.head(all_P)
        return [cls_logits, reg_preds, cnt_logits]

class rfla_net(nn.Module):
    
    def __init__(self,args):
        super().__init__()
        self.strides=[8,16,32,64,128]
        self.limit_range=[[-1,64],[64,128],[128,256],[256,512],[512,999999]]
        self.fcos_body=RFLA(args)
        self.target_layer=GenTargets(strides=self.strides,limit_range=self.limit_range)
        #self.loss_layer=LOSS()
    def grid_priors(self,
                        featmap_sizes,
                        dtype=torch.float32,
                        device='cuda',
                        with_stride=False):
            """Generate grid points of multiple feature levels.

            Args:
                featmap_sizes (list[tuple]): List of feature map sizes in
                    multiple feature levels, each size arrange as
                    as (h, w).
                dtype (:obj:`dtype`): Dtype of priors. Default: torch.float32.
                device (str): The device where the anchors will be put on.
                with_stride (bool): Whether to concatenate the stride to
                    the last dimension of points.

            Return:
                list[torch.Tensor]: Points of  multiple feature levels.
                The sizes of each tensor should be (N, 2) when with stride is
                ``False``, where N = width * height, width and height
                are the sizes of the corresponding feature level,
                and the last dimension 2 represent (coord_x, coord_y),
                otherwise the shape should be (N, 4),
                and the last dimension 4 represent
                (coord_x, coord_y, stride_w, stride_h).
            """

            #assert self.num_levels == len(featmap_sizes)
            self.num_levels = len(featmap_sizes)
            multi_level_priors = []
            for i in range(self.num_levels):
                priors = self.single_level_grid_priors(
                    featmap_sizes[i],
                    level_idx=i,
                    dtype=dtype,
                    device=device,
                    with_stride=with_stride)
                multi_level_priors.append(priors)
            return multi_level_priors

    def single_level_grid_priors(self,
                                 featmap_size,
                                 level_idx,
                                 dtype=torch.float32,
                                 device='cuda',
                                 with_stride=False):
        """Generate grid Points of a single level.

        Note:
            This function is usually called by method ``self.grid_priors``.

        Args:
            featmap_size (tuple[int]): Size of the feature maps, arrange as
                (h, w).
            level_idx (int): The index of corresponding feature map level.
            dtype (:obj:`dtype`): Dtype of priors. Default: torch.float32.
            device (str, optional): The device the tensor will be put on.
                Defaults to 'cuda'.
            with_stride (bool): Concatenate the stride to the last dimension
                of points.

        Return:
            Tensor: Points of single feature levels.
            The shape of tensor should be (N, 2) when with stride is
            ``False``, where N = width * height, width and height
            are the sizes of the corresponding feature level,
            and the last dimension 2 represent (coord_x, coord_y),
            otherwise the shape should be (N, 4),
            and the last dimension 4 represent
            (coord_x, coord_y, stride_w, stride_h).
        """
        self.offset = 0.5
        feat_h, feat_w = featmap_size
        self.strides = [(8, 8), (16, 16), (32, 32), (64, 64), (128, 128)]
        stride_w, stride_h = self.strides[level_idx]
        shift_x = (torch.arange(0, feat_w, device=device) +
                   self.offset) * stride_w
        # keep featmap_size as Tensor instead of int, so that we
        # can convert to ONNX correctly
        shift_x = shift_x.to(dtype)

        shift_y = (torch.arange(0, feat_h, device=device) +
                   self.offset) * stride_h
        # keep featmap_size as Tensor instead of int, so that we
        # can convert to ONNX correctly
        shift_y = shift_y.to(dtype)
        shift_xx, shift_yy = self._meshgrid(shift_x, shift_y)
        if not with_stride:
            shifts = torch.stack([shift_xx, shift_yy], dim=-1)
        else:
            # use `shape[0]` instead of `len(shift_xx)` for ONNX export
            stride_w = shift_xx.new_full((shift_xx.shape[0], ),
                                         stride_w).to(dtype)
            stride_h = shift_xx.new_full((shift_yy.shape[0], ),
                                         stride_h).to(dtype)
            shifts = torch.stack([shift_xx, shift_yy, stride_w, stride_h],
                                 dim=-1)
        all_points = shifts.to(device)
        return all_points
    def _meshgrid(self, x, y, row_major=True):
        yy, xx = torch.meshgrid(y, x)
        if row_major:
            # warning .flatten() would cause error in ONNX exporting
            # have to use reshape here
            return xx.reshape(-1), yy.reshape(-1)

        else:
            return yy.reshape(-1), xx.reshape(-1)        
    

    def forward(self,inputs):
        '''
        Returns
        list [cls_logits,cnt_logits,reg_preds]  
        cls_logits  list contains five [batch_size,class_num,h,w]
        cnt_logits  list contains five [batch_size,1,h,w]
        reg_preds   list contains five [batch_size,4,h,w]
        '''
        if len(inputs) == 3:   #train     
            batch_imgs,batch_boxes,batch_classes=inputs
            batch_boxes = batch_boxes
#            batch_classes = batch_classes.float()
            # if batch_boxes.shape[0] != 1 :
            #     batch_boxes_ =  torch.split(batch_boxes.reshape(-1,4),[1,1])
            #     batch_boxes = list(batch_boxes_)
            #     batch_classes_ =  torch.split(batch_classes.reshape(-1).float(),[1,1])
            #     batch_classes = list(batch_classes_)
            #batch_imgs,batch_boxes,batch_classes=inputs
            
            out=self.fcos_body(batch_imgs)


            cls_scores, bbox_preds,cen = out
            modified_list = []
            for index in range(len(bbox_preds)):
                bbox_preds_ = bbox_preds[index].clone()
                nan_mask = torch.isnan(bbox_preds_)
                has_nan = torch.any(nan_mask)

                if has_nan :
                    bbox_preds_[nan_mask] = 0.0
                modified_list.append(bbox_preds_)
            # print(torch.isnan(bbox_preds[0]).any())
            # print(torch.isnan(bbox_preds[0]).any())
            all_num_gt = sum([len(gt_bbox) for gt_bbox in batch_boxes])
            featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
            points = self.grid_priors(featmap_sizes, modified_list[0].dtype,
                                           modified_list[0].device)
            #targets=self.target_layer([out,batch_boxes,batch_classes])
            out_ = [cls_scores, modified_list,cen]
            targets = [batch_boxes, batch_classes, points]
            return out_, targets
        if len(inputs) == 1: #test
            batch_imgs=inputs
            out=self.fcos_body(batch_imgs)
            
            return out
    


class GenTargets(nn.Module):
    def __init__(self,strides,limit_range):
        super().__init__()
        self.strides=strides
        self.limit_range=limit_range
        assert len(strides)==len(limit_range)
    
    def forward(self,inputs):
        _,gt_bboxes,gt_labels = inputs
        cls_scores, bbox_preds,_ = inputs[0]
        all_num_gt = sum([len(gt_bbox) for gt_bbox in gt_bboxes])
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        points = self.grid_priors(featmap_sizes, bbox_preds[0].dtype,
                                           bbox_preds[0].device)
        #concat_points = torch.cat(points, dim=0)
        
        # inside_gt_bbox_mask_list=[]
        # bbox_targets_list=[]
        # #for index,P in enumerate(gt_bboxes):
        # for ind in range(len(gt_bboxes)):

        #     inside_gt_bbox_mask,bbox_targets = self._get_target_single(gt_bboxes[ind], concat_points)
        #     inside_gt_bbox_mask_list.append(inside_gt_bbox_mask)
        #     bbox_targets_list.append( bbox_targets)
            #cls_score, bbox_pred, ctn= self.forward_single(P, self.scales[index],self.strides[index])
            # inside_gt_bbox_mask_list.append(inside_gt_bbox_mask)
            
            # bbox_targets_list.append(bbox_targets)
           




        # inside_gt_bbox_mask_list, bbox_targets_list = multi_apply(
        #     self._get_target_single, gt_bboxes, points=concat_points)
        return [gt_bboxes, gt_labels,points]
    def _get_target_single(self, gt_bboxes, points):
        
        num_points = points.size(0)
        num_gts = gt_bboxes.size(0)
        gt_bboxes = gt_bboxes[None].expand(num_points, num_gts, 4)
        xs, ys = points[:, 0], points[:, 1]
        xs = xs[:, None]
        ys = ys[:, None]
        left = xs - gt_bboxes[..., 0]
        right = gt_bboxes[..., 2] - xs
        top = ys - gt_bboxes[..., 1]
        bottom = gt_bboxes[..., 3] - ys
        bbox_targets = torch.stack((left, top, right, bottom), -1)
        if num_gts:
            inside_gt_bbox_mask = bbox_targets.min(-1)[0] > 0
        else:
            inside_gt_bbox_mask = bbox_targets.new_zeros((num_points, num_gts),
                                                         dtype=torch.bool)

        return inside_gt_bbox_mask, bbox_targets
    def _gen_level_targets(self,out,gt_boxes,classes,stride,limit_range,sample_radiu_ratio=1.5):
        '''
        Args  
        out list contains [[batch_size,class_num,h,w],[batch_size,1,h,w],[batch_size,4,h,w]]  
        gt_boxes [batch_size,m,4]  
        classes [batch_size,m]  
        stride int  
        limit_range list [min,max]  
        Returns  
        cls_targets,cnt_targets,reg_targets
        '''
        cls_logits,cnt_logits,reg_preds=out
        batch_size=cls_logits.shape[0]
        class_num=cls_logits.shape[1]
        m=gt_boxes.shape[1]

        cls_logits=cls_logits.permute(0,2,3,1) #[batch_size,h,w,class_num]  
        coords=coords_fmap2orig(cls_logits,stride).to(device=gt_boxes.device)#[h*w,2]

        cls_logits=cls_logits.reshape((batch_size,-1,class_num))#[batch_size,h*w,class_num]  
        cnt_logits=cnt_logits.permute(0,2,3,1)
        cnt_logits=cnt_logits.reshape((batch_size,-1,1))
        reg_preds=reg_preds.permute(0,2,3,1)
        reg_preds=reg_preds.reshape((batch_size,-1,4))

        h_mul_w=cls_logits.shape[1]

        x=coords[:,0]
        y=coords[:,1]
        l_off=x[None,:,None]-gt_boxes[...,0][:,None,:]#[1,h*w,1]-[batch_size,1,m]-->[batch_size,h*w,m]
        t_off=y[None,:,None]-gt_boxes[...,1][:,None,:]
        r_off=gt_boxes[...,2][:,None,:]-x[None,:,None]
        b_off=gt_boxes[...,3][:,None,:]-y[None,:,None]
        ltrb_off=torch.stack([l_off,t_off,r_off,b_off],dim=-1)#[batch_size,h*w,m,4]

        areas=(ltrb_off[...,0]+ltrb_off[...,2])*(ltrb_off[...,1]+ltrb_off[...,3])#[batch_size,h*w,m]

        off_min=torch.min(ltrb_off,dim=-1)[0]#[batch_size,h*w,m]
        off_max=torch.max(ltrb_off,dim=-1)[0]#[batch_size,h*w,m]

        mask_in_gtboxes=off_min>0
        mask_in_level=(off_max>limit_range[0])&(off_max<=limit_range[1])

        radiu=stride*sample_radiu_ratio
        gt_center_x=(gt_boxes[...,0]+gt_boxes[...,2])/2
        gt_center_y=(gt_boxes[...,1]+gt_boxes[...,3])/2
        c_l_off=x[None,:,None]-gt_center_x[:,None,:]#[1,h*w,1]-[batch_size,1,m]-->[batch_size,h*w,m]
        c_t_off=y[None,:,None]-gt_center_y[:,None,:]
        c_r_off=gt_center_x[:,None,:]-x[None,:,None]
        c_b_off=gt_center_y[:,None,:]-y[None,:,None]
        c_ltrb_off=torch.stack([c_l_off,c_t_off,c_r_off,c_b_off],dim=-1)#[batch_size,h*w,m,4]
        c_off_max=torch.max(c_ltrb_off,dim=-1)[0]
        mask_center=c_off_max<radiu

        mask_pos=mask_in_gtboxes&mask_in_level&mask_center#[batch_size,h*w,m]

        areas[~mask_pos]=99999999
        areas_min_ind=torch.min(areas,dim=-1)[1]#[batch_size,h*w]
        reg_targets=ltrb_off[torch.zeros_like(areas,dtype=torch.bool).scatter_(-1,areas_min_ind.unsqueeze(dim=-1),1)]#[batch_size*h*w,4]
        reg_targets=torch.reshape(reg_targets,(batch_size,-1,4))#[batch_size,h*w,4]

        classes=torch.broadcast_tensors(classes[:,None,:],areas.long())[0]#[batch_size,h*w,m]
        cls_targets=classes[torch.zeros_like(areas,dtype=torch.bool).scatter_(-1,areas_min_ind.unsqueeze(dim=-1),1)]
        cls_targets=torch.reshape(cls_targets,(batch_size,-1,1))#[batch_size,h*w,1]

        left_right_min = torch.min(reg_targets[..., 0], reg_targets[..., 2])#[batch_size,h*w]
        left_right_max = torch.max(reg_targets[..., 0], reg_targets[..., 2])
        top_bottom_min = torch.min(reg_targets[..., 1], reg_targets[..., 3])
        top_bottom_max = torch.max(reg_targets[..., 1], reg_targets[..., 3])
        cnt_targets=((left_right_min*top_bottom_min)/(left_right_max*top_bottom_max+1e-10)).sqrt().unsqueeze(dim=-1)#[batch_size,h*w,1]

        assert reg_targets.shape==(batch_size,h_mul_w,4)
        assert cls_targets.shape==(batch_size,h_mul_w,1)
        assert cnt_targets.shape==(batch_size,h_mul_w,1)

        #process neg coords
        mask_pos_2=mask_pos.long().sum(dim=-1)#[batch_size,h*w]
        # num_pos=mask_pos_2.sum(dim=-1)
        # assert num_pos.shape==(batch_size,)
        mask_pos_2=mask_pos_2>=1
        assert mask_pos_2.shape==(batch_size,h_mul_w)
        cls_targets[~mask_pos_2]=0#[batch_size,h*w,1]
        cnt_targets[~mask_pos_2]=-1
        reg_targets[~mask_pos_2]=-1
        
        return cls_targets,cnt_targets,reg_targets

def coords_fmap2orig(feature,stride):
    '''
    transfor one fmap coords to orig coords
    Args
    featurn [batch_size,h,w,c]
    stride int
    Returns 
    coords [n,2]
    '''
    h,w=feature.shape[1:3]
    shifts_x = torch.arange(0, w * stride, stride, dtype=torch.float32)
    shifts_y = torch.arange(0, h * stride, stride, dtype=torch.float32)

    shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
    shift_x = torch.reshape(shift_x, [-1])
    shift_y = torch.reshape(shift_y, [-1])
    coords = torch.stack([shift_x, shift_y], -1) + stride // 2
    return coords


    


