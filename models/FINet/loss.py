import torch
import torch.nn as nn
import math
import numpy as np



def xyxy2xywh(x):
	# Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
	y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
	y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
	y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
	y[:, 2] = x[:, 2] - x[:, 0]  # width
	y[:, 3] = x[:, 3] - x[:, 1]  # height
	return y

# class get_loss(nn.Module):
#     def __init__(self,args):
#         super(get_loss,self).__init__()
#         self.args = args
#         self.model_gr = 1.0
def loss_f(p, targets, model):  # predictions, targets, model
	#targets = torch.cat((target[0],target[1]),0)
	targets = targets.cuda()
	targets.to(torch.float16)
	device = targets.device
	#targets = torch.cat((target[:,:2],xyxy2xywh(target[:,2:])),1)
	lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
	tcls, tbox, indices, anchors = build_targets(p, targets, model)  # targets
	# h = model.hyp  # hyperparameters

# Define criteria
	BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([1.0])).to(device)
	BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([1.0])).to(device)

# Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
	cp, cn = smooth_BCE(eps=0.0)

# Focal loss
	g = 0.0 # focal loss gamma = 0
	if g > 0:
		BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

# Losses
	nt = 0  # number of targets
	np = len(p)  # number of outputs
	balance = [4.0, 1.0, 0.4] if np == 3 else [4.0, 1.0, 0.4, 0.1]  # P3-5 or P3-6
	for i, pi in enumerate(p):  # layer index, layer predictions
		b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
		tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj

		n = b.shape[0]  # number of targets
		if n:
			nt += n  # cumulative targets
			ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets

			# Regression
			pxy = ps[:, :2].sigmoid() * 2. - 0.5
			pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
			pbox = torch.cat((pxy, pwh), 1).to(device)  # predicted box
			giou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, CIoU=True)  # giou(prediction, target)
			lbox += (1.0 - giou).mean()  # giou loss

			# Objectness
			tobj[b, a, gj, gi] = (1.0 - 1.0) + 1.0 * giou.detach().clamp(0).type(tobj.dtype)  # giou ratio

			# Classification
			#if self.args.num_classes > 1:  # cls loss (only if multiple classes)
			t = torch.full_like(ps[:, 5:], cn, device=device) 
			t= t.to(torch.float16) # targets
			t[range(n), tcls[i]] = cp
#			print(ps,"fjk")
			lcls += BCEcls(ps[:, 5:], t)  # BCE

			# Append targets to text file
			# with open('targets.txt', 'a') as file:
			#     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

		lobj += BCEobj(pi[..., 4], tobj) * balance[i]  # obj loss

	s = 3 / np  # output count scaling
	lbox *=0.05 * s#'giou': 0.05,  # GIoU loss gain
	lobj *= 1.0 * s * (1.4 if np == 4 else 1.)#'obj': 1.0,  # obj loss gain (scale with pixels)
	#if model.nc > 1:
	lcls *= 0.0125* s#'cls': 0.5,  # cls loss gain                         0.0125
	bs = tobj.shape[0]  # batch size

	loss = lbox + lobj + lcls
	return loss * bs, torch.cat((lbox, lobj, lcls, loss)).detach()
class FocalLoss(nn.Module):
	# Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
	def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
		super(FocalLoss, self).__init__()
		self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
		self.gamma = gamma
		self.alpha = alpha
		self.reduction = loss_fcn.reduction
		self.loss_fcn.reduction = 'none'  # required to apply FL to each element

	def forward(self, pred, true):
		loss = self.loss_fcn(pred, true)
		# p_t = torch.exp(-loss)
		# loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

		# TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
		pred_prob = torch.sigmoid(pred)  # prob from logits
		p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
		alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
		modulating_factor = (1.0 - p_t) ** self.gamma
		loss *= alpha_factor * modulating_factor

		if self.reduction == 'mean':
			return loss.mean()
		elif self.reduction == 'sum':
			return loss.sum()
		else:  # 'none'
			return loss
def is_parallel(model):
	# is model is parallel with DP or DDP
	return type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)
def build_targets(p, targets, model):
	# Build targets for compute_loss(), input targets(image,class,x,y,w,h)
	# print(type(p)) # list, 每个元素是[bs, 3, w, h, 25]  w,h是76,76或38,38或19,19
	# print(targets.size())  # [nt, 6]
	# exit()
	det = model.module.model[-1] if is_parallel(model) else model.model[-1]  # Detect() module
	na, nt = det.na, targets.shape[0]  # number of anchors, 3, targets, [nt, 6] 取nt，有nt个目标
	tcls, tbox, indices, anch = [], [], [], []
	gain = torch.ones(7, device=targets.device)  # normalized to gridspace gain
	ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
	# print(nt, ai.shape)  # (3, nt) 3行，nt列，第一行是0，第二行是1，第三行是2。如果只有一个目标就是1列，没有重复
	# print(targets.repeat(na, 1, 1).shape)  # (3, nt, 7)，相当于增加一维，并在该维度上重复3次。
	targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  # append anchor indices
	# ai[:, :, None]  ai本来是(3, nt)，变成(3, nt, 1)
	"""
	print(targets.shape) (3, nt, 7)
	3对应3个anchor，nt对应每一个目标，7的0 1元素为0，2 3 4 5分别是xywh(百分比)，最后一列是anchor的索引。
	"""

	g = 0.5  # bias
	off = torch.tensor([[0, 0],
						[1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
						# [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
						], device=targets.device).float() * g  # offsets
	# print(off.shape)  # (5, 2)

	# print(f'\nnumber of anchor: {na}, number of targets: {nt}')
	for i in range(det.nl):  # 3
		anchors = det.anchors[i]  # (3, 2)

		gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain，
		"""
		# 这里p[i].shape是[1, 3, 76, 76, 25]或者38,38  19,19，使用torch.tensor将它转换为1行5列的tensor
		# 重复取3,2,3,2是重复取出该特征层的宽和高
		# gain index [0, 1, 2, 3, 4, 5, 6]
		# gain value [1, 1, x, y, x, y, 1]
		"""

		# Match targets to anchors
		t, offsets = targets * gain, 0
		"""
		# print(t.shape)  # 广播原则，还是targets的形状(3, nt, 7)
		# 此处是吧target原本的百分比xywh转换为该特征层的实际xywh
		"""

		if nt:
			# Matches
			r = t[:, :, 4:6] / anchors[:, None]  # wh ratio
			j = torch.max(r, 1. / r).max(2)[0] < 4.0 # compare
			# j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
			t = t[j]  # filter
			# print(t.shape)  # 从(3, nt, 7)变成了(3， 7)

			# Offsets
			gxy = t[:, 2:4]  # grid xy 目标中中心点
			gxi = gain[[2, 3]] - gxy  # inverse，目标中心点相对于该特征层反转
			j, k = ((gxy % 1. < g) & (gxy > 1.)).T
			# print(j, j.shape, k, k.shape)
			l, m = ((gxi % 1. < g) & (gxi > 1.)).T
			j = torch.stack((torch.ones_like(j), j, k, l, m))
			t = t.repeat((5, 1, 1))[j]
			offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]

		# Define
		t = t.reshape(-1,7)
		b, c = t[:, :2].long().T  # image, class
		gxy = t[:, 2:4]  # grid xy
		gwh = t[:, 4:6]  # grid wh
		gij = (gxy - offsets).long()
		gi, gj = gij.T  # grid xy indices

		# Append
		a = t[:, 6].long()  # anchor indices
		indices.append((b, a, gj, gi))  # image, anchor, grid indices
		box = torch.cat((gxy - gij, gwh), 1)
		tbox.append(box)  # box
		anch.append(anchors[a])  # anchors
		tcls.append(c)  # class

		"""
		c是一维tensor，值是独热编码的分类，c的长度是n，代表该img的目标在该层可能有n个所有先验框，n是从0到多个不等。(n)
		box (n, 4)，每一行对应一个可能先验框该类的值，4列分别是：x_offset，y_offset, w, h
		indices: 索引们, list 长度是4，每个元素是一个(n,)的tensor，对应每个先验框的值，有img，anchor索引，网格索引gridj gridi
		anch: (n, 2)，每行对应一个先验框，两列分别是anchor的的宽高。
		"""

		# print('i', i, c.shape, box.shape, anchors[a].shape)
		# print('c', c)
		# print('box', box)
		# print('indices', b, a, gj, gi)
		# print('anch', anchors[a])

	return tcls, tbox, indices, anch

def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
	# return positive, negative label smoothing BCE targets
	return 1.0 - 0.5 * eps, 0.5 * eps
def bbox_iou(box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False):
	# Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
	box2 = box2.T

	# Get the coordinates of bounding boxes
	if x1y1x2y2:  # x1, y1, x2, y2 = box1
		b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
		b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
	else:  # transform from xywh to xyxy
		b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
		b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
		b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
		b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

	# Intersection area
	inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
			(torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

	# Union Area
	w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
	w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1
	union = (w1 * h1 + 1e-16) + w2 * h2 - inter

	iou = inter / union  # iou
	if GIoU or DIoU or CIoU:
		cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex (smallest enclosing box) width
		ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
		if GIoU:  # Generalized IoU https://arxiv.org/pdf/1902.09630.pdf
			c_area = cw * ch + 1e-16  # convex area
			return iou - (c_area - union) / c_area  # GIoU
		if DIoU or CIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
			# convex diagonal squared
			c2 = cw ** 2 + ch ** 2 + 1e-16
			# centerpoint distance squared
			rho2 = ((b2_x1 + b2_x2) - (b1_x1 + b1_x2)) ** 2 / 4 + ((b2_y1 + b2_y2) - (b1_y1 + b1_y2)) ** 2 / 4
			if DIoU:
				return iou - rho2 / c2  # DIoU
			elif CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
				v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
				with torch.no_grad():
					alpha = v / (1 - iou + v + 1e-16)
				return iou - (rho2 / c2 + v * alpha)  # CIoU

	return iou

