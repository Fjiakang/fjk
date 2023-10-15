import cv2
import torch
import numpy as np
from libs.utils.images import get_affine_transform
from datasets.VOC import VHR_means, VHR_std, SSDD_means, SSDD_std
import cv2
import math
import torch
import numpy as np
from libs.utils.images import color_aug, draw_umich_gaussian
from libs.utils.centernet_aug import affine_transform, gaussian_radius, get_affine_transform


class Model_train_transform:
    def __init__(self, args, means, std):
        self.args = args
        self._data_rng = np.random.RandomState(123)
        self._eig_val = np.array([0.2141788, 0.01817699, 0.00341571], dtype=np.float32)
        self._eig_vec = np.array([[-0.58752847, -0.69563484, 0.41340352], [-0.5832747, 0.00994535, -0.81221408], [-0.56089297, 0.71832671, 0.41158938]], dtype=np.float32)
        self.means = means
        self.std = std
        self.max_objs = 30  # 每张图中可能存在的最多的object

    def _get_border(self, border, size):
        i = 1
        while size - border // i <= border // i:
            i *= 2
        return border // i

    def __call__(self, img, bboxs, label):
        height, width = img.shape[0], img.shape[1]
        c = np.array([img.shape[1] / 2.0, img.shape[0] / 2.0], dtype=np.float32)  # 整张图的center
        if self.args.keep_res:
            input_h = (height | self.args.pad) + 1
            input_w = (width | self.args.pad) + 1
            s = np.array([input_w, input_h], dtype=np.float32)
        else:
            s = max(img.shape[0], img.shape[1]) * 1.0
            input_h, input_w = self.args.inp_imgsize, self.args.inp_imgsize

        flipped = False
        if self.args.rand_crop:
            s = s * np.random.choice(np.arange(0.6, 1.4, 0.1))
            w_border = self._get_border(128, img.shape[1])
            h_border = self._get_border(128, img.shape[0])
            c[0] = np.random.randint(low=w_border, high=img.shape[1] - w_border)
            c[1] = np.random.randint(low=h_border, high=img.shape[0] - h_border)
        else:
            sf = self.args.scale
            cf = self.args.shift
            c[0] += s * np.clip(np.random.randn() * cf, -2 * cf, 2 * cf)
            c[1] += s * np.clip(np.random.randn() * cf, -2 * cf, 2 * cf)
            s = s * np.clip(np.random.randn() * sf + 1, 1 - sf, 1 + sf)

        if np.random.randn() < self.args.flip:
            flipped = True
            img = img[:, ::-1, :]
            c[0] = width - c[0] - 1

        trans_input = get_affine_transform(c, s, 0, [input_w, input_h])
        inp = cv2.warpAffine(img, trans_input, (input_w, input_h), flags=cv2.INTER_LINEAR)
        # inp = inp.astype(np.float32) / 255.0
        inp = inp.astype(np.float32)
        color_aug(self._data_rng, inp, self._eig_val, self._eig_vec)
        inp = (inp - self.means) / self.std

        output_h = input_h // self.args.downratio
        output_w = input_w // self.args.downratio
        num_classes = self.args.num_classes
        trans_output = get_affine_transform(c, s, 0, [output_w, output_h])

        hm = np.zeros((num_classes, output_h, output_w), dtype=np.float32)
        wh = np.zeros((self.max_objs, 2), dtype=np.float32)
        reg = np.zeros((self.max_objs, 2), dtype=np.float32)
        reg_mask = np.zeros((self.max_objs), dtype=np.uint8)
        ind = np.zeros((self.max_objs), dtype=np.int64)
        gt_det = []
        num_objs = min(len(label), self.max_objs)

        for k in range(num_objs):
            cls_id = int(label[k])
            bbox = bboxs[k]

            bbox[[0, 2]] = width - bbox[[2, 0]] - 1 if flipped else bbox[[0, 2]]
            bbox[:2] = affine_transform(bbox[:2], trans_output)
            bbox[2:] = affine_transform(bbox[2:], trans_output)
            bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, output_w - 1)
            bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, output_h - 1)
            bbox_h, bbox_w = bbox[3] - bbox[1], bbox[2] - bbox[0]

            if bbox_h > 0 and bbox_w > 0:
                radius = gaussian_radius((math.ceil(bbox_h), math.ceil(bbox_w)))
                radius = max(0, int(radius))
                bbox_c = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
                bbox_c_int = bbox_c.astype(np.int32)
                draw_umich_gaussian(hm[cls_id], bbox_c_int, radius)
                wh[k] = 1.0 * bbox_w, 1.0 * bbox_h
                ind[k] = bbox_c_int[1] * output_w + bbox_c_int[0]  # TODO ind是啥
                reg[k] = bbox_c - bbox_c_int
                reg_mask[k] = 1

                gt_det.append(
                    [
                        bbox_c[0] - bbox_w / 2,
                        bbox_c[1] - bbox_h / 2,
                        bbox_c[0] + bbox_w / 2,
                        bbox_c[1] + bbox_h / 2,
                        1,
                        cls_id,
                    ]
                )
        target_info = {"hm": hm, "ind": ind, "wh": wh, "reg_mask": reg_mask, "reg": reg}
        inp = inp[:, :, (2, 1, 0)]
        return torch.from_numpy(inp).permute(2, 0, 1), target_info




class Test_transform:
    def __init__(self, args):
        self.args = args

    def __call__(self, image, boxes, label):
        height, width = image.shape[0:2]
        fix_res = not self.args.keep_res
        if fix_res:
            inp_height, inp_width = self.args.inp_imgsize, self.args.inp_imgsize
            c = np.array([width / 2.0, height / 2.0], dtype=np.float32)
            s = max(height, width) * 1.0
        else:
            inp_height = (height | self.args.pad) + 1
            inp_width = (width | self.args.pad) + 1
            c = np.array([width // 2, height // 2], dtype=np.float32)
            s = np.array([inp_width, inp_height], dtype=np.float32)

        trans_input = get_affine_transform(c, s, 0, [inp_width, inp_height])
        resized_image = cv2.resize(image, (width, height))
        inp_image = cv2.warpAffine(resized_image, trans_input, (inp_width, inp_height), flags=cv2.INTER_LINEAR)

        # inp_image = ((inp_image - "{}_means".format(self.args.detname)) / "{}_std".format(self.args.detname)).astype(np.float32)
        means = eval("{}_means".format(self.args.detname))
        std = eval("{}_std".format(self.args.detname))
        # inp_image = ((inp_image / 255. - means) / std).astype(np.float32)
        inp_image = ((inp_image - means) / std).astype(np.float32)
        images = inp_image.transpose(2, 0, 1).reshape(1, 3, inp_height, inp_width)
        if self.args.flip_test:
            images = np.concatenate((images, images[:, :, :, ::-1]), axis=0)
        images = torch.from_numpy(images)
        meta = {"c": c, "s": s, "out_height": inp_height // self.args.downratio, "out_width": inp_width // self.args.downratio}
        return images, meta


