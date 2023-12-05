import os
import random
import numpy as np
import copy
from PIL import Image  # using pillow-simd for increased speed

import torch
import torch.utils.data as data
from torchvision import transforms
from torchvision.transforms import functional
import cv2


def pil_loader(path):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


class CanonicalDataset(data.Dataset):
    """Superclass for monocular dataloaders

    Args:
        data_dir
        filenames
        height : 1088
        width  : 1440
        frame_idxs
        num_scales
        is_train
    """
    def __init__(self,
                 data_dir,
                 filenames,
                 height,
                 width,
                 frame_idxs,
                 num_scales,
                 is_train=False,
                 use_sky_mask=False):
        super(CanonicalDataset, self).__init__()

        self.data_dir = data_dir
        self.filenames = filenames

        # 模型输入统一高宽，不满足的使用padding 0补齐
        self.height = height
        self.width = width

        # 统一相机空间使用的focal lengths。参考Metric3D的做法“transforming depth labels”
        # 对于双目自监督，如果相片实际焦距不同，直接等比例修改双目相机的baseline
        self.canonical_focal_length = 1000
        self.stereo_baseline = 0

        self.num_scales = num_scales
        self.interp = Image.LANCZOS

        self.frame_idxs = frame_idxs

        self.is_train = is_train

        self.loader = pil_loader
        self.to_tensor = transforms.ToTensor()

        # We need to specify augmentations differently in newer versions of torchvision.
        # We first try the newer tuple version; if this fails we fall back to scalars
        self.brightness = (0.8, 1.2)
        self.contrast = (0.8, 1.2)
        self.saturation = (0.8, 1.2)
        self.hue = (-0.1, 0.1)

        self.resize = {}
        for i in range(self.num_scales):
            s = 2 ** i
            self.resize[i] = transforms.Resize((self.height // s, self.width // s),
                                               interpolation=self.interp)

        if is_train:
            self.load_depth = False
        else:
            self.load_depth = True
            
        self.use_sky_mask = use_sky_mask


    def preprocess(self, inputs, color_aug):
        """Resize colour images to the required scales and augment if required

        We create the color_aug object in advance and apply the same augmentation to all
        images in this item. This ensures that all images input to the pose network receive the
        same augmentation.
        """
        for k in list(inputs):
            if "color" in k:
                n, im, i = k
                for i in range(1, self.num_scales):
                    inputs[(n, im, i)] = self.resize[i](inputs[(n, im, i - 1)])
        
        for k in list(inputs):
            f = inputs[k]
            if "color" in k:
                n, im, i = k
                inputs[(n, im, i)] = self.to_tensor(f)
                if i == 0:
                    inputs[(n + "_aug", im, i)] = self.to_tensor(color_aug(f))
                

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        """Returns a single training item from the dataset as a dictionary.

        Values correspond to torch tensors.
        Keys in the dictionary are either strings or tuples:

            ("color", <frame_id>, <scale>)          for raw colour images,
            ("color_aug", <frame_id>, <scale>)      for augmented colour images,
            ("K", scale) or ("inv_K", scale)        for camera intrinsics,
            "stereo_T"                              for camera extrinsics, and
            "depth_gt"                              for ground truth depth maps.
            "sky_mask"                              mask for sky

        <frame_id> is either:
            an integer (e.g. 0, -1, or 1) representing the temporal step relative to 'index',
        or
            "s" for the opposite image in the stereo pair.

        <scale> is an integer representing the scale of the image relative to the fullsize image:
            -1      images at native resolution as loaded from disk
            0       images resized to (self.width,      self.height     )
            1       images resized to (self.width // 2, self.height // 2)
            2       images resized to (self.width // 4, self.height // 4)
            3       images resized to (self.width // 8, self.height // 8)
        """
        inputs = {}

        do_color_aug = self.is_train and random.random() > 0.5
        do_flip = self.is_train and random.random() > 0.5
        
        # only use left img as target
        data_dict = eval(self.filenames[index])["CAM_Fl"]

        for i in self.frame_idxs:
            if i == 0:
                file_path = data_dict["filepath"]
            if i == "s":
                file_path = data_dict["depth"]["img_stero"]
            elif i == -1:
                file_path = data_dict["depth"]["img_pre"]
            elif i == 1:
                file_path = data_dict["depth"]["img_next"]
            
            inputs[("color", i, 0)] = self.get_color(file_path, do_flip)

        K, baseline = self.read_calib_yaml(data_dict["depth"]["calib"], do_flip)

        # adjusting intrinsics to match each scale in the pyramid
        for scale in range(self.num_scales):
            resize_K = K.copy()
            resize_K[0, :] *= self.width // (2 ** scale)
            resize_K[1, :] *= self.height // (2 ** scale)

            inv_K = np.linalg.pinv(resize_K)

            inputs[("K", scale)] = torch.from_numpy(resize_K)
            inputs[("inv_K", scale)] = torch.from_numpy(inv_K)

        if do_color_aug:
            color_aug = transforms.ColorJitter(self.brightness, self.contrast, self.saturation, self.hue)
        else:
            color_aug = (lambda x: x)

        self.preprocess(inputs, color_aug)

        # for i in self.frame_idxs:
        #     del inputs[("color", i, -1)]
            # del inputs[("color_aug", i, -1)]

        if self.load_depth:
            depth_gt = self.get_depth(data_dict["depth"]["depth_img"], do_flip)
            inputs["depth_gt"] = depth_gt
        
        if self.use_sky_mask:
            sky_mask = self.get_sky_mask(file_path, do_flip)
            inputs["sky_mask"] = sky_mask

        if "s" in self.frame_idxs:
            stereo_T = np.eye(4, dtype=np.float32)
            baseline_sign = -1 if do_flip else 1
            side_sign = -1
            stereo_T[0, 3] = side_sign * baseline_sign * baseline

            inputs["stereo_T"] = torch.from_numpy(stereo_T)

        return inputs

    def read_calib_yaml(self, file_path: str, do_flip: bool):
        cv_file = cv2.FileStorage(os.path.join(self.data_dir, file_path), cv2.FILE_STORAGE_READ)
        ori_K = cv_file.getNode("K").mat()
        ori_imgH = cv_file.getNode("imgH").real()
        ori_imgW = cv_file.getNode("imgW").real()
        ori_baseline = cv_file.getNode("baseline").real()

        if do_flip:
            ori_K[0, 2] = ori_imgW - ori_K[0, 2]
        
        K = np.eye(4, dtype=np.float32)
        K[:3, :3] = ori_K
        K[0, :] /= ori_imgW
        K[1, :] /= ori_imgH

        baseline = ori_baseline * self.canonical_focal_length / ori_K[0, 0]

        return K, baseline

    def get_color(self, file_path: str, do_flip):
        color = self.loader(os.path.join(self.data_dir, file_path))

        if do_flip:
            color = color.transpose(Image.FLIP_LEFT_RIGHT)

        color = functional.crop(color, 0, 0, self.height, self.width)
        return color

    def get_depth(self, file_path: str, do_flip: bool):
        depth_gt = cv2.imread(os.path.join(self.data_dir, file_path), cv2.IMREAD_UNCHANGED)
        depth_gt = depth_gt.astype(np.float32) / 256    # 深度真实值为原图像值统一除以256

        if do_flip:
            depth_gt = np.fliplr(depth_gt)

        depth_gt = np.expand_dims(depth_gt, 0)
        depth_gt = torch.from_numpy(depth_gt)
        depth_gt = functional.crop(depth_gt, 0, 0, self.height, self.width)

        return depth_gt
    
    def get_sky_mask(self, file_path: str, do_flip: bool):
        sky_mask = cv2.imread(os.path.join(self.data_path, file_path), cv2.IMREAD_UNCHANGED)
        if do_flip:
            sky_mask = np.fliplr(sky_mask)
        
        sky_mask = np.expand_dims(sky_mask, 0)
        sky_mask = torch.from_numpy(sky_mask)
        sky_mask = functional.crop(sky_mask, 0, 0, self.height, self.width)
        sky_mask = sky_mask == 255
        
        return sky_mask