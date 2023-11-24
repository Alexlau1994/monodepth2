import os
import random
import numpy as np
import copy
from PIL import Image  # using pillow-simd for increased speed

import torch
import torch.utils.data as data
from torchvision import transforms
import cv2


def pil_loader(path):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


class ZedDataset(data.Dataset):
    """Superclass for monocular dataloaders

    Args:
        data_path
        filenames
        height : 576
        width  : 1024
        frame_idxs
        num_scales
        is_train
    """
    def __init__(self,
                 data_path,
                 filenames,
                 height,
                 width,
                 frame_idxs,
                 num_scales,
                 is_train=False,
                 use_sky_mask=False,
                 all_files_order="splits/zed/all_files_order.txt"):
        super(ZedDataset, self).__init__()

        self.K = np.array([[518.916 / 1280, 0, 655.332 / 1280, 0],
                           [0, 518.916 / 720, 351.812 / 720, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)
        
        self.data_path = data_path
        self.filenames = filenames
        self.order2file = dict()
        # self.file2order = dict()
        self.height = height
        self.width = width
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
        
        self.get_filenames_and_order(all_files_order)

        self.load_depth = True
        self.use_sky_mask = use_sky_mask

    def get_filenames_and_order(self, all_files_order):
        with open(all_files_order, mode="r", encoding="utf-8") as fr:
            lines = fr.readlines()
        for f_d in lines:
            fd_split = f_d.split("###")
            assert len(fd_split) == 2
            self.order2file[int(fd_split[1])] = fd_split[0]
            # self.file2order[fd_split[0]] = fd_split[1]

    def preprocess(self, inputs, color_aug):
        """Resize colour images to the required scales and augment if required

        We create the color_aug object in advance and apply the same augmentation to all
        images in this item. This ensures that all images input to the pose network receive the
        same augmentation.
        """
        for k in list(inputs):
            frame = inputs[k]
            if "color" in k:
                n, im, i = k
                for i in range(self.num_scales):
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
        id_lr = self.filenames[index].split("###")
        id = int(id_lr[0])
        file_path = self.order2file[id]

        for i in self.frame_idxs:
            if i == "s":
                new_file_path = file_path.replace("left_image", "right_image").replace("left_img", "right_img")
            elif i == -1:
                new_file_path = self.order2file[id - 1]
            elif i == 1:
                new_file_path = self.order2file[id + 1]
            else:
                new_file_path = file_path

            inputs[("color", i, -1)] = self.get_color(new_file_path, do_flip)

        # adjusting intrinsics to match each scale in the pyramid
        for scale in range(self.num_scales):
            K = self.K.copy()
            if do_flip:
                K[0, 2] = 1 - K[0, 2]

            K[0, :] *= self.width // (2 ** scale)
            K[1, :] *= self.height // (2 ** scale)

            inv_K = np.linalg.pinv(K)

            inputs[("K", scale)] = torch.from_numpy(K)
            inputs[("inv_K", scale)] = torch.from_numpy(inv_K)

        if do_color_aug:
            color_aug = transforms.ColorJitter(self.brightness, self.contrast, self.saturation, self.hue)
        else:
            color_aug = (lambda x: x)

        self.preprocess(inputs, color_aug)

        for i in self.frame_idxs:
            del inputs[("color", i, -1)]
            # del inputs[("color_aug", i, -1)]

        if self.load_depth:
            depth_gt = self.get_depth(file_path, do_flip)
            depth_gt = np.expand_dims(depth_gt, 0)
            inputs["depth_gt"] = torch.from_numpy(depth_gt.astype(np.float32))
        
        if self.use_sky_mask:
            sky_mask = self.get_sky_mask(file_path, do_flip)
            sky_mask = np.expand_dims(sky_mask, 0).astype(np.float32)
            inputs["sky_mask"] = torch.from_numpy(sky_mask)
            inputs["sky_mask"] = inputs["sky_mask"] == 255

        if "s" in self.frame_idxs:
            stereo_T = np.eye(4, dtype=np.float32)
            baseline_sign = -1 if do_flip else 1
            side_sign = -1
            stereo_T[0, 3] = side_sign * baseline_sign * 0.12   # basic_line is 12cm for zed 2i camera

            inputs["stereo_T"] = torch.from_numpy(stereo_T)

        return inputs

    def get_color(self, file_path: str, do_flip):
        color = self.loader(os.path.join(self.data_path, file_path))

        if do_flip:
            color = color.transpose(Image.FLIP_LEFT_RIGHT)

        return color

    def get_depth(self, file_path: str, do_flip: bool):
        depth_file = file_path.replace("left_img", "depth_img").replace("left_image.jpg", "depth_map_0.2_40.png")
        confidence_file = file_path.replace("left_img", "confidence_img").replace("left_image.jpg", "confidence_map.png")
        
        depth_gt = cv2.imread(os.path.join(self.data_path, depth_file), cv2.IMREAD_UNCHANGED)
        depth_gt = cv2.resize(depth_gt, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
        depth_gt = depth_gt.astype(np.float32) / 1000

        cfd = cv2.imread(os.path.join(self.data_path, confidence_file), cv2.IMREAD_UNCHANGED)
        cfd = cv2.resize(depth_gt, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
        cfd = cfd.astype(np.int32)

        assert cfd.shape == depth_gt.shape

        depth_gt[cfd > 5] = 0   # 通过confidence map去除部分深度结果

        if do_flip:
            depth_gt = np.fliplr(depth_gt)

        return depth_gt
    
    def get_sky_mask(self, file_path: str, do_flip: bool):
        sky_mask_file = file_path.replace("left_img", "sky_mask_img_new").replace("left_image.jpg", "sky_mask_image.png")
        sky_mask = cv2.imread(os.path.join(self.data_path, sky_mask_file), cv2.IMREAD_UNCHANGED)
        sky_mask = cv2.resize(sky_mask, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
        if do_flip:
            sky_mask = np.fliplr(sky_mask)
        
        # sky_mask = sky_mask.astype(np.float32)
        # sky_mask[sky_mask != 255] = 0
        # sky_mask[sky_mask == 255] = 1

        # sky_depth = np.zeros_like(sky_mask, dtype=np.float32)
        
        return sky_mask