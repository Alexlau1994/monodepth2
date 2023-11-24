from __future__ import absolute_import, division, print_function
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import cv2
import numpy as np

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from layers import disp_to_depth
from utils import readlines
import datasets
import networks

cv2.setNumThreads(0)  # This speeds up evaluation 5x on our unix systems (OpenCV 3.3.1)


splits_dir = os.path.join(os.path.dirname(__file__), "splits")


def compute_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    """
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25     ).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3

def evaluate():
    """Evaluates a pretrained model using a specified test set
    """

    print("-> Loading weights from {}".format(load_weights_folder))

    filenames = readlines(os.path.join(splits_dir, "zed/val_files.txt"))
    encoder_path = os.path.join(load_weights_folder, "encoder.pth")
    decoder_path = os.path.join(load_weights_folder, "depth.pth")

    encoder_dict = torch.load(encoder_path)

    dataset = datasets.ZedDataset(data_path, filenames,
                                    encoder_dict['height'], encoder_dict['width'],
                                    [0], 1, is_train=False)
    dataloader = DataLoader(dataset, 8, shuffle=False, num_workers=8,
                            pin_memory=True, drop_last=False)

    encoder = networks.efficientnet_b0()
    depth_decoder = networks.DepthDecoder(encoder.num_ch_enc)

    model_dict = encoder.state_dict()
    encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
    depth_decoder.load_state_dict(torch.load(decoder_path))

    encoder.cuda()
    encoder.eval()
    depth_decoder.cuda()
    depth_decoder.eval()

    pred_depths = []
    gt_depths = []

    print("-> Computing predictions with size {}x{}".format(
        encoder_dict['width'], encoder_dict['height']))

    with torch.no_grad():
        for data in tqdm(dataloader):
            input_color = data[("color", 0, 0)].cuda()

            output = depth_decoder(encoder(input_color))

            _, pred_depth = disp_to_depth(output[("disp", 0)], min_depth, max_depth)
            pred_depth = pred_depth.cpu()[:, 0].numpy()
            pred_depths.append(pred_depth)
            gt_depths.append(data["depth_gt"][:, 0, :, :])

    print("-> Evaluating")

    gt_depths = np.concatenate(gt_depths)
    pred_depths = np.concatenate(pred_depths)

    errors = []

    for i in range(pred_depths.shape[0]):

        gt_depth = gt_depths[i]
        pred_depth = pred_depths[i]

        mask = np.logical_and(gt_depth > min_depth, gt_depth < max_depth)

        pred_depth = pred_depth[mask]
        gt_depth = gt_depth[mask]

        pred_depth[pred_depth < min_depth] = min_depth
        pred_depth[pred_depth > max_depth] = max_depth

        errors.append(compute_errors(gt_depth, pred_depth))

    mean_errors = np.array(errors).mean(0)

    print("\n  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
    print(("&{: 8.3f}  " * 7).format(*mean_errors.tolist()) + "\\\\")
    print("\n-> Done!")


if __name__ == "__main__":
    load_weights_folder = "log/MS_efn_zed_1024x576_skymask/models/weights_9"
    data_path = "/mnt/data/datasets/zed_data"
    min_depth = 0.1
    max_depth = 100.0
    evaluate()
