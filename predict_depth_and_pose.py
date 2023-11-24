# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import glob
import numpy as np
import PIL.Image as pil
import cv2
import torch
from torchvision import transforms
from tqdm import tqdm
import matplotlib as mpl
import matplotlib.cm as cm

import networks
from layers import disp_to_depth
from layers import transformation_from_parameters, rot_from_axisangle, get_translation_matrix


def predcit_depth(file_list, save_dir):
    device = torch.device("cuda")
    print("-> Loading model from ", model_path)
    depth_encoder_path = os.path.join(model_path, "encoder.pth")
    depth_decoder_path = os.path.join(model_path, "depth.pth")

    # LOADING PRETRAINED MODEL
    print("   Loading pretrained depth encoder")
    depth_encoder = networks.efficientnet_b0()
    depth_encoder_dict= torch.load(depth_encoder_path, map_location=device)

    # extract the height and width of image that this model was trained with
    feed_height = depth_encoder_dict['height']
    feed_width = depth_encoder_dict['width']
    depth_encoder.load_state_dict(depth_encoder_dict, strict=False)
    depth_encoder.to(device)
    depth_encoder.eval()

    print("   Loading pretrained depth decoder")
    depth_decoder = networks.DepthDecoder(
        num_ch_enc=depth_encoder.num_ch_enc, scales=range(4))

    loaded_dict = torch.load(depth_decoder_path, map_location=device)
    depth_decoder.load_state_dict(loaded_dict)
    depth_decoder.to(device)
    depth_decoder.eval()

    # PREDICTING ON EACH IMAGE IN TURN
    with torch.no_grad():
        for image_path in tqdm(file_list):
            # Load image and preprocess
            input_image = pil.open(image_path).convert('RGB')
            input_image = input_image.resize((feed_width, feed_height), pil.Resampling.LANCZOS)
            input_image = transforms.ToTensor()(input_image).unsqueeze(0)

            # PREDICTION
            input_image = input_image.to(device)
            features = depth_encoder(input_image)
            outputs = depth_decoder(features)
            disp = outputs[("disp", 0)]

            # Saving numpy file
            output_name = os.path.splitext(os.path.basename(image_path).replace("_left_image", ""))[0]
            _, depth = disp_to_depth(disp, 0.1, 100)
            name_dest_npy = os.path.join(save_dir, "{}_depth.npy".format(output_name))
            metric_depth = depth.cpu().numpy()
            np.save(name_dest_npy, metric_depth)
            # metric_depth = (metric_depth * 256).astype(np.uint16)[0,0,:]
            # cv2.imwrite(os.path.join(save_dir, "{}_depth.png".format(output_name)), metric_depth)

            # Saving colormapped depth image
            disp_resized_np = disp.squeeze().cpu().numpy()
            vmax = np.percentile(disp_resized_np, 80)
            normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=vmax)
            mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
            colormapped_im = (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(np.uint8)
            im = pil.fromarray(colormapped_im)
            name_dest_im = os.path.join(save_dir, "{}_disp.jpeg".format(output_name))
            im.save(name_dest_im)

    print('predict depth finished!')


def predict_pose(file_list, save_dir):
    if len(file_list) <= 1:
        print("length of file list is less than 2, not predict pose")
        return

    device = torch.device("cuda")
    print("-> Loading model from ", model_path)
    pose_encoder_path = os.path.join(model_path, "pose_encoder.pth")
    pose_decoder_path = os.path.join(model_path, "pose.pth")

    # LOADING PRETRAINED MODEL
    print("   Loading pretrained pose encoder")
    pose_encoder = networks.efficientnet_b0(num_input_images=2)
    pose_encoder_dict= torch.load(pose_encoder_path, map_location=device)

    pose_encoder.load_state_dict(pose_encoder_dict, strict=False)
    pose_encoder.to(device)
    pose_encoder.eval()

    print("   Loading pretrained pose decoder")
    pose_decoder = networks.PoseDecoder(
        num_ch_enc=pose_encoder.num_ch_enc,
        num_input_features=1,
        num_frames_to_predict_for=2)

    loaded_dict = torch.load(pose_decoder_path, map_location=device)
    pose_decoder.load_state_dict(loaded_dict)
    pose_decoder.to(device)
    pose_decoder.eval()

    # PREDICTING ON EACH IMAGE IN TURN
    with torch.no_grad():
        for idx in tqdm(range(len(file_list) - 1)):
            img1_path = file_list[idx]
            img2_path = file_list[idx + 1]
            # Load image and preprocess
            input1 = pil.open(img1_path).convert('RGB')
            input1 = input1.resize((img_width, img_height), pil.Resampling.LANCZOS)
            input1 = transforms.ToTensor()(input1).unsqueeze(0)
            input2 = pil.open(img2_path).convert('RGB')
            input2 = input2.resize((img_width, img_height), pil.Resampling.LANCZOS)
            input2 = transforms.ToTensor()(input2).unsqueeze(0)

            # PREDICTION
            input_data = torch.cat([input1, input2], dim=1)
            input_data = input_data.to(device)
            features = [pose_encoder(input_data)]
            axisangle, translation = pose_decoder(features)
            R = rot_from_axisangle(axisangle[:, 0]).cpu().numpy()
            T = get_translation_matrix(translation[:, 0]).cpu().numpy()
            base1 = os.path.splitext(os.path.basename(img1_path).replace("_left_image", ""))[0]
            base2 = os.path.splitext(os.path.basename(img2_path).replace("_left_image", ""))[0]
            save_R_file = os.path.join(save_dir, "{}_to_{}_R.npy".format(base1, base2))
            save_T_file = os.path.join(save_dir, "{}_to_{}_T.npy".format(base1, base2))
            np.save(save_R_file, R)
            np.save(save_T_file, T)

def get_file_list_from_time():
    all_files_order = "splits/zed/all_files_order.txt"
    with open(all_files_order, mode="r", encoding="utf-8") as fr:
        all_lines = fr.readlines()
    
    file_list = []
    for line in all_lines:
        time = int(line.split("left_img/")[1].split("_left_image")[0])
        if time >= start_time and time <= end_time:
            file_list.append(line.split("###")[0])
    
    return file_list


def predict_depth_and_pose():
    file_list = get_file_list_from_time()

    base_dir = "F:/data/depth_estimation/zed_dataset/"
    files = [os.path.join(base_dir, item) for item in file_list]

    depth_save_dir = "predict_result/predict_depth_w{}".format(model_id)
    pose_save_dir = "predict_result/predict_pose_w{}".format(model_id)
    os.makedirs(depth_save_dir, exist_ok=True)
    os.makedirs(pose_save_dir, exist_ok=True)

    predcit_depth(file_list=files, save_dir=depth_save_dir)
    predict_pose(file_list=files, save_dir=pose_save_dir)


if __name__ == '__main__':
    img_height = 576
    img_width = 1024

    # camera intrinsics [518.916*0.8, 518.916*0.8, 655.332*0.8, 351.812*0.8]    # fx, fy, cx, cy
    K = np.array([[518.916*0.8, 0, 655.332*0.8, 0],
                  [0, 518.916*0.8, 351.812*0.8, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]], dtype=np.float32)
    inv_K = np.linalg.pinv(K)

    model_id = 63
    model_path = "log/MS_efn_zed_1024x576_kangbao_skymask/models/weights_{}".format(model_id)

    # huailai
    start_time = int("1686281683893")
    end_time = int("1686281703894")

    # start_time = int("1686277956277")
    # end_time = int("1686277968678")

    # start_time = int("1686278009881")  #
    # end_time = int("1686278018082")

    # start_time = int("1686278282837")
    # end_time = int("1686278298638")

    # start_time = int("1686278308539")
    # end_time = int("1686278380678")

    # start_time = int("1686295740814") #
    # end_time = int("1686295740914")

    # kangbao
    # start_time = int("1693809267352")
    # end_time = int("1693809287353")

    # start_time = int("1693809317456")
    # end_time = int("1693809379461")

    # start_time = int("1693809482069")
    # end_time = int("1693809558476")

    # start_time = int("1693809713589")
    # end_time = int("1693809738491")

    # start_time = int("1693809971810")
    # end_time = int("1693810052617")

    # start_time = int("1693810102821")
    # end_time = int("1693810121923")

    # start_time = int("1693810277236")
    # end_time = int("1693810286236")


    # start_time = int("1693810305338")
    # end_time = int("1693810305338")

    predict_depth_and_pose()
    