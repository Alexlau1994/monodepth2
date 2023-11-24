import os
import numpy as np
import PIL.Image as pil
import cv2


def depth2campoints(depth):
    # img to 3d
    meshgrid = np.meshgrid(range(img_width), range(img_height), indexing='xy')
    id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
    ones = np.ones((1, img_height * img_width), dtype=np.float32)
    pix_coords = np.stack([id_coords[0].reshape(-1), id_coords[1].reshape(-1)], 0)
    pix_coords = np.vstack([pix_coords, ones])

    cam_points = np.matmul(inv_K[:3, :3], pix_coords)
    cam_points = depth.reshape(1, -1) * cam_points  

    return cam_points   # [3, imgh*imgw]


def get_mask_and_newpoint_with_trans(cam_points, T, last_cam_idx):
    ones = np.ones((1, cam_points.shape[1]), dtype=np.float32)
    cam_points = np.vstack([cam_points, ones])

    # transform points
    eps = 1e-7
    P = np.matmul(K, T)[:3, :]
    new_cam_points = np.matmul(T[:, :], cam_points)

    # 只寻找最近一次相机数据的mask
    points = np.matmul(K[:3, :3], new_cam_points[:3, last_cam_idx:])
    pix_coords = points[:2, :] / (points[2, :] + eps)
    pix_coords = pix_coords.reshape((2, -1))
    pix_coords = pix_coords.transpose((1, 0))   

    # 去除在新相机位置的框内上下左右像素值的点
    mask_x = np.logical_and(pix_coords[:, 0] > img_mask_W_min, pix_coords[:, 0] <= img_width - img_mask_W_max)
    mask_y = np.logical_and(pix_coords[:, 1] > img_mask_H_min, pix_coords[:, 1] <= img_height - img_mask_H_max)
    mask_z = new_cam_points[2, last_cam_idx:] > 0
    mask = ~np.logical_and(np.logical_and(mask_x, mask_y), mask_z)

    all_mask = np.ones((last_cam_idx), dtype=np.bool_)
    all_mask = np.hstack((all_mask, mask))
    return new_cam_points[:3, :], all_mask  # [3, points_num]  [points_num]


def read_img(img_file, use_sky_mask=True):
    img = cv2.imread(img_file, cv2.IMREAD_UNCHANGED)
    img = cv2.resize(img, (img_width, img_height), interpolation=cv2.INTER_LANCZOS4)

    if use_sky_mask:
        sky_mask_file = img_file.replace("left_img", "sky_mask_img").replace("left_image", "sky_mask_image")
        sky_mask_img = cv2.imread(sky_mask_file, cv2.IMREAD_UNCHANGED)
        sky_mask_img = cv2.resize(img, (img_width, img_height), interpolation=cv2.INTER_LANCZOS4)
        sky_mask_img = sky_mask_img[:, :, 0]


    img_mask = np.zeros((img_height, img_width))
    img_mask[img_mask_H_min:img_height - img_mask_H_max, img_mask_W_min:img_width - img_mask_W_max] = 1
    img_mask = img_mask == 1

    if use_sky_mask:
        sky_mask = sky_mask_img < 200
        img_mask = np.logical_and(sky_mask, img_mask)

    img_mask = img_mask.reshape(-1)
    img = img.reshape((img_height*img_width, 3)).T
    return img, img_mask  # [3, H*W]  [H*W]

def write_ply(all_points, all_bgr, save_file):
    file = open(save_file, "w")
    file.write('''ply
    format ascii 1.0
    element vertex %d
    property float x
    property float y
    property float z
    property uchar red
    property uchar green
    property uchar blue
    property uchar alpha
    end_header
    ''' % (all_points.shape[1]))

    for i in range(all_points.shape[1]):
        file.write("{} {} {} {} {} {} 0\n".format(
            all_points[0, i],
            -all_points[1, i],
            -all_points[2, i],
            all_bgr[2, i],
            all_bgr[1, i],
            all_bgr[0, i]
            ))
    
    file.close()


def get_file_list_from_time():
    all_files_order = "splits/zed/all_files_order.txt"
    with open(all_files_order, mode="r", encoding="utf-8") as fr:
        all_lines = fr.readlines()
    
    file_list = []
    time_list = []
    for line in all_lines:
        time = int(line.split("left_img/")[1].split("_left_image")[0])
        if time >= start_time and time <= end_time:
            file_list.append(line.split("###")[0])
            time_list.append(str(time))
    
    return file_list, time_list

def test():
    base_dir = "F:/data/depth_estimation/zed_dataset/"
    file_list, time_list = get_file_list_from_time()

    all_points = None
    all_bgr = None

    ply_file = "predict_result/predict_depth_w{}/{}_to_{}_max{}_step{}.ply".format(model_id, time_list[0], time_list[-1], max_depth, interval)

    first_img_file = os.path.join(base_dir, file_list[0])
    first_depth_file = "predict_result/predict_depth_w{}/{}_depth.npy".format(model_id, time_list[0])
    first_img, first_img_mask = read_img(first_img_file, use_sky_mask=use_sky_mask)
    first_depth = np.load(first_depth_file)[0, 0,...]
    first_cam_points = depth2campoints(first_depth)
    
    mask = first_cam_points[2, :] <= max_depth
    mask = np.logical_and(mask, first_img_mask)
    first_cam_points = first_cam_points[:, mask]
    first_img = first_img[:, mask]

    all_points = first_cam_points
    all_bgr = first_img

    # 记录最近一次相机数据起始点
    last_cam_idx = 0


    for idx in range(interval, len(time_list), interval):
        img_file = os.path.join(base_dir, file_list[idx])
        depth_file = "predict_result/predict_depth_w{}/{}_depth.npy".format(model_id, time_list[idx])

        M = np.eye(4, 4)
        for j in range(interval, 0, -1):
            R_file = "predict_result/predict_pose_w{}/{}_to_{}_R.npy".format(model_id, time_list[idx - j], time_list[idx - j + 1])
            T_file = "predict_result/predict_pose_w{}/{}_to_{}_T.npy".format(model_id, time_list[idx - j], time_list[idx - j + 1])
            R = np.load(R_file)[0, ...]
            T = np.load(T_file)[0, ...]
            M = np.matmul(M, np.matmul(T, R))

        img, img_mask = read_img(os.path.join(base_dir, img_file), use_sky_mask=use_sky_mask)
        depth = np.load(depth_file)[0, 0,...]

        new_cam_points, mask = get_mask_and_newpoint_with_trans(all_points, M, last_cam_idx)
        new_cam_points = new_cam_points[:, mask]
        all_bgr = all_bgr[: ,mask]

        campoints2 = depth2campoints(depth)

        mask = campoints2[2, :] <= max_depth
        mask = np.logical_and(mask, img_mask)
        campoints2 = campoints2[:, mask]
        img = img[:, mask]

        last_cam_idx = new_cam_points.shape[1]
        all_points = np.hstack((new_cam_points, campoints2))
        all_bgr = np.hstack((all_bgr, img))

    write_ply(all_points, all_bgr, ply_file)

if __name__ == '__main__':
    img_height = 576
    img_width = 1024

    max_depth = 50
    interval = 10
    img_mask_H_min = 30 # 相机上下左右边缘的信息过滤不要
    img_mask_H_max = 50 # 相机上下左右边缘的信息过滤不要
    img_mask_W_min = 50 # 相机上下左右边缘的信息过滤不要
    img_mask_W_max = 50 # 相机上下左右边缘的信息过滤不要

    use_sky_mask = False

    # camera intrinsics [518.916*0.8, 518.916*0.8, 655.332*0.8, 351.812*0.8]    # fx, fy, cx, cy
    K = np.array([[518.916*0.8, 0, 655.332*0.8, 0],
                  [0, 518.916*0.8, 351.812*0.8, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]], dtype=np.float32)
    inv_K = np.linalg.pinv(K)

    model_id = 64
    model_path = "log/MS_efn_zed_1024x576_kangbao_skymask/models/weights_{}".format(model_id)

    # huailai
    # start_time = int("1686281683893")
    # end_time = int("1686281703894")

    # start_time = int("1686277956277")
    # end_time = int("1686277968678")

    # start_time = int("1686278009881")  #
    # end_time = int("1686278018082")

    # start_time = int("1686278282837")
    # end_time = int("1686278298638")

    # start_time = int("1686278308539")
    # end_time = int("1686278380678")

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

    start_time = int("1693810277236")
    end_time = int("1693810286236")


    test()