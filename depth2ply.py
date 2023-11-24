import cv2
import numpy as np
import open3d as o3d
import os
import time
from tqdm import tqdm


class point_cloud_generator():
    def __init__(self, rgb_file, depth_file, save_ply, camera_intrinsics):
        self.height = 576
        self.width = 1024
        self.rgb_file = rgb_file
        self.depth_file = depth_file
        self.save_ply = save_ply

        self.rgb = cv2.cvtColor(cv2.imread(rgb_file, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)
        self.rgb = cv2.resize(self.rgb, (self.width, self.height), interpolation=cv2.INTER_LANCZOS4)
        self.depth = np.load(self.depth_file).reshape((self.height, self.width))

        # self.depth = self.depth * self.depth_filter(self.depth)

        # 根据距离过滤深度值
        # self.depth[self.depth > 10] = 0

        print("your depth shape is:",self.depth.shape)

        assert self.width == self.depth.shape[1]
        assert self.height == self.depth.shape[0]

        self.camera_intrinsics = camera_intrinsics
    
    # 过滤深度图中梯度较大的区域
    def depth_filter(self, depth :np.array):
        threshold = 0.5

        mask = np.ones((self.height, self.width), dtype=np.int32)

        row_0 = depth[:self.height - 1, :] - depth[1:, :]
        row_0 = np.vstack([row_0, np.zeros((1, self.width))])
        mask[row_0 > threshold] = 0
        
        row_1 = depth[1:, :] - depth[:self.height - 1, :]
        row_1 = np.vstack([np.zeros((1, self.width)), row_1])
        mask[row_1 > threshold] = 0
        

        col_0 = depth[:, :self.width - 1] - depth[:, 1:]
        col_0 = np.hstack([col_0, np.zeros((self.height, 1))])
        mask[col_0 > threshold] = 0

        col_1 = depth[:, 1:] - depth[:, :self.width - 1]
        col_1 = np.hstack([np.zeros((self.height, 1)), col_1])
        mask[col_0 > threshold] = 0

        return mask


    def compute(self):
        t1 = time.time()

        # depth[depth==65535]=0
        self.Z = self.depth.T
        fx, fy, cx, cy = self.camera_intrinsics

        X = np.zeros((self.width, self.height))
        Y = np.zeros((self.width, self.height))
        for i in range(self.width):
            X[i, :] = np.full(X.shape[1], i)

        self.X = ((X - cx) * self.Z) / fx
        for i in range(self.height):
            Y[:, i] = np.full(Y.shape[0], i)
        self.Y = ((Y - cy) * self.Z) / fy

        data_ply = np.zeros((6, self.width * self.height))
        data_ply[0] = self.X.T.reshape(-1)
        data_ply[1] = -self.Y.T.reshape(-1)
        data_ply[2] = -self.Z.T.reshape(-1)
        img = np.array(self.rgb, dtype=np.uint8)
        data_ply[3] = img[:, :, 0:1].reshape(-1)
        data_ply[4] = img[:, :, 1:2].reshape(-1)
        data_ply[5] = img[:, :, 2:3].reshape(-1)
        self.data_ply = data_ply
        t2 = time.time()
        print('calcualte 3d point cloud Done.', t2 - t1)

    def write_ply(self):
        start = time.time()
        float_formatter = lambda x: "%.4f" % x
        points = []
        for i in self.data_ply.T:
            points.append("{} {} {} {} {} {} 0\n".format
                          (float_formatter(i[0]), float_formatter(i[1]), float_formatter(i[2]),
                           int(i[3]), int(i[4]), int(i[5])))

        file = open(self.save_ply, "w")
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
        %s
        ''' % (len(points), "".join(points)))
        file.close()

        end = time.time()
        print("Write into .ply file Done.", end - start)

    def show_point_cloud(self):
        pcd = o3d.io.read_point_cloud(self.save_ply)
        o3d.visualization.draw([pcd])


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

if __name__ == '__main__':
    camera_intrinsics = [518.916*0.8, 518.916*0.8, 655.332*0.8, 351.812*0.8]    # fx, fy, cx, cy

    # start_time = int("1686281683893")
    # end_time = int("1686281703894")

    # start_time = int("1686277956277")
    # end_time = int("1686277968678")

    # start_time = int("1686278009881")
    # end_time = int("1686278018082")

    # start_time = int("1686278282837")
    # end_time = int("1686278298638")

    # start_time = int("1686278308539")
    # end_time = int("1686278380678")

    # start_time = int("1686295740814")
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

    start_time = int("1693809971810")
    end_time = int("1693810052617")

    # start_time = int("1693810102821")
    # end_time = int("1693810121923")

    # start_time = int("1693810277236")
    # end_time = int("1693810286236")
    
    # start_time = int("1693810305338")
    # end_time = int("1693810305338")

    model_id = 54

    base_dir = "F:/data/depth_estimation/zed_dataset/"

    file_list, time_list = get_file_list_from_time()

    for idx, file in enumerate(file_list):
        rgb_file = os.path.join(base_dir, file)
        depth_file = "predict_result/predict_depth_w{}/{}_depth.npy".format(model_id, time_list[idx])
        save_ply = "predict_result/predict_depth_w{}/{}_depth.ply".format(model_id, time_list[idx])

        a = point_cloud_generator(rgb_file=rgb_file,
                                depth_file=depth_file,
                                save_ply=save_ply,
                                camera_intrinsics=camera_intrinsics)
        a.compute()
        a.write_ply()
    # a.show_point_cloud()
  


