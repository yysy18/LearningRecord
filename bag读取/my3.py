import os
import rosbag
import sensor_msgs.point_cloud2 as pc2
import numpy as np
import open3d as o3d

bag_file = 'rgbd_dataset_freiburg3_teddy-2hz-with-pointclouds.bag'  # 替换为你的.bag文件路径
pointcloud_topic = '/camera/rgb/points'  # 替换为包含点云数据的ROS主题
output_folder = 'my_ply'  # 存储点云数据的目标文件夹路径
output_txt_file = 'timestamps.txt'  # 输出时间戳的.txt文件名

# 创建目标文件夹（如果不存在）
os.makedirs(output_folder, exist_ok=True)

# 打开.bag文件
bag = rosbag.Bag(bag_file)

frame_number = 0

# 打开.txt文件以写入时间戳
txt_file = open(os.path.join(output_folder, output_txt_file), 'w')

# 遍历点云数据主题
for _, msg, _ in bag.read_messages(topics=[pointcloud_topic]):
    # 记录时间戳到.txt文件
    txt_file.write(f"{msg.header.stamp}\n")

    # 生成点云数据
    gen = pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
    points = []
    for p in gen:
        points.append([p[0], p[1], p[2]])
    points = np.array(points)

    # 保存点云数据到PLY文件
    output_file = os.path.join(output_folder, f'frame_{frame_number}.ply')
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud(output_file, cloud)

    frame_number += 1

# 关闭.txt文件和.bag文件
txt_file.close()
bag.close()

print(f"成功保存了 {frame_number} 个点云数据和时间戳到文件夹 '{output_folder}' 中。")