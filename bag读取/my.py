import os
import rosbag

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
    txt_file.write(f"timestamp {msg.header.stamp.to_sec()}\n")  # 将ROS时间戳转换为秒并写入文件

    # 其他处理代码，例如保存点云数据到PLY文件等
    # ...

    frame_number += 1

# 关闭.txt文件和.bag文件
txt_file.close()
bag.close()

print(f"成功保存了 {frame_number} 个时间戳到文件 '{output_txt_file}' 中。")
