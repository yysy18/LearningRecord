import numpy as np
import os
from scipy.spatial.transform import Rotation as R

# 输入文件路径
input_file = "/home/hanglok/Desktop/HXJ/code/OpenIns3D/data/hxj/rgbd/d_cloud3/poses3.txt"
# 输出文件夹路径
output_folder = "/home/hanglok/Desktop/HXJ/code/OpenIns3D/data/hxj/rgbd/d_cloud3/pose"

# 创建输出文件夹（如果不存在）
os.makedirs(output_folder, exist_ok=True)

# 用于存储位姿信息的列表
pose_list = []

# 读取poses3.txt文件
with open(input_file, 'r') as f:
    lines = f.readlines()

# 解析每一行的数据，跳过注释行
for line in lines:
    if line.startswith("#"):
        continue

    # 解析每一行
    elements = line.strip().split()
    timestamp = float(elements[0])
    x, y, z = float(elements[1]), float(elements[2]), float(elements[3])
    qx, qy, qz, qw = float(elements[4]), float(elements[5]), float(elements[6]), float(elements[7])
    pose_id = int(elements[8])

    # 将数据保存到列表中
    pose_list.append((pose_id, x, y, z, qx, qy, qz, qw))

# 按pose_id排序
pose_list.sort(key=lambda x: x[0])

# 遍历排序后的列表，生成外参矩阵并按顺序保存
for new_id, (pose_id, x, y, z, qx, qy, qz, qw) in enumerate(pose_list):
    # 将四元数转换为旋转矩阵
    rotation = R.from_quat([qx, qy, qz, qw]).as_matrix()

    # 创建4x4外参矩阵
    pose_matrix = np.eye(4)
    pose_matrix[:3, :3] = rotation
    pose_matrix[:3, 3] = [x, y, z]

    # 保存矩阵为.npy文件，按照新的编号命名
    output_file = os.path.join(output_folder, f"pose_matrix_calibrated_angle_{new_id}.npy")
    np.save(output_file, pose_matrix)

    print(f"已保存矩阵至: {output_file}")
