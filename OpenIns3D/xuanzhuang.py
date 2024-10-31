import open3d as o3d
import numpy as np
import copy

def load_and_display_point_cloud(file_path):
    """
    加载并显示原始和旋转后的点云。

    :param file_path: 点云文件的路径 (如 .ply 或 .pcd 文件)。
    """
    # 读取点云文件
    pcd = o3d.io.read_point_cloud(file_path)
    
    # 显示原始点云
    print("Displaying original point cloud:")
    o3d.visualization.draw_geometries([pcd])

    # 复制点云
    pcd_rotated = copy.deepcopy(pcd)

    # 绕Y轴旋转90度
    R = pcd_rotated.get_rotation_matrix_from_xyz((0, np.pi/2, 0))  # 90度 = pi/2 弧度
    pcd_rotated.rotate(R, center=(0, 0, 0))  # 默认绕原点旋转

    # 显示旋转后的点云
    print("Displaying 90° rotated point cloud:")
    o3d.visualization.draw_geometries([pcd_rotated])

# 示例：提供你的点云文件路径
point_cloud_path = "/home/hanglok/Desktop/HXJ/code/OpenIns3D/data/hxj/scenes/demo_2.ply"  # 请更改为你的实际文件路径
load_and_display_point_cloud(point_cloud_path)
