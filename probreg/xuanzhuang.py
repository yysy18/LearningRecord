import open3d as o3d
import numpy as np
import copy


def load_and_display_point_cloud(file_path):
    """
    加载并同时显示原始和旋转后的点云。

    :param file_path: 点云文件的路径 (如 .ply 或 .pcd 文件)。
    """
    # 读取点云文件
    pcd = o3d.io.read_point_cloud(file_path)

    # 复制点云
    pcd_rotated = copy.deepcopy(pcd)

    # 绕Y轴旋转90度
    R = pcd_rotated.get_rotation_matrix_from_xyz((np.pi, np.pi / 2, 0))  # 90度 = pi/2 弧度
    pcd_rotated.rotate(R, center=(0, 0, 0))  # 默认绕原点旋转

    # 创建第一个窗口用于显示原始点云
    vis1 = o3d.visualization.Visualizer()
    vis1.create_window(window_name="Original Point Cloud", width=800, height=600)
    vis1.add_geometry(pcd)

    # 创建第二个窗口用于显示旋转后的点云
    vis2 = o3d.visualization.Visualizer()
    vis2.create_window(window_name="Rotated Point Cloud", width=800, height=600)
    vis2.add_geometry(pcd_rotated)

    # 更新和渲染窗口
    vis1.update_geometry(pcd)
    vis1.poll_events()
    vis1.update_renderer()

    vis2.update_geometry(pcd_rotated)
    vis2.poll_events()
    vis2.update_renderer()

    # 保持两个窗口显示
    while True:
        vis1.poll_events()
        vis1.update_renderer()

        vis2.poll_events()
        vis2.update_renderer()


# 示例：提供你的点云文件路径
point_cloud_path = "D:/HXJ/code/OpenIns3D/data/demo_single_voc/scenes/demo_2.ply"  # 请更改为你的实际文件路径
load_and_display_point_cloud(point_cloud_path)
