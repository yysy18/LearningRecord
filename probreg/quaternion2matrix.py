import numpy as np

def quaternion_to_rotation_matrix(q):
    """Convert quaternion to rotation matrix."""
    w, x, y, z = q
    R = np.array([[1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
                  [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
                  [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]])
    return R

def rotation_matrix_to_euler_angles(R):
    """Convert rotation matrix to Euler angles (ZYX convention)."""
    sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6

    if not singular:
        x = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(-R[2, 0], sy)
        z = np.arctan2(R[1, 0], R[0, 0])
    else:
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z])

# 示例数据
q1 = np.array([0.7185, -0.3724, 0.3244, -0.4898])
q2 = np.array([-0.1966, 0.8005, -0.5502, 0.1332])

# 转换为旋转矩阵
R1 = quaternion_to_rotation_matrix(q1)
R2 = quaternion_to_rotation_matrix(q2)

# 计算旋转矩阵之间的角度差
angle_diff = rotation_matrix_to_euler_angles(R1.T.dot(R2))

print("旋转角度差 (Euler 角，弧度)：", angle_diff)

# 计算平移向量（假设给定位置向量）
p1 = np.array([-3.4590, -0.5675, 1.5454])
p2 = np.array([2.5221, 1.7862, -2.8527])
translation = p2 - p1
print("平移向量：", translation)
