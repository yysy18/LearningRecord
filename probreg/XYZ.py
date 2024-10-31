import numpy as np

# 读取原始txt文件
input_file = r'C:\Users\HP\Desktop\cloud21 - Cloud.txt'  # 替换为你的文件路径
output_file = 'cloud21_1.txt'  # 你希望保存的文件路径

# 读取txt文件，假设是以空格或制表符分隔
data = np.loadtxt(input_file, delimiter=None)

# 提取前3列
data_first_3_cols = data[:, :3]

# 保存前3列到新的txt文件
np.savetxt(output_file, data_first_3_cols, delimiter=' ', fmt='%f')

print(f"前3列已保存到 {output_file}")
