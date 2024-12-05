import csiread
import numpy as np
import matplotlib.pyplot as plt
from csi_read import read, cal_abs
from scipy.signal import medfilt
from utils.methods import hampel_remove, apply_pca, wavelet_denoise, wavelet_noising

# 该文件是直接读取原始csi图像

# 文件路径为以下格式
# csi_file = '11_28/c1_1_hor'
csi_file = 'csi_p1_10_hor'


csi_np = read(csi_file, False)
tx = csi_np.shape[1]
rx = csi_np.shape[2]

for t in range(0, tx):
    for r in range(0, rx):
        csi = csi_np[:, t, r, :]
        csi = np.abs(csi)

        # 这两行代码为两种去噪方法，相比而言，第一个比第二个好

        # csi = medfilt(csi, kernel_size=7)
        csi = hampel_remove(csi, 11, 3)
        # csi = apply_pca(csi, 0)
        # csi = wavelet_noising(csi)
        plt.subplot(tx, rx, t*rx+r+1)
        plt.plot(csi)
        plt.xlabel('Package')
        plt.ylabel('Abs')
        plt.title(f'{t+1} tx {r+1} rx')
plt.show()

