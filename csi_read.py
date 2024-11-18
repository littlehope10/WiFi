import csiread
import numpy as np
import os
from utils.jsoncsi import JsonCSI

# 读取csi文件
def read(csi_name, save_json):
    file_name = f'{csi_name}.dat'
    csi_file = os.path.join('csi', file_name)
    csi_data = csiread.Intel(csi_file)
    csi_data.read()

    csi_array = np.array(csi_data.csi)

    # 数据包，发射天线，接收天线，子载波
    csi_np = np.transpose(csi_array, (0,3,2,1))

    if save_json:
        jsoncsi = JsonCSI()
        jsoncsi.save_primary_csi_data(csi_np, csi_name)

    return csi_np

def cal_abs(csi_np, tx, rx):
    rx_csi_amp = []

    packet_number = csi_np.shape[0]
    # 第一步，绘制波形图像
    # 得到的数据: 横轴：数据包，纵轴：载波
    for i in range(0, packet_number):
        # 取第i个数据包
        csi_package = csi_np[i]
        # 第i个数据包的第一个天线
        csi_carrier = csi_package[tx - 1, rx - 1, :]
        # 提取幅值
        csi1_amp = np.abs(csi_carrier)

        rx_csi_amp.append(csi1_amp[:])

    rx_csi_amp = np.array(rx_csi_amp)

    return rx_csi_amp

