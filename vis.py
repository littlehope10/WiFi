import numpy as np
import matplotlib.pyplot as plt
from pyts.image import RecurrencePlot
from scipy.signal import remez, freqz, lfilter
from scipy.fft import fft
from utils.methods import mad_remove, apply_kalman_smooth, wavelet_denoise
from utils.jsoncsi import JsonCSI
from csi_read import read, cal_abs

import matplotlib
matplotlib.use('tkAgg')

# 设置图形配置
config = {
    "font.family": 'serif',
    "mathtext.fontset": 'stix',  # matplotlib渲染数学字体时使用的字体，和Times New Roman差别不大
    "font.serif": ['SimSun'],  # 宋体
    'axes.unicode_minus': False  # 处理负号，即-号
}
matplotlib.rcParams.update(config)


csi_np = read('csi-static', True)
# 数据包，发射天线，接收天线，子载波
packet_number = csi_np.shape[0]
tx_number = csi_np.shape[1]
rx_number = csi_np.shape[2]
subcarrier_number = csi_np.shape[3]

# 第一步，绘制波形图像
# 得到的数据: 横轴：数据包，纵轴：载波

first_rx_amp_csi = cal_abs(csi_np, 1, 1)

first = first_rx_amp_csi

# 第二步，绝对中位差（MAD）异常值剔除
# 剔除公式：[M - 3 * MAD， M + 3 * MAD]超过此范围的值替换为中位数M
first_rx_amp_csi = mad_remove(first_rx_amp_csi, 3)

# 第三步，小波去噪

first_rx_amp_csi_denoised = np.apply_along_axis(wavelet_denoise, axis=0, arr=first_rx_amp_csi)

# 第四步，卡尔曼滤波器平滑

first_rx_amp_csi_smoothed = apply_kalman_smooth(first_rx_amp_csi_denoised)




jsoncsi = JsonCSI()

first_rx_amp_csi_smoothed = jsoncsi.load_csi_data('csi-static', 1, 1, True)

recurrence_plot = RecurrencePlot(threshold='point', percentage=20)
best_index = None
max_similarity = 0
rp_image = None

for i, subcarrier in enumerate(first_rx_amp_csi_smoothed.T):
    rp_image = recurrence_plot.fit_transform(subcarrier.reshape(1, -1))[0]
    similarity = np.sum(rp_image)  # 计算图中点的总和表示相似度
    if similarity > max_similarity:
        max_similarity = similarity
        best_index = i

csi_remove_DC = np.zeros_like(first_rx_amp_csi_smoothed)

for i in range(0, 30):
    carrier = first_rx_amp_csi_smoothed[:, i]
    row_mean = np.mean(carrier, axis=0, keepdims=True)
    row_mean = np.tile(row_mean, carrier.shape[0])
    carrier_remove_DC = carrier - row_mean
    csi_remove_DC[:, i] = carrier_remove_DC

best_csi = first_rx_amp_csi_smoothed[:,best_index]
best_csi_remove_DC = csi_remove_DC[:,best_index]
csi_remove_DC_plus = np.sum(csi_remove_DC, axis=1)

# best_csi = first_rx_amp_csi_smoothed[:,best_index]
# row_mean = np.mean(best_csi, axis=0, keepdims=True)
# row_mean = np.tile(row_mean, best_csi.shape[0])
# best_csi_remove_DC = best_csi - row_mean


# filtered_data, fft_result = filter_and_fft(best_csi_remove_DC)


# 绘制最佳子载波
plt.subplot(2,3,1)
a = first_rx_amp_csi_smoothed[:,best_index]
plt.plot(best_csi)
plt.plot(best_csi_remove_DC)
plt.plot(csi_remove_DC_plus)
plt.xlabel('Package')
plt.ylabel('Amplitude')
plt.title(f'Recurrence Plot of Best Subcarrier (Index {best_index})')

# plt.subplot(2,2,1)
# plt.plot(first)
# plt.title('1st TX 1st RX Static CSI Magnitude')
# plt.xlabel('Package')
# plt.ylabel('Amplitude')
#
# plt.subplot(2,2,2)
# plt.plot(first_rx_amp_csi)
# plt.title('After MAD Remove')
# plt.xlabel('Package')
# plt.ylabel('Amplitude')
#
# plt.subplot(2,2,3)
# plt.plot(first_rx_amp_csi_smoothed)
# plt.title('After Kalman Filter')
# plt.xlabel('Package')
# plt.ylabel('Amplitude')

plt.subplot(2,3,2)
plt.plot(first_rx_amp_csi_smoothed)
plt.title('After Wavelet Denoise')
plt.xlabel('Package')
plt.ylabel('Amplitude')


N = len(best_csi_remove_DC)
fs = 10
y1 = fft(best_csi_remove_DC)
y2 = fft(csi_remove_DC_plus)
f = np.linspace(0, fs/2, N//2, endpoint=False)

plt.subplot(2,3,3)
plt.plot(f, np.abs(y1[:N//2]))
# plt.plot(f, np.abs(y2[:N//2]))
plt.grid()
plt.xlabel('Hz')
plt.title('处理前频谱')


# 定义带通滤波器的参数
f2 = 0.7
f3 = 0.9
wc1 = 2 * f2 / fs
wc2 = 2 * f3 / fs
f1 = [0, wc1 - 0.05, wc1, wc2, wc2 + 0.05, 0.5]
A = [0, 1, 0]
weigh = [1, 1, 1]
M = 512

# 设计滤波器
b = remez(60, f1, A, weight=weigh)
w, h1 = freqz(b, worN=M)

# 绘制滤波器频率响应
f = np.linspace(0, fs/2, M//2, endpoint=False)
plt.subplot(2,3,4)
plt.plot(f, np.abs(h1[:M//2]))
plt.grid()
plt.title('带通')

# 过滤信号
x1 = lfilter(b, 1, best_csi_remove_DC)
x2 = lfilter(b, 1, csi_remove_DC_plus)
S1 = fft(x1)
S2 = fft(x2)

# 绘制处理后的频谱
f = np.linspace(0, fs/2, N//2, endpoint=False)
plt.subplot(2,3,5)
plt.plot(f, np.abs(S1[:N//2]))
# plt.plot(f, np.abs(S2[:N//2]))
plt.grid()
plt.xlabel('Hz')
plt.title('处理后频谱')

plt.show()

print(f'best_index:{best_index}')