import csiread
import numpy as np
import matplotlib.pyplot as plt
from pyts.image import RecurrencePlot
from scipy.signal import remez, freqz, lfilter
from scipy.fft import fft
from utils.methods import mad_remove, apply_kalman_smooth, wavelet_denoise, hilbert_transform
from utils.jsoncsi import JsonCSI
from csi_read import read, cal_abs

# 加载CSI数据
csi_np = read('csi-static', True)

# 数据维度
packet_number = csi_np.shape[0]
tx_number = csi_np.shape[1]
rx_number = csi_np.shape[2]
subcarrier_number = csi_np.shape[3]

# 提取第1发射天线和第1接收天线的振幅数据
first_rx_amp_csi = cal_abs(csi_np, 1, 1)

# 第一步：MAD异常值剔除
first_rx_amp_csi = mad_remove(first_rx_amp_csi, gama=3)

# 第二步：小波去噪
first_rx_amp_csi_denoised = np.apply_along_axis(wavelet_denoise, axis=0, arr=first_rx_amp_csi)

# 第三步：卡尔曼平滑
first_rx_amp_csi_smoothed = apply_kalman_smooth(first_rx_amp_csi_denoised)

# 第四步：希尔伯特变换
hilbert_abs, hilbert_phase = hilbert_transform(first_rx_amp_csi_smoothed)

# 选取最佳子载波（基于复现图的相似性）
jsoncsi = JsonCSI()
first_rx_amp_csi_smoothed = jsoncsi.load_csi_data('csi-static', 1, 1, True)

recurrence_plot = RecurrencePlot(threshold='point', percentage=20)
best_index = None
max_similarity = 0
for i, subcarrier in enumerate(first_rx_amp_csi_smoothed.T):
    rp_image = recurrence_plot.fit_transform(subcarrier.reshape(1, -1))[0]
    similarity = np.sum(rp_image)  # 计算图中点的总和表示相似度
    if similarity > max_similarity:
        max_similarity = similarity
        best_index = i

# 去直流成分（DC）
csi_remove_DC = np.zeros_like(first_rx_amp_csi_smoothed)
for i in range(0, 30):
    carrier = first_rx_amp_csi_smoothed[:, i]
    row_mean = np.mean(carrier, axis=0, keepdims=True)
    row_mean = np.tile(row_mean, carrier.shape[0])
    carrier_remove_DC = carrier - row_mean
    csi_remove_DC[:, i] = carrier_remove_DC

best_csi = first_rx_amp_csi_smoothed[:, best_index]
best_csi_remove_DC = csi_remove_DC[:, best_index]
csi_remove_DC_plus = np.sum(csi_remove_DC, axis=1)

# 绘图：原始、去直流、频谱等
plt.figure(figsize=(12, 8))

# 绘制最佳子载波的振幅和去直流结果
plt.subplot(2, 3, 1)
plt.plot(best_csi, label="Smoothed Amplitude")
plt.plot(best_csi_remove_DC, label="DC Removed")
plt.plot(csi_remove_DC_plus, label="Sum of All")
plt.xlabel('Package')
plt.ylabel('Amplitude')
plt.title(f'Recurrence Plot of Best Subcarrier (Index {best_index})')
plt.legend()

# 绘制去噪数据的完整图像
plt.subplot(2, 3, 2)
plt.plot(first_rx_amp_csi_smoothed)
plt.title('After Wavelet Denoise')
plt.xlabel('Package')
plt.ylabel('Amplitude')

# FFT频谱分析（去直流后）
N = len(best_csi_remove_DC)
fs = 10
y1 = fft(best_csi_remove_DC)
y2 = fft(csi_remove_DC_plus)
f = np.linspace(0, fs / 2, N // 2, endpoint=False)
plt.subplot(2, 3, 3)
plt.plot(f, np.abs(y1[:N // 2]))
plt.grid()
plt.xlabel('Hz')
plt.title('FFT Before Filtering')

# 定义带通滤波器参数
f2, f3 = 0.7, 0.9
wc1, wc2 = 2 * f2 / fs, 2 * f3 / fs
f1 = [0, wc1 - 0.05, wc1, wc2, wc2 + 0.05, 0.5]
A = [0, 1, 0]
weight = [1, 1, 1]
M = 512

# 设计滤波器
b = remez(60, f1, A, weight=weight)
w, h1 = freqz(b, worN=M)

# 绘制滤波器频率响应
f = np.linspace(0, fs / 2, M // 2, endpoint=False)
plt.subplot(2, 3, 4)
plt.plot(f, np.abs(h1[:M // 2]))
plt.grid()
plt.title('Band-pass Filter Response')

# 滤波后的结果
x1 = lfilter(b, 1, best_csi_remove_DC)
x2 = lfilter(b, 1, csi_remove_DC_plus)
S1 = fft(x1)
S2 = fft(x2)

# 绘制滤波后的频谱
f = np.linspace(0, fs / 2, N // 2, endpoint=False)
plt.subplot(2, 3, 5)
plt.plot(f, np.abs(S1[:N // 2]))
plt.grid()
plt.xlabel('Hz')
plt.title('FFT After Filtering')

# 绘制希尔伯特变换的包络
plt.subplot(2, 3, 6)
plt.plot(hilbert_abs[:, best_index], label="Hilbert Envelope")
plt.xlabel('Package')
plt.ylabel('Amplitude')
plt.title(f'Hilbert Transform (Envelope)')
plt.legend()

plt.tight_layout()
plt.show()

print(f'Best Index: {best_index}')
