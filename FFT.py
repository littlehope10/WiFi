import csiread
import numpy as np
import matplotlib.pyplot as plt
import pywt
from pyts.image import RecurrencePlot
from scipy.signal import remez, freqz, lfilter, medfilt, butter, filtfilt
from scipy.fft import fft
from utils.methods import hampel_remove, apply_pca, wavelet_denoise
from utils.jsoncsi import JsonCSI
from csi_read import read, cal_abs

csi_file = '11_28/c1_1_hor'
# csi_file = 'csi_p1_10_hor'


csi_data = read(csi_file, False)

package = csi_data.shape[0]
tx = csi_data.shape[1]
rx = csi_data.shape[2]
subcarrier = csi_data.shape[3]


csi1 = csi_data[:,0,0,:]
csi2 = csi_data[:,1,2,:]

csi = np.abs(csi1)
csi = medfilt(csi, kernel_size=7)
csi = hampel_remove(csi, 11, 3)
csi = wavelet_denoise(csi, 'sym3', 20)
csi = apply_pca(csi, 0)



# 创建带通滤波器
# def butter_bandpass(lowcut, highcut, fs, order=4):
#     nyquist = 0.5 * fs
#     low = lowcut / nyquist
#     high = highcut / nyquist
#     b, a = butter(order, [low, high], btype='band')
#     return b, a

# t = np.linspace(0, 2, 300)

# plt.figure(figsize=(12, 5))
#
# for i in range(8):
#     b, a = butter_bandpass(i * 10 if i != 0 else 1, (i + 1) * 10 if i != 7 else 74, 150, order=4)
#     filtered_signal = filtfilt(b, a, csi1[100:400, 0])
#     plt.subplot(4, 2, i + 1)
#     plt.plot(t, filtered_signal)
#     plt.xlabel(f"{i * 10} - {(i + 1) * 10} Hz")
#     plt.ylabel("ED")

csi1 = csi[1100:1300]

N = len(csi1)
fs = 150
y1 = fft(csi1)
f = np.linspace(0, fs/2, N//2, endpoint=False)

plt.plot(f, np.abs(y1[:N//2]))
# plt.plot(f, np.abs(y2[:N//2]))
plt.grid()
plt.xlabel('Hz')
plt.title('After FFT')
plt.show()

wp = pywt.WaveletPacket(data=csi1, wavelet='dmey', mode='symmetric', maxlevel=3)


rex3 = []
for i, node in enumerate(wp.get_level(3, 'freq')):  # 第3层有8个节点
    rex3.append(node.reconstruct())

rex3 = np.array(rex3).T  # 转换为列矩阵

fs = 150

# 绘制第3层各节点的频谱
plt.figure(figsize=(12, 6))
for i in range(8):
    x_sign = rex3[:, i]
    N = len(x_sign)
    signalFFT = np.abs(np.fft.fft(x_sign, N))  # 计算FFT
    Y = 2 * signalFFT / N  # 真实幅值
    f = np.linspace(fs / 8 * i, fs / 8 * (i + 1), N // 2 + 1)  # 频率轴

    # 绘制子图
    plt.subplot(2, 4, i + 1)
    plt.plot(f, Y[:N // 2 + 1])
    plt.ylabel('Amplitude')
    plt.xlabel('Frequency (Hz)')
    plt.grid(True)
    # plt.axis([0, 50, 0, 0.03])
    plt.title(f'{i} ')

plt.tight_layout()
plt.show()
