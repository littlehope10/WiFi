from pykalman import KalmanFilter
import pywt
import numpy as np
from scipy.signal import cheby1, filtfilt, hilbert
from scipy.fft import fft

# 绝对中位差（MAD）异常值剔除
def mad_remove(csi_abs, gama):
    median = np.median(csi_abs, axis=0)
    difference = csi_abs - median[np.newaxis, :]
    MAD = np.median(np.abs(difference), axis=0)

    for i in range(0, 30):
        minimum = median[i] - gama * MAD[i]
        maximum = median[i] + gama * MAD[i]
        for j in range(0, csi_abs.shape[0]):
            if csi_abs[j][i] < minimum or csi_abs[j][i] > maximum:
                csi_abs[j][i] = median[i]

    return csi_abs

# 卡尔曼平滑方法
def kalman_smooth(csi_data, mean=0, dim_obs=1, iter=10):
    kf = KalmanFilter(initial_state_mean=mean, n_dim_obs=dim_obs)
    kf = kf.em(csi_data, n_iter=iter)
    smooth_data, _ = kf.smooth(csi_data)
    return smooth_data

def apply_kalman_smooth(csi_abs_data):
    amp_csi_smoothed = np.apply_along_axis(kalman_smooth, axis=0, arr=csi_abs_data)
    amp_csi_smoothed = amp_csi_smoothed.squeeze()
    return amp_csi_smoothed

# 离散小波变换去噪
def wavelet_denoise(csi_data, wavelet='db1', level=1):
    coeffs = pywt.wavedec(csi_data, wavelet, level=level)
    threshold = np.median(np.abs(coeffs[-1]))
    coeffs_denoised = [pywt.threshold(c, value=threshold) for c in coeffs]
    return pywt.waverec(coeffs_denoised, wavelet)

# 带通滤波器
def chebyshev_filter(data, lowcut, highcut, fs, order=4, ripple=0.5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = cheby1(order, ripple, [low, high], btype='band')
    filtered_data = filtfilt(b, a, data)
    return filtered_data

# 滤波和FFT
def filter_and_fft(data, fs=1000, lowcut=0.3, highcut=0.5):
    filtered_data = chebyshev_filter(data, lowcut, highcut, fs)
    fft_result = fft(filtered_data)
    return filtered_data, fft_result

# 希尔伯特变换
def hilbert_transform(csi_data):
    analytic_signal = hilbert(csi_data, axis=0)
    hilbert_abs = np.abs(analytic_signal)
    hilbert_phase = np.angle(analytic_signal)
    return hilbert_abs, hilbert_phase
