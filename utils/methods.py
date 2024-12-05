from pykalman import KalmanFilter
import pywt
import numpy as np
from scipy.signal import cheby1, filtfilt
from scipy.fft import fft
import math

# 第二步，绝对中位差（MAD）异常值剔除
# 剔除公式：[M - 3 * MAD， M + 3 * MAD]超过此范围的值替换为中位数M
def hampel_remove(csi_abs, window_size, sigma):
    size = window_size // 2
    for i in range(csi_abs.shape[0]):
        start = i - size if i >= size else 0
        end = i + size + 1 if i + size + 1 > csi_abs.shape[0] else csi_abs.shape[0]
        window_data = csi_abs[start:end, :]

        median = np.median(window_data, axis=0)
        std = np.std(window_data, axis=0)

        for carrier in range(0, 30):
            if abs(csi_abs[i, carrier] - median[carrier]) > sigma * std[carrier]:
                csi_abs[i, carrier] = median[carrier]

    return csi_abs

# 卡尔曼平滑方法
def kalman_smooth(csi_data, mean=0, dim_obs=1, iter=10):
    kf = KalmanFilter(initial_state_mean=mean, n_dim_obs=dim_obs)
    kf = kf.em(csi_data, n_iter=iter)
    smooth_data,_ = kf.smooth(csi_data)
    return smooth_data

# 应用卡尔曼平滑方法
def apply_kalman_smooth(csi_abs_data):
    amp_csi_smoothed = np.apply_along_axis(kalman_smooth, axis=0, arr=csi_abs_data)
    amp_csi_smoothed = amp_csi_smoothed.squeeze()
    return amp_csi_smoothed

# 离散小波变换，默认db小波函数
def wavelet_denoise(csi_data, wavelet='db1', level=1, mode='soft'):
    csi_data = csi_data.T
    w = pywt.Wavelet(wavelet)
    carrier = []
    hybrid = 1
    for i in range(csi_data.shape[0]):
        data = csi_data[i].tolist()
        coeffs = pywt.wavedec(data, w, level=level)
        # heursure阈值
        sigma = np.std(coeffs[-1])
        threshold = sigma * np.sqrt(2 * np.log(len(data)))
        denoised_coeffs = [coeffs[0]]
        for c in coeffs[1:]:
            # 硬阈值
            denoised_coeffs.append(pywt.threshold(c, threshold, mode=mode))
            # hard_coeff = np.where(np.abs(c) > threshold, c, 0)
            # soft_coeff = np.sign(hard_coeff) * np.maximum(np.abs(hard_coeff) - hybrid * hard_coeff, 0)
            # denoised_coeffs.append(soft_coeff)
        recoeffs = pywt.waverec(denoised_coeffs, wavelet)
        carrier.append(recoeffs)
    return np.array(carrier).T

def apply_pca(csi_data, pca_index = 0):
    Z = csi_data - np.mean(csi_data, axis=0)
    C = np.cov(Z, rowvar=False)

    eig_values, eig_vectors = np.linalg.eig(C)

    sorted_index = np.argsort(eig_values)[::-1]
    eig_values = eig_values[sorted_index]
    eig_vectors = eig_vectors[:, sorted_index]

    pca = eig_vectors[:, pca_index]

    return Z @ pca

def sgn(num):
    if num > 0.0:
        return 1.0
    elif num == 0.0:
        return 0.0
    else:
        return -1.0

def wavelet_noising(new_df):
    # data = new_df
    # data = data.values.T.tolist()  # 将np.ndarray()转为列表
    carrier = []
    for i in range(new_df.shape[1]):
        data = new_df[:,i].T.tolist()
        w = pywt.Wavelet('sym3')#选择sym8小波基
        [ca5, cd5, cd4, cd3, cd2, cd1] = pywt.wavedec(data, w, level=5)  # 5层小波分解

        length1 = len(cd1)
        length0 = len(data)

        Cd1 = np.array(cd1)
        abs_cd1 = np.abs(Cd1)
        median_cd1 = np.median(abs_cd1)

        sigma = (1.0 / 0.6745) * median_cd1
        lamda = sigma * math.sqrt(2.0 * math.log(float(length0 ), math.e))#固定阈值计算
        usecoeffs = []
        usecoeffs.append(ca5)  # 向列表末尾添加对象

        #软硬阈值折中的方法
        a = 0.5

        for k in range(length1):
            if (abs(cd1[k]) >= lamda):
                cd1[k] = sgn(cd1[k]) * (abs(cd1[k]) - a * lamda)
            else:
                cd1[k] = 0.0

        length2 = len(cd2)
        for k in range(length2):
            if (abs(cd2[k]) >= lamda):
                cd2[k] = sgn(cd2[k]) * (abs(cd2[k]) - a * lamda)
            else:
                cd2[k] = 0.0

        length3 = len(cd3)
        for k in range(length3):
            if (abs(cd3[k]) >= lamda):
                cd3[k] = sgn(cd3[k]) * (abs(cd3[k]) - a * lamda)
            else:
                cd3[k] = 0.0

        length4 = len(cd4)
        for k in range(length4):
            if (abs(cd4[k]) >= lamda):
                cd4[k] = sgn(cd4[k]) * (abs(cd4[k]) - a * lamda)
            else:
                cd4[k] = 0.0

        length5 = len(cd5)
        for k in range(length5):
            if (abs(cd5[k]) >= lamda):
                cd5[k] = sgn(cd5[k]) * (abs(cd5[k]) - a * lamda)
            else:
                cd5[k] = 0.0

        usecoeffs.append(cd5)
        usecoeffs.append(cd4)
        usecoeffs.append(cd3)
        usecoeffs.append(cd2)
        usecoeffs.append(cd1)
        recoeffs = pywt.waverec(usecoeffs, w)#信号重构
        carrier.append(recoeffs)
    return np.array(carrier).T
