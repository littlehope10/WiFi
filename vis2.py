import csiread
import numpy as np
import matplotlib.pyplot as plt
from pyts.image import RecurrencePlot
from scipy.signal import remez, freqz, lfilter
from scipy.fft import fft
from utils.methods import mad_remove, apply_kalman_smooth, wavelet_denoise
from utils.jsoncsi import JsonCSI
from csi_read import read, cal_abs






csi_np = read('csi', True)

first_rx_amp_csi = cal_abs(csi_np, 1, 3)

plt.subplot(2,2,1)
plt.plot(first_rx_amp_csi)
plt.title('1st TX 1st RX Static CSI Magnitude')
plt.xlabel('Package')
plt.ylabel('Amplitude')
plt.show()