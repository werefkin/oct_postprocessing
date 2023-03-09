# -*- coding: utf-8 -*-
"""
Created on Tue May  3 11:21:59 2022

@author: r.zor
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from scipy.interpolate import interp1d
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d

# Params
PV = True # peak valleys?

# Load data
spectrum = np.load('spectrum.npy')
interference = np.load('fringes.npy')

N = len(spectrum) # Length

# Pure fringes
fri = interference-spectrum

# Detect peaks and make linear space (one should adjust filter and peak detection parameters)
signal_nonlin_wl = gaussian_filter1d(fri, 2) #filter noise in fringe detection
# peaks, _ = find_peaks(signal_nonlin_wl, height=0.01,distance=10)


if PV == True:
        
    # Peaks and valleys detection

    grad_signal = np.gradient(signal_nonlin_wl) #remove baseline

    peaks_p, _ = find_peaks(grad_signal, height=0.05,distance=10)
    peaks_n, _ = find_peaks(-grad_signal, height=0.05,distance=10)


    x = peaks_p
    y = peaks_n
    len_x = len(x)
    len_y = len(y)


    i = 0
    j = 0
    out = np.zeros(len(peaks_n)+len(peaks_p))

    # Fill the output array by alternating between x and y values
    for k in range(len(peaks_n)+len(peaks_p)):
        if k % 2 == 0:
            if i < len_x:
                out[k] = x[i]
                i += 1
            else:
                out[k] = y[j]
                j += 1
        else:
            if j < len_y:
                out[k] = y[j]
                j += 1
            else:
                out[k] = x[i]
                i += 1

    plt.figure()
    plt.plot(out)
    plt.show()

    peaks = out.astype(int)
    plt.figure()
    xx=np.linspace(0,N,N,endpoint=True)
    plt.plot(xx,grad_signal)
    plt.plot(xx[peaks],grad_signal[peaks],'*')

if PV != True:
    peaks, _ = find_peaks(signal_nonlin_wl, height=0.22,distance=10)
    plt.figure()
    xx=np.linspace(0,N,N,endpoint=True)
    plt.plot(xx,signal_nonlin_wl)
    plt.plot(xx[peaks],signal_nonlin_wl[peaks],'*')


linpeaks=np.linspace(peaks[0],peaks[-1],len(peaks))


peak_pos_cor_interp_func = interp1d(peaks, linpeaks,  kind="quadratic",fill_value="extrapolate")
siglength=np.linspace(peaks[0],peaks[-1],len(signal_nonlin_wl[peaks[0]:peaks[-1]]),endpoint=True) 


peak_corrected_range = peak_pos_cor_interp_func(siglength)

uncorrected=abs(np.fft.fftshift(np.fft.fft(fri)))

remap_interp_func = interp1d(peak_corrected_range,signal_nonlin_wl[peaks[0]:peaks[-1]], kind="quadratic",fill_value="extrapolate")
signal_2nonlin_wl = remap_interp_func(np.linspace(peaks[0],peaks[-1],len(fri),endpoint=True)) 

corrected=abs(np.fft.fftshift(np.fft.fft(signal_2nonlin_wl)))

np.save('calibration_vector.npy',peak_corrected_range)
boundaries=np.array([peaks[0],peaks[-1]])
np.save('boundaries.npy',boundaries)

plt.figure()
plt.plot(np.log10(uncorrected))
plt.plot(np.log10(corrected))
plt.show()

