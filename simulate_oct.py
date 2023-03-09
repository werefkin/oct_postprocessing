import numpy as np
from scipy.signal import chirp, spectrogram
import matplotlib.pyplot as plt
from scipy.fft import fft
import oct_lib as oct
from scipy.ndimage import gaussian_filter
import numpy.polynomial.polynomial as poly
from scipy.signal import savgol_filter


# In this code we neglect that the integral of the spectral interferogram should be approx. constant

N = 1024 # Samples number

# Noise levels
n_up = 0.25
n_low = 0.01

# Chirp coefficient
c_k = 10

# f0 frequency range (tilted plate shaped)
f_start_range1 = np.linspace(20,2,N,endpoint=True)
f_start_range2 = np.linspace(40,20,N,endpoint=True)
offset = 10 # aka plate thickness

# Generate chirped signal (for calibration vector)
time = np.linspace(0, 12, N) # time
start_freq = 10
fringes = chirp(time, f0=start_freq, f1=start_freq/c_k, t1=10, method='linear') #fringes
plt.plot(fringes)
plt.xlabel('Spectral ch.(f)')
plt.show()

# Generate spectral envelope
env = np.exp(-1/2*((time-6)/(1.5*1))**2)
plt.figure()
plt.plot(env)
plt.show()

# Generate some noise
# Generate the noise vector
noise = np.random.normal(0, 1, N)

# Multiply the noise by a 1/f filter to make the spectrum decrease with frequency
f = np.fft.fftfreq(N)
f_filter = np.abs(f)**(-0.1)
f_filter[0] = f_filter[1] # fix NaN in the 0th column

plt.figure()
plt.title('Spectral behaviour of noise')
plt.plot(f_filter)
plt.show()

noise_fft = np.fft.fft(noise, axis=0)
noise_filtered_fft = noise_fft#*f_filter

plt.figure()
plt.title('Spectragram of noise')
plt.plot(noise_filtered_fft)
plt.show()

noise_filtered = np.real(np.fft.ifft(noise_filtered_fft, axis=0))

# Normalize the noise between whatever
noise_min = np.min(noise_filtered)
noise_max = np.max(noise_filtered)
noise_range = noise_max - noise_min
noise_fin = (noise_filtered - noise_min) / noise_range * n_up + n_low

# plt.figure()
# plt.title('Noise')
# plt.plot(noise_fin)
# plt.show()

# Generate a spectral interferogram as a reference for amplitude based correction
spectral_interferogram = env*fringes + 2*env + noise_fin
plt.figure()
plt.plot(spectral_interferogram)
plt.show()



np.save('./correction/spectrum.npy', 2*env)
np.save('./correction/fringes.npy', spectral_interferogram)


# Generate B_scan data

# Empty array
b_data = np.zeros([N,N])

# Generate chirped data
for i in range(N):
    b_data[i,:] = 2*env + \
                 0.2*env* \
                 (chirp(time, f0=f_start_range1[i], f1=f_start_range1[i]/c_k, t1=10, method='linear') + 
                  chirp(time, f0=f_start_range2[i]+offset, f1=(f_start_range2[i]+offset)/c_k, t1=10, method='linear')) \

plt.figure()
plt.title('B-scan data without noise')
plt.imshow(b_data)
plt.show()

# Generate the noise 2D array
# 1/f noise
f = np.fft.fftfreq(N)
f_filter = np.abs(f)**(-0.3)
f_filter[0] = f_filter[1] # fix NaN in the 0th column


noise_b = np.random.normal(0, 1, b_data.shape)
noise_b_fft = np.fft.fft(noise_b, axis=0)

noise_b_filtered_fft = noise_b_fft#*f_filter[None, :]
noise_b_filtered = np.real(np.fft.ifft(noise_b_filtered_fft, axis=0))

# Normalize the noise between 0.05 and 10
noise_b_range = np.max(noise_b_filtered) - np.min(noise_b_filtered)
noise_b_fin = (noise_b_filtered - np.min(noise_b_filtered)) / noise_b_range * n_up + n_low

# I have a raw data from SD-OCT (sine like modulation) and I reconstruct the image by taken fourier transform. I do see some aliasing artefacts (like lines on different frequencies). how one can get rid of these artifacts in python?

# plt.figure()
# plt.title('Noise')
# plt.imshow(noise_b_fin)
# plt.show()

b_data += noise_b_fin

plt.figure()
plt.title('B-scan data with noise')
plt.imshow(b_data)
plt.show()



cal_vector = np.load('./correction/calibration_vector.npy')
boundaries = np.load('./correction/boundaries.npy')
cal_vector = savgol_filter(cal_vector, 200, 6) 
# cal_vector = gaussian_filter(cal_vector, sigma=4)

# Fit a function to the data using np.polyfit

# x = np.linspace(0, len(cal_vector), len(cal_vector))
# coefs = poly.polyfit(x, cal_vector, 70)
# ffit = poly.polyval(x, coefs)

# plt.figure()
# plt.plot(cal_vector)
# plt.plot(ffit)

# cal_vector = ffit
# plt.show()



speint = oct.remap_to_k(b_data,2*env,0,0,cal_vector,boundaries)

plt.figure()
plt.plot(b_data[100,:])
plt.plot(speint[100,:])

# Zero padding if needed
# speint = np.pad(speint, ((0,0),(1000,1000)), 'constant')
scan = oct.scanProcess(speint, typ='n', px=np.shape(speint)[1], a=0, Std=100)

bscan = np.transpose(scan[:,int(np.shape(speint)[1]/2):])

plt.figure()
plt.imshow(20*np.log10(bscan),cmap='gray')
plt.show()

# plt.figure()
# plt.title('Some A-scan')
# plt.plot(bscan[:,100])
# plt.show()


