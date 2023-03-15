import numpy as np
from scipy.signal import chirp, spectrogram
import matplotlib.pyplot as plt
from scipy.fft import fft
import oct_lib as oct
from scipy.ndimage import gaussian_filter
import numpy.polynomial.polynomial as poly
from scipy.signal import savgol_filter



# SAVE new reference pattern

SAVE = False

N = 1024 # Samples number

# Noise levels
n_up = 0.25
n_low = 0.01

# Chirp coefficient
c_k = 10

# f0 frequency range (tilted plate shaped)
f_start_range1 = np.linspace(20,2,N,endpoint=True)
f_start_range2 = np.linspace(40,20,N,endpoint=True)
offset = 0.5 # aka plate thickness

# Generate chirped signal (for calibration vector); the chirp is linear just for simplicity; one can add other nonlinearities to compensate for
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
noise_fin = np.random.normal(0, 0.03, N)

# plt.figure()
# plt.title('Noise')
# plt.plot(noise_fin)
# plt.show()

# Generate a spectral interferogram as a reference for amplitude based correction
spectral_interferogram = env*fringes + 2*env + noise_fin
plt.figure()
plt.plot(spectral_interferogram)
plt.show()


# save it
if SAVE == True:
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

noise_b = np.random.normal(0, 0.01, b_data.shape)


# Set the size of the array and the exponent of the power-law filter
alpha = -0.4
# Generate the 1/f filter
freqs = np.fft.fftfreq(N)
freqs_shifted = np.fft.fftshift(freqs)
filt = np.ones_like(freqs_shifted)
filt[N//2+1:] = 0
ramp = np.abs(np.linspace(-1, 1, N))
filt *= ramp ** alpha
filt[N//2] = 0

# Symmetrize the filter
f_filter = filt + filt[::-1]

# Normalize the filter
f_filter /= np.max(f_filter)

f_filter = np.fft.fftshift(f_filter)
# Plot the filter
plt.plot(f_filter)
plt.show()


noise_b = np.random.normal(0, 2, b_data.shape)
noise_b_fft = abs(np.fft.fft(noise_b, axis=0))

noise_b_filtered_fft = noise_b_fft*f_filter[:, None]
noise_b_filtered = np.real(np.fft.ifft(noise_b_filtered_fft, axis=0))


# plt.figure()
# plt.title('Noise')
# plt.imshow(noise_b_fin)
# plt.show()

b_data += np.transpose(noise_b_filtered)/2

plt.figure()
plt.title('B-scan data with noise')
plt.imshow(b_data)
plt.show()



cal_vector = np.load('./correction/calibration_vector.npy')
boundaries = np.load('./correction/boundaries.npy')
# cal_vector = savgol_filter(cal_vector, 200, 2) 



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


