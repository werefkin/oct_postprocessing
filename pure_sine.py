import numpy as np
import matplotlib.pyplot as plt

# Define the x and y dimensions
x_len = 2048
y_len = 1024

# Define the frequency parameters
freq1 = 0.03  # Frequency of the first sine wave
freq2 = 0.2  # Frequency of the second sine wave

# frequency step

n = 0.00001

# Initialize the data array
data = np.zeros((x_len, y_len))

# Loop through the chirp frequencies and create the 2D array with varying frequency
state = 0
y = np.arange(y_len)
env = np.exp(-1/2*((y-int(y_len/2))/(100))**2)

for i in range(x_len):
    data[i, :] = env+env*np.sin(2*np.pi*(freq1+state)*y)+ env*2*np.sin(2*np.pi*(freq2+state)*y) + np.random.normal(0, 0.01, y_len)
    state += n
    
# Take the FFT along the y-axis
data_fft = np.fft.fftshift(np.fft.fft(data, axis=1),axes=1)


speint = data - env

# Zero padding if needed
data_fft = oct.scanProcess(speint, typ='n', px=np.shape(speint)[1], a=0, Std=100)

# Plot the results
plt.imshow(20*np.log10((np.abs(np.transpose(data_fft)))), aspect='auto', cmap='jet')

plt.colorbar()
plt.show()
