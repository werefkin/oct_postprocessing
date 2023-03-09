
"""
LIBRARY FUNCTIONS:
    
    1. "remap_to_k" converts spectral interferograms measured in wavelengths (with subtraction of the reference)
    in wavenumbers 
    performing interpolation to get equidistant sampling in k-space 
    OUTPUT is spectral interferograms in wavenumbers
    in this code wavelengths assumed to be linear
    
    2. scanProcess performs postprocessing of the signal in following order:
    (Hilbert transform --> Analytic signal --> Envelope --> High Pass Filter --> Gaussian window --> FFT)
    
    3. depthAxis calculates depth axis you have after FFT according to number of pixels and spectrum bandwidth
    
@author: r.zor
"""

import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import hilbert


def remap_to_k(data,ref,lmin,lmax,cal_vector=None,boundaries=None):  
    data=data-ref
    if cal_vector is not None and boundaries is not None:
        
        print("Calibration-vector based")
        N=np.shape(data)[1]
        remap_interp_func = interp1d(cal_vector,data[:,boundaries[0]:boundaries[-1]],axis=1, kind="quadratic",fill_value="extrapolate")
        spectral_interferograms = remap_interp_func(np.linspace(boundaries[0],boundaries[-1],N,endpoint=True)) 
    else:
        print("Standard processing")
        N=np.shape(data)[1]
        rang=np.linspace(lmin,lmax,N,endpoint=True) 
        wnrang=1/rang
        wn_corr=np.linspace(wnrang[0],wnrang[-1],N)
        remap_interp_func = interp1d(wnrang,data, axis=1, kind="quadratic",fill_value="extrapolate")
        spectral_interferograms = remap_interp_func(wn_corr)
    return(spectral_interferograms)

  
    
def scanProcess(swn,typ='n', N=20, Wn=0.01, dpmin=0, dpmax=45, px=510, a=0, Std=25, zp=0, pads=0 ):
    
    """
    swn - spectrums in k-space
    typ - 's' - substraction of the envelope (less noise), typ='d' is division
    
    Std - standard deviation
    a - Position of the gaussian window
    px - number of pixels
    N  - The order of the envelope freq filter.
    Wn : Nyqust freqs from o to 1 (see scipy.signal.butter)
    
    dpmin and dpmax: interval where the signal should be zero to remove very high values after division
    
    RETURNS OCT B-SCAN

    """
    from scipy import ndimage
    from scipy import signal
    from scipy.signal import sosfiltfilt, butter
    
    #CREATE GAUSSIAN WINDOW
    x=np.linspace(-(px-1)/2,(px-1)/2,px)
    gw = np.exp(-1/2*((x-a)/(4*Std))**2);
    
    print('Gaussian width:',Std)
    print('Gaussian position:',a)

    #CREATE ARRAYS FOR DATA  --- --- needed you you want to organize output of intermediate data
    postprocess=np.zeros([len(swn),len(swn[0])])
    scan=np.zeros([len(swn),len(swn[0])])
    amplitude_envelope=np.zeros([len(swn),len(swn[0])])

    #FILTER
    #sos = butter(20, 0.4, output='sos')
    sos = butter(N, Wn, output='sos')


    #POSTPROCESSING
    for i in range(0, len(swn)): 
        
        signal=swn[i,:]
        
        
        analytic_signal = hilbert(signal)                                       #Hilbert transform
        amplitude_envelope[i,:] = np.abs(analytic_signal)                       #Envelope
        # amplitude_envelope[i,:] = sosfiltfilt(sos, amplitude_envelope[i,:])     #Filtering of the envelope
        
        if typ=='d':
            postprocess[i,:]=(signal/amplitude_envelope[i,:])*gw                    #Applying of the filter and envelope substraction or division
        elif typ=='s':
            postprocess[i,:]=(signal-amplitude_envelope[i,:])*gw
        elif typ=='n':
            postprocess[i,:]=signal*gw
        scan[i,:]=np.abs(np.fft.fftshift(np.fft.fft(postprocess[i,:])))
    #Save data in to array
    return scan



    
def norma(d,norma):
    d -= np.min(d)
    if norma==1:
        d = d/np.max(d)
    return d


