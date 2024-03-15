import numpy as np
from scipy.signal import butter, filtfilt 

def low_pass(X):
    # Define the low-pass filter parameters
    cutoff_frequency = 40  # Cut-off frequency in Hz
    sampling_rate = 125  # Sampling rate in Hz (adjust based on your data)

    # Design a Butterworth low-pass filter
    order = 3  # Filter order
    b, a = butter(N=order, Wn=cutoff_frequency / (0.5 * sampling_rate), btype='low', analog=False, output='ba')

    X_lpf = np.zeros(X.shape)
    # Apply the filter to the data
    for i in range(X.shape[0]):
        for j in range(X.shape[2]):
            filtered_data = filtfilt(b, a, X[i,0,j,:])
            X_lpf[i,0,j,:] = filtered_data

    return X_lpf

def high_pass(X):
    # Define the high-pass filter parameters
    cutoff_frequency = 4  # Cut-off frequency in Hz
    sampling_rate = 125  # Sampling rate in Hz (adjust based on your data)

    # Design a Butterworth high-pass filter
    order = 3  # Filter order
    b, a = butter(N=order, Wn=cutoff_frequency / (0.5 * sampling_rate), btype='high', analog=False, output='ba')

    X_hpf = np.zeros(X.shape)
    # Apply the filter to the data
    for i in range(X.shape[0]):
        for j in range(X.shape[2]):
            filtered_data = filtfilt(b, a, X[i,0,j,:])
            X_hpf[i,0,j,:] = filtered_data

    return X_hpf

def band_pass(X):
    # Define the band-pass filter parameters
    cutoff_frequency = np.array([4, 40])  # Cut-off frequency in Hz
    sampling_rate = 125  # Sampling rate in Hz (adjust based on your data)

    # Design a Butterworth band-pass filter
    order = 3  # Filter order
    b, a = butter(N=order, Wn=cutoff_frequency / (0.5 * sampling_rate), btype='band', analog=False, output='ba')

    X_bpf = np.zeros(X.shape)
    # Apply the filter to the data
    for i in range(X.shape[0]):
        for j in range(X.shape[2]):
            filtered_data = filtfilt(b, a, X[i,0,j,:])
            X_bpf[i,0,j,:] = filtered_data
    
    return X_bpf