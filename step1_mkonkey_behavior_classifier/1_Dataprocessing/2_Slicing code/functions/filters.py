from scipy.signal import butter, filtfilt, iirnotch
import numpy as np

def notch_filter(data, fs, freq):
    """60/120 Hz notch filter.

    data: (T, Ch) numpy array
    freq: notch frequency (e.g., 60, 120)
    """
    b, a = iirnotch(w0=freq/(fs/2), Q=30)
    return filtfilt(b, a, data, axis=0)

def bandpass_filter(data, fs, low=1.0, high=120.0, order=4):
    nyq = fs / 2.0
    b, a = butter(order, [low/nyq, high/nyq], btype='band')
    return filtfilt(b, a, data, axis=0)

def car(data):
    """Common Average Reference.

    data: (T, Ch)
    """
    return data - np.mean(data, axis=0, keepdims=True)
