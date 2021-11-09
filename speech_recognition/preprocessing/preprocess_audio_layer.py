import string
import sys
import pathlib
import numpy as np
import tensorflow as tf
sys.path.append(str(pathlib.Path(__file__).parents[2]))
import speech_recognition.config as config

class PreprocessAudioLayer(tf.keras.layers.Layer):

  def __init__(self, sample_rate=44100, frame_size=0.025, frame_stride=0.01):
    
    self.sample_rate = sample_rate
    self.frame_size = frame_size
    self.frame_stride = frame_stride

  @staticmethod
  def hz_to_mel(hz):

    return 1125 * np.log(1 + hz/700)

  @staticmethod
  def mel_to_hz(mel):

    return 700 * (np.exp(mel / 1125) - 1)

  def boost(self, alpha=0.95):
    """
    Applying a first order high-pass filter to boost the higher frequencies
    """

    data = np.empty_like(self.data)
    data[0] = self.data[0]
    data[1:] = self.data[1:] - alpha * self.data[:-1]
    return data

  def rolling_window(self):
    
    shape = self.data.shape[:-1] + (int(self.data.shape[-1] - self.window_size + 1), int(self.window_size))
    strides = self.data.strides + (self.data.strides[-1],)
    return np.lib.stride_tricks.as_strided(self.data, shape=shape, strides=strides)[::self.shift_size]

  def smoothing_filter(self, filter=np.hamming):

    w = filter(self.window_size)
    return self.data * w

  def compute_nfft(self):

    nfft = 1
    while nfft < self.window_size:
        nfft *= 2
    return nfft

  def spectrum(self, nfft):
    """
    returns: given an NxM input matrix, output will be Nx(NFFT/2+1). Each row will be the magnitude spectrum of the corresponding frame.
    """

    fft = np.fft.rfft(self.data, n=nfft, axis=1)
    spectrum = fft * fft / self.window_size
    return spectrum

  def mel_binning(self, spectrum, nfft, lower_freq=300, upper_freq=8000, num_filt=26, norm=True):
    """
    Multiply each filter bank with the power spectrum and add up the coefficents.
    This will give us the amount of 'energy' is each filter bank 
    """
    low_mel = PreprocessAudio.hz_to_mel(lower_freq)
    high_mel = PreprocessAudio.hz_to_mel(upper_freq)
    # get equally spaced points in mel scale
    mel_points = np.linspace(low_mel, high_mel, num_filt + 2)
    hz_points = PreprocessAudio.mel_to_hz(mel_points)
    # round hz_points to the nearest fft bin
    bins = np.floor((nfft + 1) * hz_points / self.sample_rate)

    fbank = np.zeros((num_filt, int(np.floor(nfft / 2 + 1))))
    for m in range(1, num_filt + 1):
        f_m_minus = int(bins[m - 1])   # left
        f_m = int(bins[m])             # center
        f_m_plus = int(bins[m + 1])    # right

        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bins[m - 1]) / (bins[m] - bins[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bins[m + 1] - k) / (bins[m + 1] - bins[m])

    filter_banks = np.dot(spectrum, fbank.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)  # Numerical Stability

    if norm:
        filter_banks -= (np.mean(filter_banks, axis=0) + 1e-8)

    filter_banks = 20 * np.log10(filter_banks)
    return filter_banks

  def mfcc(self, filter_banks, num_ceps=12, norm=True):

    mfcc = pack.dct(filter_banks, type=2, axis=1, norm='ortho')[:, 1 : (num_ceps + 1)] # Keep 2-13
    if norm:
        mfcc -= (np.mean(mfcc, axis=0) + 1e-8)
    return mfcc

  def energy(self, spectrum):
    """
    E = sum(x[t]^2) --> sum over each frame
    """
    # total energy in each frame
    energy = np.sum(spectrum, 1)
    # if energy is zero, we get problems with log
    energy = np.where(energy == 0, np.finfo(float).eps,energy)
    return energy

  def lifter(self, cepstra, L=22):
    """
    Apply a cepstral lifter the the matrix of cepstra. This has the effect of increasing the
    magnitude of the high frequency DCT coeffs.
    """
    if L > 0:
        nframes, ncoeff = np.shape(cepstra)
        n = np.arange(ncoeff)
        lift = 1 + (L/2.)*np.sin(np.pi*n/L)
        return lift*cepstra
    else:
        # values of L <= 0, do nothing
        return cepstra
  
  def delta(self, features, N=2):
    """
    Compute delta features from a feature vector sequence.
    """

    num_rows = len(features)
    denominator = 2 * sum([i**2 for i in range(1, N+1)])
    delta_feat = np.empty_like(features)
    padded = np.pad(features, ((N, N), (0, 0)), mode='edge')   # padded version of feat
    for t in range(num_rows):
        delta_feat[t] = np.dot(np.arange(-N, N+1), padded[t : t+2*N+1]) / denominator   # [t : t+2*N+1] == [(N+t)-N : (N+t)+N+1]
    return delta_feat

  def call(self, audio_files):

    for audio_file in audio_files:
      self.number_of_samples = len(audio_file)
      # number of samples per window
      self.window_size = int(self.sample_rate * self.frame_size)
      # number of samples to skip per stride
      self.shift_size = int(self.sample_rate * self.frame_stride)
    return self.get_filter_banks() # add more preprocessing options