import mne
import numpy as np

class MNE_Filter:
    def __init__(self, sfreq, l_freq, h_freq):
        """
        MNE Filter for preprocessing signals.

        :param sfreq: Sampling frequency of the input data (Hz).
        :param l_freq: Low cutoff frequency (Hz).
        :param h_freq: High cutoff frequency (Hz).
        """
        self.sfreq = sfreq
        self.l_freq = l_freq
        self.h_freq = h_freq
    
    def transpose(self, x):
        """
        Transpose the input array.

        :param x: Input array of shape (a, b).
        :return: Transposed array of shape (b, a).
        """
        return x.T

    def __call__(self, x):
        """
        Forward pass for the MNE filter.

        :param x: Input array of shape (n_samples, n_features).
        :return: Filtered output of same shape as input.
        """
        x = np.asarray(x, dtype=np.float64)

        x_t = self.transpose(x)
        x_f = mne.filter.filter_data(data=x_t,
                                     sfreq=self.sfreq,
                                     l_freq=self.l_freq,
                                     h_freq=self.h_freq)
        x_ft = self.transpose(x_f)

        return x_ft