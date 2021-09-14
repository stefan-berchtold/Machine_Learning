import numpy as np
import matplotlib.pyplot as plt

from scipy import signal
from scipy.io import wavfile

from hidden_markov import hmm


fs, data = wavfile.read('C:/home/stefan/PRA/morse.wav')

# spectral representation
f, t, S = signal.spectrogram(data, fs)

fmin = 550 # Hz
fmax = 700 # Hz
freq_slice = np.logical_and(f >= fmin, f <= fmax)
print(freq_slice)
f1 = f[freq_slice]
S1 = S[freq_slice]
#t1 = t[freq_slice]

plt.figure(0) # spectrogram
plt.pcolormesh(t, f1, S1)
plt.ylabel('frequency [Hz]')
plt.xlabel('time [sec]')
plt.title('spectrogram')
plt.show()