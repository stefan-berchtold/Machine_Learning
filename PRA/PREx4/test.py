import numpy as np
import matplotlib.pyplot as plt

from scipy import signal
from scipy.io import wavfile

from hidden_markov import hmm

# basic morse dictionary, we use '-' to mark word spaces
alphabet = list('abcdefghijklmnopqrstuvwxyz-')
values = ['.-', '-...', '-.-.', '-..', '.', '..-.', '--.', '....', '..', '.---', '-.-',
          '.-..', '--', '-.', '---', '.--.', '--.-',
          '.-.', '...', '-', '..-', '...-', '.--', '-..-', '-.--', '--..', '-....-']

morse_dict = dict(zip(alphabet, values))
ascii_dict = dict(map(reversed, morse_dict.items()))  # inverse mapping


# convert text to morse code
def morse_encode(text):
    return ' '.join([''.join(morse_dict.get(i, '')) for i in text.lower()])


# convert morse code to text
def morse_decode(code):
    return ''.join([ascii_dict.get(i, '') for i in code.split(' ')])


# read audio stream
fs, data = wavfile.read('C:/home/stefan/PRA/morse.wav')

# spectral representation
f, t, S = signal.spectrogram(data, fs)
f1 = f[0:23]
print(f)
plt.figure(0)  # spectrogram
plt.pcolormesh(t, f1, S)
plt.ylabel('frequency [Hz]')
plt.xlabel('time [sec]')
plt.title('spectrogram')
plt.show()

fmin = 500
fmax = 700
freq_slice = np.where((f >= fmin) and (f<= fmax))
f = f[freq_slice]
S = S[freq_slice, :][0]
print(f)

plt.figure(2)  # spectrogram
plt.pcolormesh( f1, S1)
plt.ylabel('frequency [Hz]')
plt.xlabel('time [sec]')
plt.title('spectrogram')
plt.show()



def select_active_frequencies(S, threshold=0.975):
    '''Return the indexes of S that meet the threshold.
    Threshold will be used to select the quantiles
    '''
    indices = []
    sums = []
    for i, intensity in enumerate(S):
        sums.append(sum(intensity))
    limiter = np.quantile(sums, threshold)
    for i, intensity_sum in enumerate(sums):
        if intensity_sum >= limiter:
            indices.append(i)
    return indices




def get_binary(S, threshold=0.5):
    '''Return binary representation
    '''
    steps = np.mean(S, axis=0)
    #steps = S[np.argmax(np.sum(S, axis=1)),:]
    return list(map(lambda x: 1 if x >= np.quantile(steps, threshold) else 0, steps))

