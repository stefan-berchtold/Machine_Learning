import numpy as np
import matplotlib.pyplot as plt

from scipy import signal
from scipy.io import wavfile

from hidden_markov import hmm

fs, data = wavfile.read('C:/home/stefan/PRA/morse.wav')

# spectral representation
f, t, S = signal.spectrogram(data, fs)

print(f[20])
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

f_index = select_active_frequencies(S)

threshold = 0.975
indices = []
sums = []
for i, intensity in enumerate(S):
    sums.append(sum(intensity))
limiter = np.quantile(sums, threshold)
for i, intensity_sum in enumerate(sums):
    if intensity_sum >= limiter:
        indices.append(i)

a = S[0]
b = np.sum(a)
print(b)
print(sums)
print(indices)
limiter = np.quantile (sums, threshold)
print(limiter)
stats.describe(sums)
def get_binary(S, threshold=0.5):
    '''Return binary representation
    '''
    steps = np.mean(S, axis=0)
    #steps = S[np.argmax(np.sum(S, axis=1)),:]
    return list(map(lambda x: 1 if x >= np.quantile(steps, threshold) else 0, steps))


steps = np.mean(S, axis = 0)
plt.plot(S         )
plt.show()
#binary_data = get_binary(S[f_index,:])
#print(binary_data)
threshold = 0.8
a = list(map(lambda x: 1 if x >= np.quantile(steps, threshold) else 0, steps))
print(a)