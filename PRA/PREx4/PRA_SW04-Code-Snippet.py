
import numpy as np
import matplotlib.pyplot as plt

from scipy import signal
from scipy.io import wavfile

from hidden_markov import hmm

# basic morse dictionary, we use '-' to mark word spaces
alphabet = list('abcdefghijklmnopqrstuvwxyz-')
values = ['.-', '-...', '-.-.', '-..', '.', '..-.', '--.', '....', '..', '.---', '-.-', 
          '.-..', '--', '-.','---', '.--.', '--.-', 
          '.-.', '...', '-', '..-', '...-', '.--', '-..-', '-.--', '--..','-....-']

morse_dict = dict(zip(alphabet, values))
ascii_dict = dict(map(reversed, morse_dict.items())) # inverse mapping

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

plt.figure(0) # spectrogram
plt.pcolormesh(t, f, S)
plt.ylabel('frequency [Hz]')
plt.xlabel('time [sec]')
plt.title('spectrogram')
plt.show()

f_index = __TODO__   # select frequency components closest to 600 Hz

binary_data = S[f_index,:] __TODO__  # convert to binary data

plt.figure(1)    # binary sequence
plt.plot(t,binary_data,'ro', markersize=1)
plt.ylabel('signal')
plt.xlabel('time [sec]')
plt.title('binary signal')
plt.show()

#==============================================================================
# states and symbols used 

# dot, dash, symbol space, character space, indifferent
states = ('.1','.2','-1','-2','-3','-4','-5','-6',' 1',' 2','_1','_2','_3','_4','_5','_6','x')
symbols = (True,False)

# morse symbols correspondig to states
state_values = ['.','','-','','','','','','','',' ','','','','','','']
state_dict = dict(zip(states, state_values))

# convert state sequence to morse symbols, remove trailing or leading spaces
def state_decode(sequence):
    return (''.join([''.join(state_dict[i]) for i in sequence])).strip()
 
#==============================================================================
# set up and run hidden markov model
    
# we assume that we do not start in the middle of a symbol
initial = np.matrix([
        [1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0]
    ])

__TODO__ define transition matrix
# state  .1 .2 -1 -2 -3 -4 -5 -6  1  2 _1 _2 _3 _4 _5 _6 x    
transition = np.matrix([
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # .1 
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # .2 nominal length
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # -1
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # -2
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # -3
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # -4
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # -5  
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], # -6 nominal length
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], #  1
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], #  2 nominal length
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], # _1
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], # _2
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], # _3
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0], # _4
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0], # _5
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], # _6 nominal length
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]  # x  indifferent  
    ])

# emission probabilities
emission = np.matrix([
        [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1], # emitting 'True'
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]  # emitting 'False'
    ]).transpose() 
  
    
# scale as probabilities, e.g. that components sum up to 1
initial = initial / np.sum(initial, axis=1).reshape((-1, 1))
transition = transition / np.sum(transition, axis=1).reshape((-1, 1))
emission = emission / np.sum(emission, axis=1).reshape((-1, 1))

# generate model
model = hmm(states, symbols, initial, transition, emission)

# run viterbi decoder
state_sequence = model.viterbi(binary_data)

result = morse_decode(state_decode(state_sequence))

print('Decoded string =>',result,'<=')

#==============================================================================

