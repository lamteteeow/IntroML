from math import ceil, floor, log2
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fftfreq as scfftfreq
import os
from numpy.fft import fftfreq
from tqdm import tqdm
from scipy.fftpack import fft, fftfreq

# divide with integral result (discard remainder)
def load_sample(filename, duration=4*44100, offset=44100//10):
    # Read Signal from input file
    # Extract range (duration) from signal starting at highest peak + offset
    signal = np.load(filename)
    maxIdx = np.argmax(signal)
    start = maxIdx + offset
    end = start + duration
    return signal[start:end]

# helper function to compute (freq, amp)-array
def compute_pos_freq_amp_pairs(signal, min_freq):
    # Scale X-Axis according to sampling rate
    sampling_rate = 44100
    T = 1 / sampling_rate
    freq = scfftfreq(signal.size, T)
    # calculate absolute values - Y axis
    fft = np.fft.fft(signal).real
    abs_amplitudes = np.abs(fft)
    # Combine arrays to one, transpose it to have matching pairs (freq, amp)
    comb_freq_amp = np.array([freq, abs_amplitudes]).T
    # only select positive values
    cond_min_freqs = freq > min_freq
    pos_comb_freq_amp = comb_freq_amp[cond_min_freqs]
    return pos_comb_freq_amp

def compute_frequency(signal, min_freq=20):
    # Compute Fast Fourier Transform (real part)
    pos_comb_freq_amp = compute_pos_freq_amp_pairs(signal, min_freq)
    # find index of max in both columns
    idx_max_freq = np.argmax(pos_comb_freq_amp, axis=0)[1]
    # idx of max freq corresponds to second entry (frequencies are on the left) 
    max_freq=pos_comb_freq_amp[idx_max_freq][0]
    print("Max Freq: {}".format(max_freq))
    return max_freq

###########################################
# Musterlösung
###########################################
#def compute_frequency(signal, min_freq=20):
#    duration = signal.shape[0]
#    f = np.abs(np.fft.fft(signal))
#    f[:ceil(min_freq*duration/44100)] = 0
#    pos = np.argmax(f[:f.shape[0]//2])
#    return pos / duration * 44100

def main():
    # list with filenames to iterate
    files = ['Piano.ff.A2.npy',
             'Piano.ff.A3.npy',
             'Piano.ff.A4.npy',
             'Piano.ff.A5.npy',
             'Piano.ff.A6.npy',
             'Piano.ff.A7.npy',
             'Piano.ff.XX.npy']
    # buffer arrays to store signal objects
    signals = np.empty(7, dtype=object)
    fourier_space = np.empty(7, dtype=object)
    frequencies = []
    # counter for iteration (easier access of idxs)
    i = 0
    # calculate signal for each file
    for file in files:
        signals[i] = load_sample("./sounds/{}".format(file))
        # select only positive frequencies for fourier_space
        fourier_space[i] = compute_pos_freq_amp_pairs(signals[i], 20).T[1]
        frequencies.append(compute_frequency(signals[i]))
        i += 1

    # print all frequencies
    print(frequencies)

    # Plot signals
    for i in range(7):
        plt.subplot(7, 2, 2*i + 1)
        plt.title(files[i])
        plt.plot(signals[i])

        plt.subplot(7, 2, 2*i + 2) 
        plt.title('Fourier Transform of {}'.format(files[i]))
        plt.plot(fourier_space[i])
    plt.show()

    ########################################################
    # Determine mystery note
    ########################################################

    
    # gives me key number of specific frequency (mystery note is last entry in frequencies)
    key = floor(12*log2(frequencies[6]/440) + 49)
    print("Mysterious note has this key number: ", key)
    print("This corresponds to the Note D6 (freq: 1174.659, source wikipedia)")
    # gives me frequency of the key, not necessary to calculate :)
    freq = 2**((key-49)/12) * 440
    print("Mysterious note has this frequency (Hz): ", round(freq, 3))

    names = ['A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'XX']
    expected_values = [110, 220, 440, 880, 1760, 3520, 'unknown'] 
    comp_arr = list(zip(names, frequencies, expected_values))
    for name, freq, exp_val in comp_arr:
        print(name, " has following frequency (Hz): ", freq, ", expexted value: ", exp_val)
    
if __name__ == '__main__':
    # Implement the code to answer the questions here
    main()

# This will be helpful:
# https://en.wikipedia.org/wiki/Piano_key_frequencies
