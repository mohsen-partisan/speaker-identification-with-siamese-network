
import numpy as np
from scipy.io import wavfile

path = '/home/mohsen/Desktop/family/FAEZEH-20200715-WA0000.wav'
frequency, signal = wavfile.read(path)

slice_length = 4 # in seconds
overlap = 1 # in seconds
slices = np.arange(0, len(signal), slice_length-overlap, dtype=np.int)

for start, end in zip(slices[:-1], slices[1:]):
    start_audio = start * frequency
    end_audio = end * frequency
    audio_slice = signal[start_audio: end_audio]
    a=0