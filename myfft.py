import csv
import os
import sys
import string
import pandas
import scipy
import math
import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import fft, ifft

y = np.empty(shape=[0, 1])
with open(sys.argv[1], encoding='utf-8-sig') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    i = 0
    for row in csv_reader:
        #y = np.append(y, float(row[0]) / 55000)
        #y = np.append(y, float(row[0]) / 65536)
        y = np.append(y, (float(row[0]) - 32768) * 5 / 65536)
        i = i + 1
sr = len(y)
#sr = 4096
ts = 1.0/sr
t = np.arange(0,1,ts)

plt.figure(figsize = (8, 6))
plt.plot(t, y, 'r')
plt.ylabel('Amplitude')
plt.show()



X = fft(y)
N = len(X)
n = np.arange(N)
T = N/sr
freq = n/T

plt.figure(figsize = (12, 6))
plt.subplot(121)

plt.stem(freq, np.abs(X), 'b', \
         markerfmt=" ", basefmt="-b")
plt.xlabel('Freq (Hz)')
plt.ylabel('FFT Amplitude |X(freq)|')
plt.xlim(0, 150)

plt.subplot(122)
plt.plot(t, ifft(X), 'r')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')

plt.tight_layout()
plt.show()
