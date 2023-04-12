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

# python3 myfft.py t.csv
# python3 myfft.py t8.csv
y = []
for i in range(8):
    y.append(np.empty(shape=[0, 1]))
with open(sys.argv[1], encoding='utf-8-sig') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    i = 0
    for row in csv_reader:
        for col in range(8):
            y[col] = np.append(y[col],  (float(row[col]) - 32768) / 65536)
        i = i + 1
sr = len(y[0])
ts = 1.0/sr
t = np.arange(0, 1, ts)

X = []
for i in range(8):
    X.append(fft(y[i]))
N = len(X[0])
n = np.arange(N)
T = N/sr
freq = n/T

plt.figure(figsize = (18, 3))
for i in range(8):
    plt.subplot(2, 8, 1 + i)
    plt.plot(t, y[i], 'r')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.subplot(2, 8, 9 + i)
    plt.stem(freq, np.abs(X[i]), 'b', markerfmt=" ", basefmt="-b")
    plt.xlabel('Freq (Hz)')
    plt.ylabel('FFT Amplitude |X(freq)|')
    plt.xlim(0, 256)

plt.tight_layout()
plt.show()
