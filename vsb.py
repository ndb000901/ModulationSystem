import numpy as np
from matplotlib.pylab import *
from pylab import *
import matplotlib.pyplot as plt

mpl.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# This function is FFT to calculate a signal’s Fourier transform
# Input: t: sampling time , st : signal data. Time length must greater thean 2
# output: f : sampling frequency , sf: frequen
# output is the frequency and the signal spectrum


def FFT_SHIFT(t, st):
    dt = t[2] - t[1]
    T = t[-1]
    df = 1 / T
    N = len(t)
    f = np.arange(-N / 2, N / 2, 1) * df
    sf = fft(st)
    sf = T / N * np.fft.fftshift(sf)
    return f, sf


def IFFT_SHIFT(f, Sf):
    df = f[2] - f[1]
    fmax = f[-1] - f[1] + df
    dt = 1 / fmax
    N = len(f)
    t = np.arange(1, N + 1, 1) * dt
    Sf = np.fft.fftshift(Sf)
    st = fmax * np.fft.ifft(Sf)
    st = real(st)
    return t, st

# This function is a residual bandpass filter
# Inputs  f: sample frequency, sf: frequency spectrum data
#        B1: residual bandwidth, B2: highest freq of the baseband signal
# Outputs t:sample time,  st: signal data


def vsbmd(f, sf, B1, B2, fc):
    df = f[2] - f[1]
    T = 1 / df
    hf = np.zeros(len(f))
    bf1 = np.arange(np.floor((fc - B1) / df), np.floor((fc + B1) / df) + 1, 1)
    bf2 = np.arange(np.floor((fc - B1) / df) + 1, np.floor((fc + B2) / df) + 1, 1)
    f1 = bf1 + floor(len(f) / 2)
    f2 = bf2 + floor(len(f) / 2)
    stepf = 1 / len(f1)

    j = 0
    for i in f1:
        hf[int(i - 1)] = np.arange(0, 1, stepf)[j]
        j += 1
    for i in f2:
        hf[int(i - 1)] = 1
    f3 = -bf1 + floor(len(f) / 2)
    f4 = -bf2 + floor(len(f) / 2)

    j = 0
    for i in f3:
        hf[int(i - 1)] = np.arange(0, 1, stepf)[j]
        j += 1
    for i in f4:
        hf[int(i - 1)] = 1

    yf = hf * sf
    t, st = IFFT_SHIFT(f, yf)
    st = real(st)
    return t, st

# 显示模拟调制的波形及其解调方法VSB，文件名：VSB.m
# Signal
dt = 0.001
fmax = 5
fc = 20
T = 5
N = T / dt
t = np.arange(0, N, 1) * dt
mt = sqrt(2) * (cos(2 * pi * fmax * t) + sin(2 * pi * 0.5 * fmax * t))

# VSB modulation
s_vsb = mt * cos(2 * pi * fc * t)
B1 = 0.2 * fmax
B2 = 1.2 * fmax
f, sf = FFT_SHIFT(t, s_vsb)
t, s_vsb = vsbmd(f, sf, B1, B2, fc)

# Power Spectrum Density
f, sf = FFT_SHIFT(t, s_vsb)
PSD = (abs(sf) ** 2) / T

# Plot VSB and PSD
plt.figure(figsize=(9, 3.5))
plt.plot(t, s_vsb, color='blue', linewidth=1, linestyle=":")
plt.plot(t, mt, color='red', linewidth=1, linestyle=":")
plt.title('VSB调制信号')
plt.xlabel('t')
plt.show()


plt.plot(f, PSD)
plt.title('VSB信号功率谱')
plt.xlim(-40, 40)
plt.xlabel('f')
plt.show()




