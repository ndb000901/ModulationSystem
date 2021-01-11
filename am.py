import numpy as np
from matplotlib.pylab import *
from pylab import *
import matplotlib.pyplot as plt

try:
    mpl.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
except:
    print('haha')


def T2F(t, st):
    T = t[-1]
    df = 1 / T
    N = len(st)
    f = np.arange(-N / 2 * df, N / 2 * df, df)
    sf = fft(st)
    sf = T / N * np.fft.fftshift(sf)
    return f, sf

def F2T(f, Sf):
    df = f[2] - f[1]
    fmax = (f[-1] - f[1] + df)
    dt = 1 / fmax
    N = len(f)
    t = np.arange(0, N, 1) * dt
    Sf = np.fft.fftshift(Sf[0])
    st = fmax * np.fft.ifft(Sf)
    st = np.real(st)
    return t, st


def lpf(f, sf, B):
    df = f[2] - f[1]
    fN = len(f)
    ym = zeros(fN)[np.newaxis, ]
    xm = floor(B / df)
    xm_shift = np.arange(-xm, xm, 1) + floor(fN / 2)
    for i in xm_shift:
        ym[0][int(i)] = 1

    yf = ym * sf[np.newaxis, ][0]
    t, st = F2T(f, yf)
    return t, st


dt = 0.001  # 时间采样频谱
Fs = 100
fm = 1  # 信源的最高频率
fc = 10  # 载波中心频率
T = 4  # 信号时长
N = T / dt  # 采样点个数
t = np.arange(0, N, 1) * dt  # 采样点的时间序列
wc = 2 * pi * fc
mt = cos(2 * pi * t)  # 信源


plt.plot(t, mt)
plt.title('基带调制信号')
plt.show()
# mt的最大值是1
A = 2
Fc = A * cos(wc * t)

sam = (A + mt) * cos(wc * t)

plt.plot(t, Fc)
plt.title('载波信号')
plt.show()

plt.plot(t, sam)  # 画出AM信号波形
plt.plot(t, A + mt)
plt.title('AM调制信号及其包络 A=2')
plt.show()

# 相干解调
st = sam * cos(wc * t)

plt.plot(t, st)
plt.title('调制信号与载波信号相乘')
plt.show()

f, sf = T2F(t, st)  # 傅里叶变换
t1, st = lpf(f, sf, 2 * fm)  # 低通滤波
print('')
plt.plot(t1, st)
plt.title('经过低通滤波的相干解调信号波形')
plt.show()