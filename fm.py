import numpy as np
from matplotlib.pylab import *
from pylab import *
import matplotlib.pyplot as plt

try:
    mpl.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
except:
    print('haha')

ts = 0.00125  # 信号抽样时间间隔
t = np.arange(0, 10, ts)  # 时间向量
am = 10
fs = 1 / ts  # 抽样频率
df = fs / len(t)  # fft的频率分辨率
msg = am * cos(2 * pi * 10 * np.arange(0, 1, 0.01))
msg1 = np.outer(msg.conj().T, np.ones([1, int(fs / 10)]))  # 扩展成取样信号形式
msg2 = msg1.T.reshape(1, len(t), order='F')
Pm = fft(msg2)  # 求消息信号的频谱
f = np.arange(-fs / 2, fs / 2, df)


plt.plot(t[np.newaxis, ][0], fft(abs(Pm))[0], 'g')
plt.title('信号消息频谱')
plt.show()

m = fft(msg, 1024)  # 对msg进行傅利叶变换
N = arange(0, len(m), 1) * fs / len(m) - fs / 2

plt.plot(N[np.newaxis, ][0], abs(m))  # 调制信号频谱图
plt.title('调制信号频谱')
plt.show()

int_msg = np.arange(8000.)[np.newaxis, ]
int_msg[0][0] = 0   # 消息信号积分
for i in range(0, len(t) - 1):
    int_msg[0][i + 1] = int_msg[0][i] + msg2[0][i] * ts


kf = 50
fc = 250  # 载波频率
Sfm = am * cos(2 * pi * fc * t + 2 * pi * kf * int_msg)  # 调频信号
Pfm = fft(Sfm) / fs  # FM信号频谱

# 画出已调信号频谱
plt.plot(f[np.newaxis, ][0], fftshift(abs(Pfm))[0])
plt.title('FM信号频谱')
plt.show()


Pc = sum(abs(Sfm) ** 2) / len(Sfm)  # 已调信号功率
Ps = sum(abs(msg2) ** 2) / len(msg2)  # 消息信号功率
fm = 50
betaf = kf * max(msg) / fm  # 调制指数
W = 2 * (betaf + 1) * fm  # 调制信号带宽










