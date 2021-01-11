from pylab import *
import numpy as np
from matplotlib.pylab import *
import matplotlib.pyplot as plt

mpl.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 调制信号频率
w11 = 100 * pi
w12 = 50 * pi

# 调制信号振幅
u11m = 0.1
u12m = 0.4

# 载波信号频率
w2 = 1000 * pi

# 载波信号振幅
u2m = 2

k = 20
t = np.arange(0, 0.1, 0.0000001)

# 调制信号
u1 = u11m * cos(w11 * t) + u12m * cos(w12 * t)

# 载波信号
u2 = u2m * cos(w2 * t)

# SSBU已调信号
u3 = 0.5 * k * u2m * (u11m * cos((w2 + w11) * t) + u12m * cos((w2 + w12) * t))

# SSBL已调信号
u4 = 0.5 * k * u2m * (u11m * cos((w2 - w11) * t) + u12m * cos((w2 - w12) * t))

plt.plot(t, u1, 'r')
plt.title('调制信号')
plt.show()

Y1 = fft(u1)  # 对调制信号进行傅里叶变换

plot(abs(Y1), 'r')
plt.xlim(0, 100)
plt.title('调制信号频谱')
plt.show()


plt.plot(t, u2)
plt.title("载波信号")
plt.xlabel('t')
plt.ylabel('w')
show()

Y2 = fft(u2)  # 对载波信号进行傅里叶变换
plt.plot(abs(Y2))
plt.title('载波信号频谱')
plt.ylabel('w')
plt.xlim(0, 100)
plt.show()

plt.plot(t, u3)
plt.title('SSBU调幅信号')
plt.xlabel('t')
plt.show()

Y3 = fft(u3)  # 对SSBU已调信号进行傅里叶变换
plt.plot(abs(Y3))
plt.title('SSBU频谱')
plt.xlabel('w')
plt.xlim(10, 90)
plt.show()

plt.plot(t, u4)
plt.title('SSBL调幅信号')
plt.xlabel('t')
plt.show()

Y4 = fft(u4)  # 对SSBL已调信号进行傅里叶变换
plt.plot(abs(Y4))
plt.title('SSBL频谱')
plt.xlabel('w')
plt.xlim(10, 90)
plt.show()




