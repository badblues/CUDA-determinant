import matplotlib.pyplot as plt
import numpy as np

cpu_time_f = open('cmake-build-debug/cpu_time.txt', 'r')
gpu_time_f = open('cmake-build-debug/gpu_time.txt', 'r')

cpu_time = []
gpu_time = []
size = []

for line in cpu_time_f:
    size.append(int(line.partition(' - ')[0]))
    cpu_time.append(float(line.partition(' - ')[2]))

for line in gpu_time_f:
    gpu_time.append(float(line.partition(' - ')[2]))


def lin_smooth(X, Y):
    n = len(Y) - 1
    y = np.full(len(Y), 0.)
    y[0] = (5 * Y[0] + 3 * Y[1] + Y[2] + Y[3] + Y[4]) / 11
    y[1] = (3 * Y[0] + 5 * Y[1] + 3 * Y[2] + 2 * Y[3] + Y[4]) / 14
    y[n - 1] = (3 * Y[n] + 5 * Y[n - 1] + 3 * Y[n - 2] + 2 * Y[n - 3] + Y[n - 2]) / 14
    y[n] = (5 * Y[n] + 3 * Y[n - 1] + Y[n - 2] + Y[n - 3] + Y[n - 4]) / 11
    for i in range(2, n - 1):
        y[i] = (Y[i - 2] + Y[i - 1] + Y[i] + Y[i + 1] + Y[i + 2]) / 5
    return y

plt.plot(size, cpu_time, label='cpu')
plt.plot(size, gpu_time, label='gpu')
plt.legend()
plt.grid()
plt.xlabel('size')
plt.ylabel('time, s')
plt.show()
