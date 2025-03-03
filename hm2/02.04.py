# 1. сценарий
# 2. командная оболочка IPython
# 3. Jupyter

# 1.
# plt.show - запускается один раз
# Figure


import matplotlib.pyplot as plt
import numpy as np
import random as rnd


x = np.linspace(0, 10, 100)

# plt.plot(x, np.sin(x))
# plt.plot(x, np.cos(x))

# plt.show()

# IPython
#  %matplotlib
# import matplotlib.pyplot as plt
# plt.plot(...)
# plt.draw()

# Jupyter
# %matplotlib inline - статическая 
# %matplotlib notebook - интерактивная  

# Два способа вывода графиков
# MATLAB
# ОО стиль

x = np.linspace(0, 10, 100)

# plt.figure()

# plt.subplot(2, 1, 1)
# plt.plot(x, np.sin(x))

# plt.subplot(2, 1, 2)
# plt.plot(x, np.cos(x))

# plt.show()

#  Цвета линий 'color'
# - 'blue'
# - 'rgbcmyk' -> 'rgb'
# - '0.14' - градация серого
# - 'RRGGBB' - 'FF00EE'
# - 'RGB' - (1.0, 0.3, 0.2)
# - HTML - 'salmon'


# Стиль линий
# - сплошная '-' 'solid'
# - '--', 'dashed'
# - '-.', 'dashdot'
# - ':', 'dotted'

# fig, ax = plt.subplots(2)

# ax[0].plot(x, np.sin(x))
# ax[1].plot(x, np.cos(x))

# fig = plt.figure()
# ax = plt.axes()
#
# ax.plot(x, np.sin(x), color = 'blue')
# ax.plot(x, np.sin(x - 1))
# ax.plot(x, np.sin(x - 2))
# ax.plot(x, np.sin(x - 3))
# ax.plot(x, np.sin(x - 4))
# ax.plot(x, np.sin(x - 5))

# plt.show()

# plt.subplot()
...

x = np.linspace(0, 10, 30)

rng = np.random.default_rng(0)
colors = rng.random(30)
sizes = 100 * rng.random(30)

# plt.scatter(x, np.sin(x), marker = 'o', c = colors, s = sizes)
# plt.colorbar()
# plt.show()

dy = 0.4
y = np.sin(x) + dy * np.random.randn(30)

# plt.errorbar(x, y, yerr = dy, fmt = '.k')
# plt.fill_between(x, y - dy, y + dy, color = 'red', alpha = 0.4)
# plt.show()

def f(x, y):
    return np.sin(x) **5 + np.cos(20 + x * y) * np.cos(x)

x = np.linspace(0, 5, 50)
y = np.linspace(0, 5, 40)
X, Y = np.meshgrid(x, y)

Z = f(X, Y)

# plt.contour(X, Y, Z, cmap = 'RdGy')
# plt.contourf(X, Y, Z, cmap = 'RdGy')
# plt.imshow(Z , extent=[0,5,0,5], cmap='RdGy', interpolation='gaussian')
# plt.show()

rng = np.random.default_rng(1)
data = rng.normal(size=1000)

# plt.hist(data,
#          bins=30,
#          density=True,
#          histtype='step',
#          edgecolor='red'
#         )
#
x1 = rng.normal(0, 0.8, 1000)
x2 = rng.normal(-2, 1, 1000)
x3 = rng.normal(3, 2, 1000)
#
# plt.hist(x1)
# plt.hist(x2)
# plt.hist(x3)

# print(np.histogram(x1, bins=1))
# print(np.histogram(x2, bins=2))
# print(np.histogram(x3, bins=40))

# ДВумерные гистограммы

mean = [0, 0]
cov = [[1, 1], [1, 2]]

x, y = rng.multivariate_normal(mean, cov, 10000).T

plt.hist2d(x, y, bins=100)
# plt.hexbin(x, y, gridsize=30)

# Легенда

# x = np.linspace(0, 10, 1000)
#
# fig, ax = plt.subplots()
#
# y = np.sin(x[:, np.newaxis] + np.pi * np.arange(0.2, 0.5))
#
# lines = plt.plot(x, y)
# plt.legend(lines, ['1', '2', '3', '4'])

plt.show()