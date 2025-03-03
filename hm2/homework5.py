import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import random


x1 = np.linspace(0, 20, 5)
y1 = [random.randrange(i, 5+i) for i in range(5)]

x2 = np.linspace(0, 20, 5)
y2 = [random.randrange(i, 5+i) for i in range(5)]

plt.plot(x1, y1, marker = 'o', color = 'red', label = 'line 1')
plt.plot(x2, y2, 'g-.o', label = 'line 2')
plt.legend()
plt.show()

plt.figure(figsize=(12, 7))

plt.subplot(2, 2, (1, 2))
plt.plot(x1, y1)

plt.subplot(2, 2, 3)
plt.plot(x1, (x1 - 10) ** 2)

plt.subplot(2, 2, 4)
x3 = np.linspace(0, 20, 10)
y3 = [x ** 2 if x <= 10 else (50 / (x - 10)) for x in x3]
plt.plot(x3, y3)
plt.show()

x4 = np.linspace(-5, 5, 10)
plt.plot(x4, x4 ** 2)
plt.annotate('min', (0, 0), xytext=(0, 7), arrowprops=dict(facecolor = 'green'))
plt.show()

rng = np.random.default_rng()
data = rng.normal(5, 3, (7, 7))
plt.figure()
plt.pcolormesh(data)
plt.colorbar(shrink = 0.5, aspect = 7)
plt.clim(0, 10)
plt.show()

plt.figure()
x5 = np.linspace(0, 5, 100)
plt.plot(x5, np.cos(x5 * np.pi), linewidth=2, color = 'red')
plt.fill_between(x5, np.cos(x5 * np.pi))
plt.show()

plt.figure()
x5 = np.linspace(0, 5, 1000)
y_masked = np.ma.masked_where(np.cos(x5 * np.pi) < -0.5, np.cos(x5 * np.pi))
plt.ylim(-1, 1)
plt.plot(x5, y_masked, linewidth=2)
plt.show()


x = np.arange(0, 6, 1)
y = np.arange(0, 6, 1)

plt.figure(figsize=(12, 5))
plt.subplot(1, 3, 1)
plt.step(x, y, 'og-', where='pre')
plt.grid(True)

plt.subplot(1, 3, 2)
plt.step(x, y,'og-', where='post')
plt.grid(True)

plt.subplot(1, 3, 3)
plt.step(x, y,'og-', where='mid')
plt.grid(True)
plt.show()


plt.figure()
x6 = np.linspace(0, 10, 20)
y1 = - 0.2 * (x6 - 5) ** 2 + 4
y2 = - 0.6 * (x6 - 5) ** 2 + 12
y3 = - 0.5 * (x6 - 7) ** 2 + 21
plt.fill_between(x6, y1, label = 'y1')
plt.fill_between(x6, y1, y2, label = 'y2')
plt.fill_between(x6, y2, y3, label = 'y3')
plt.xlim(0, 10)
plt.ylim(0, 25)
plt.legend(loc = 'upper left')
plt.show()

plt.figure()
vals = [25, 10, 50, 15, 35]
labels = ["Ford", "Toyota", "BMW", "Audi", "Jaguar"]
plt.pie(vals, labels=labels, explode=(0, 0, 0.1, 0, 0))
plt.show()

plt.figure()
plt.pie(vals, labels=labels, wedgeprops=dict(width = 0.5))
plt.show()