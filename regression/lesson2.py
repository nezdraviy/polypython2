# Линейная регрессия

# На основе наблюдений строим гиперплоскость, стараемся подогнать функцию к нашим данным, чтобы спронозировать новые даные
# Пытаемся установить линейную связь между переменными

import numpy as np
import matplotlib.pyplot as plt
import random
'''from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression

features, target = make_regression(n_samples=20, n_features=1, n_informative=1, n_targets=1, noise=1, random_state=1)

model = LinearRegression().fit(features, target)
plt.scatter(features, target)

x = np.linspace(features.min(), features.max(), 100)
plt.plot(x, model.coef_[0] * x + model.intercept_, color='red')
plt.show()

## Простая линейная регрессия

# Линейная -> линейная зависимость

# + пронозирование
# + анализ влияния переменных друг на друга

# - точки обучаемых данных НЕ будет точно лежать на прямой, возникает область погрешности
# - НЕ позволяет делать пронозы ВНЕ диапозона имеющихся данных

data = np.array(
    [
        [1, 5],
        [2, 7],
        [3, 9],
        [4, 11],
        [5, 13],
        [6, 15],
        [7, 17],
        [8, 19],
        [9, 21],
        [10, 25]
    ]
)

x = data[:, 0]
y = data[:, 1]
n = len(x)

# Аналитическое решение задачи линейной регрессии

# Формулки
w_1 = (n * sum(x[i] * y[i] for i in range(n)) - sum(x[i] for i in range(n)) * sum(y[i] for i in range(n))) / (n * sum(x[i] ** 2 for i in range(n)) - sum(x[i] for i in range(n)) ** 2)

w_0 = (sum(y[i] for i in range(n))) / n - w_1 * sum(x[i] for i in range(n)) / n

print(w_1, w_0)

#Метод обратных матриц
x_1 = np.vstack([x, np.ones(len(x))]).T
w = np.linalg.inv(x_1.transpose() @ x_1) @ (x_1.transpose() @ y)
print(w)

#Разложение матриц (QR - разложение)
Q, R = np.linalg.qr(x_1)
w = np.linalg.inv(R) @ Q.T @ y.T
print(w)'''

#Градиентный спуск
def f(x):
    return (x - 3) ** 2 + 4

def dx_f(x):
    return 2 * x - 6

#x = np.linspace(-2, 2, 100)
#ax = plt.gca()

#ax.xaxis.set_major_locator(plt.MultipleLocator(0.5))
#plt.plot(x, f(x))
#plt.plot(x, dx_f(x))
#plt.grid()
#plt.show()
data = np.array(
    [
        [1, 5],
        [2, 7],
        [3, 9],
        [4, 11],
        [5, 13],
        [6, 15],
        [7, 17],
        [8, 19],
        [9, 21],
        [10, 25]
    ]
)

x = data[:, 0]
y = data[:, 1]
n = len(x)
L = 0.001
iterations = 100000
w0 = 0.0
w1 = 0.0
#for i in range(iterations):
 #   D_w0 = 2 * sum(y[i] - w0 - w1 * x[i] for i in range(n))
  #  D_w1 = 2 * sum((x[i] * (-y[i] - w0 - w1 * x)) for i in range(n))
   # w1 -= L * D_w1
    #w0 -= L * D_w0

#print(w1, w0)

