import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, KFold, cross_val_score

data = np.array(
    [
        [1, 5],
        [2, 7],
        [3, 7],
        [4, 10],
        [5, 11],
        [6, 14],
        [7, 17],
        [8, 19],
        [9, 22],
        [10, 28]
    ]
)

# Градиентный спуск - пакетный градиентный спуск.
# На практике используют стохастический градиентный спуск, на каждой итерации обучаемся на одной выборке из данных.

x = data[:, 0]
y = data[:, 1]

n = len(x)

w1 = 0.0
w0 = 0.0

iterations = 100000
L = 0.001
# Размер выборки

sample_size = 1

for i in range(iterations):
    id = np.random.choice(n, sample_size, replace=False)
    D_w0 = 2 * sum(-y[id] + w0 + w1 * x[id])
    D_w1 = 2 * sum(x[id] * (-y[id] + w0 + w1 * x[id]))
    w1 -= L * D_w1
    w0 -= L * D_w0

# Как оценить эффективность модели?
data_df = pd.DataFrame(data)
#print(data_df.corr(method='pearson'))

# Коэффициент корреляции помогает понять, есть ли связь между двемя переменными

# Обучающие и тестовые выборки
# Основной метод борьбы с переобучением, заключается в том, что набор данных делится на обучающую и тестовую выборки

# Во всех видах машинного обучения с учителем это встречается

# 2/3 - на обучение, 1/3 на тест (4/5 к 1/5, 9/10 к 1/10)

X = data_df.values[:, :-1]
Y = data_df.values[:, -1]

#X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 1/3)
#print(X_train)

kfold = KFold(n_splits=3, random_state=1, shuffle=True)
model = LinearRegression()
#model.fit(X_train, Y_train)
results = cross_val_score(model, X, Y, cv=kfold)

#print(results)#Средние квадратические ошибки
#print(results.mean(), results.std()) #Метрики показывают насколько ЕДИНООБРАЗНО ведёт себя модель на разных выборках

# Вылидационная выборка - для сравнения различных моеделй и конфигураций

# Многомерная линейная регрессия

data_df = pd.read_csv('C:/Users/temak/Downloads/Telegram Desktop/multiple_independent_variable_linear.csv')
print(data_df.head())

X = data_df.values[:,:-1]
Y = data_df.values[:,-1]

model = LinearRegression().fit(X, Y)
print(model.coef_, model.intercept_)

x1 = X[:, 0]
x2 = X[:, 1]
y = Y

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(x1, x2, y)

x1_ = np.linspace(min(x1), max(x1), 100)
x2_ = np.linspace(min(x2), max(x2), 100)

X1_, X2_ = np.meshgrid(x1_, x2_)

Y_ = model.intercept_ + model.coef_[0] * X1_ + model.coef_[1] * X2_
ax.plot_surface(X1_, X2_, Y_, cmap="Greys", alpha=0.5)

plt.show()


