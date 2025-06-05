# Наивная байесовская классификация
# Набор моделей, которые предлагают решение задачи классификации 
# В основе лежит теорема Байеса: P(a|b) = P(b|a) * p(a) / p(b)
# В машинном обучении - вероятность присваивания определённой метки при условии, что пришёл какой-то набор признаков
# В бинарной классификации считаем отношение вероятностей

# Гауссовский наивный байесовский классификатор
# Допущение - данные всех категорий взяты из простого нормального распределения

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.naive_bayes import GaussianNB

iris = sns.load_dataset('iris')
#sns.pairplot(iris, hue='species')

data = iris[['sepal_length', 'petal_length', 'species']]

# setosa versicolor

data_df = data[(data['species'] == 'setosa') | (data['species'] == 'versicolor')]

X = data_df[['sepal_length', 'petal_length']]
Y = data_df['species']

model = GaussianNB()
model.fit(X, Y)
var0 = model.var_[0]
theta0 = model.theta_[0]
var1 = model.var_[1]
theta1 = model.theta_[1]

data_df_seposa = data_df[['sepal_length', 'petal_length']]
data_df_versicolor = data_df[data_df['species'] == 'versicolor']

plt.scatter(data_df_seposa['sepal_length'], data_df_seposa['petal_length'])
plt.scatter(data_df_versicolor['sepal_length'], data_df_versicolor['petal_length'])

x1_p = np.linspace(min(data_df['sepal_length']), max(data_df['sepal_length']), 100)
x2_p = np.linspace(min(data_df['petal_length']), max(data_df['petal_length']), 100)

X1_p, X2_p = np.meshgrid(x1_p, x2_p)

X_p = pd.DataFrame(
    np.vstack([X1_p.ravel(), X2_p.ravel()]).T, columns=['sepal_length', 'petal_length']
)

y_p = model.predict(X_p)

z1 = (
    1 
    / (1 * np.pi * (var0[0] * var0[1]) ** 0.5) 
    * np.exp(
        -0.5 
        * ((X1_p - theta0[0]) ** 2 / (var0[0])) + (X2_p - theta0[1]) ** 2 / (var0[1]))
    )
plt.contour(X1_p, X2_p, z1)

X_p['species'] = y_p

X_p_setosa = X_p[X_p['species'] == 'setosa']
X_p_versicolor = X_p[X_p['species'] == 'versicolor']

plt.scatter(X_p_setosa['sepal_length'], X_p_setosa['petal_length'], alpha=0.2)
plt.scatter(X_p_versicolor['sepal_length'], X_p_versicolor['petal_length'], alpha=0.2)

plt.show()

# setosa virginica