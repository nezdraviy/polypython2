# Метод опорных векторов (SVM - support vector machine)
# Разделяющая классификация
# Выбирается линия с максимальным отступом

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.svm import SVC

'''iris = sns.load_dataset('iris')
data = iris[['sepal_length', 'petal_length', 'species']]
data_df = data[(data['species'] == 'setosa') | (data['species'] == 'versicolor')]

X = data_df[['sepal_length', 'petal_length']]
y = data_df['species']

data_df_seposa = data_df[['sepal_length', 'petal_length']]
data_df_versicolor = data_df[data_df['species'] == 'versicolor']

ax[i, j].scatter(data_df_seposa['sepal_length'], data_df_seposa['petal_length'])
ax[i, j].scatter(data_df_versicolor['sepal_length'], data_df_versicolor['petal_length'])

model = SVC(kernel='linear', C=10000)
model.fit(X, y)
print(model.support_vectors_)

ax[i, j].scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1], s=300, facecolor='none', edgecolor='black')

x1_p = np.linspace(min(data_df['sepal_length']), max(data_df['sepal_length']), 100)
x2_p = np.linspace(min(data_df['petal_length']), max(data_df['petal_length']), 100)

X1_p, X2_p = np.meshgrid(x1_p, x2_p)

X_p = pd.DataFrame(
    np.vstack([X1_p.ravel(), X2_p.ravel()]).T, columns=['sepal_length', 'petal_length'])

y_p = model.predict(X_p)
X_p['species'] = y_p

X_p_setosa = X_p[X_p['species'] == 'setosa']
X_p_versicolor = X_p[X_p['species'] == 'versicolor']

ax[i, j].scatter(X_p_setosa['sepal_length'], X_p_setosa['petal_length'], alpha=0.2)
ax[i, j].scatter(X_p_versicolor['sepal_length'], X_p_versicolor['petal_length'], alpha=0.2)

#Дз. убрать часть точек, на которых обучаемся и убедиться, что на предсказание вляют только порные вектора'''

iris = sns.load_dataset('iris')
data = iris[['sepal_length', 'petal_length', 'species']]
data_df = data[(data['species'] == 'virginica') | (data['species'] == 'versicolor')]

X = data_df[['sepal_length', 'petal_length']]
y = data_df['species']

data_df_virginica = data_df[['sepal_length', 'petal_length']]
data_df_versicolor = data_df[data_df['species'] == 'versicolor']

c_value = [[10000, 1000, 100, 10], [1, 0.1, 0.01, 0.001]]

fig, ax = plt.subplots(2, 4, sharex='col', sharey='row')

for i in range(2):
    for j in range(4):
        
        ax[i][j].scatter(data_df_virginica['sepal_length'], data_df_virginica['petal_length'])
        ax[i][j].scatter(data_df_versicolor['sepal_length'], data_df_versicolor['petal_length'])

        # Если С большое, то отступ задаётся жестко, чем меньше С, тем более размытый отсутп
        model = SVC(kernel='linear', C=c_value[i][j])
        model.fit(X, y)

        ax[i][j].scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1], s=300, facecolor='none', edgecolor='black')
plt.show()

'''x1_p = np.linspace(min(data_df['sepal_length']), max(data_df['sepal_length']), 100)
x2_p = np.linspace(min(data_df['petal_length']), max(data_df['petal_length']), 100)

X1_p, X2_p = np.meshgrid(x1_p, x2_p)

X_p = pd.DataFrame(
    np.vstack([X1_p.ravel(), X2_p.ravel()]).T, columns=['sepal_length', 'petal_length'])

y_p = model.predict(X_p)
X_p['species'] = y_p

X_p_setosa = X_p[X_p['species'] == 'virginica']
X_p_versicolor = X_p[X_p['species'] == 'versicolor']

ax[i, j].scatter(X_p_setosa['sepal_length'], X_p_setosa['petal_length'], alpha=0.1)
ax[i, j].scatter(X_p_versicolor['sepal_length'], X_p_versicolor['petal_length'], alpha=0.1)
ax[i, j].show()'''

# Модель зависит от небольшого числа векторо -> компактность модели
# На работу метода влияют только точки возле границы

# При большом количестве образцов много вычислительных затрат
# Нет вероятностной интерпретации
# Большая зависимость от размытости


# Деревья решение и случайные леса
# СЛ - непараметрический алгоритм
# Пример ансамблевого метода

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier

iris = sns.load_dataset('iris')
data = iris[['sepal_length', 'petal_length', 'species']]
data_df = data[(data['species'] == 'setosa') | (data['species'] == 'versicolor')]

X = data_df[['sepal_length', 'petal_length']]
y = data_df['species']

data_df_seposa = data_df[['sepal_length', 'petal_length']]
data_df_versicolor = data_df[data_df['species'] == 'versicolor']

plt.scatter(data_df_seposa['sepal_length'], data_df_seposa['petal_length'])
plt.scatter(data_df_versicolor['sepal_length'], data_df_versicolor['petal_length'])

model = DecisionTreeClassifier()
model.fit(X, y)

x1_p = np.linspace(min(data_df['sepal_length']), max(data_df['sepal_length']), 100)
x2_p = np.linspace(min(data_df['petal_length']), max(data_df['petal_length']), 100)

X1_p, X2_p = np.meshgrid(x1_p, x2_p)

X_p = pd.DataFrame(
    np.vstack([X1_p.ravel(), X2_p.ravel()]).T, columns=['sepal_length', 'petal_length'])

y_p = model.predict(X_p)
X_p['species'] = y_p

X_p_setosa = X_p[X_p['species'] == 'setosa']
X_p_versicolor = X_p[X_p['species'] == 'versicolor']

#plt.scatter(X_p_setosa['sepal_length'], X_p_setosa['petal_length'], alpha=0.2)
#plt.scatter(X_p_versicolor['sepal_length'], X_p_versicolor['petal_length'], alpha=0.2)
plt.contour(X1_p, X2_p, y_p.reshape(X1_p.shape), alpha=0.4, level=2, cmap='rainbow', zorder=1)

plt.show()