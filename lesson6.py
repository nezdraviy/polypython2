# Ансамблиевые методы. В основе идея объединение нескольких переобученный моделей для уменьшения эффекта переобучения
# Это называется баггинг
# Баггинг усредняет результаты -> оптимальная классификации

# Ансамбль случайных деревьев называется случайным лесом

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier

iris = sns.load_dataset('iris')
species_int = []
for r in iris.values:
    match r[4]:
        case 'setosa':
            species_int.append(1)
        case 'versicolor':
            species_int.append(2)
        case 'virginica':
            species_int.append(3)

species_int_df = pd.DataFrame(species_int)
print(species_int_df.head())

data = iris[['sepal_length', 'petal_length']]
data['species'] = species_int

print(data.head())
print(data.shape)

data_setosa = data[data['species'] == 1]
data_versicolor = data[data['species'] == 2]
data_virginica = data[data['species'] == 3]

x1_p = np.linspace(min(data['sepal_length']), max(data['sepal_length']))
x2_p = np.linspace(min(data['petal_length']), max(data['petal_length']))

X1_p, X2_p = np.meshgrid(x1_p, x2_p)

X_p = pd.DataFrame(
    np.vstack([X1_p.ravel(), X2_p.ravel()]).T, columns=['sepal_length', 'petal_length']
)

fix, ax = plt.subplots(1, 1, sharex='col', sharey='row')
ax.scatter(data_setosa['sepal_length'], data_setosa['petal_length'])
ax.scatter(data_versicolor['sepal_length'], data_versicolor['petal_length'])
ax.scatter(data_virginica['sepal_length'], data_virginica['petal_length'])

max_depth = [1, 3, 5, 7]

X = data[['sepal_length', 'petal_length']]
y = data['species']

md = 6

model = DecisionTreeClassifier(max_depth=md)
model.fit(X, y)
y_p1 = model.predict(X_p)

ax.contourf(
    X1_p, 
    X2_p,
    y_p1.reshape(X1_p.shape),
    alpha=0.2,
    levels=2, 
    cmap='rainbow',
    zorder=1,
)

'''
X = data_df_B[['sepal_length', 'petal_length']]
y = data_df_B['species']
j = 0

for md in max_depth:

    model = DecisionTreeClassifier(max_depth=md)
    model.fit(X, y)
    ax[1, j].scatter(data_virginica_B['sepal_length'], data_virginica_B['petal_length'])
    ax[1, j].scatter(data_versicolor_B['sepal_length'], data_versicolor_B['petal_length'])

    y_p = model.predict(X_p)
    ax[1, j].contourf(
        X1_p, 
        X2_p,
        y_p.reshape(X1_p.shape),
        alpha=0.2,
        levels=2, 
        cmap='rainbow',
        zorder=1,
    )
    j += 1'''
plt.show()


# Переобучение присуще всем деревьям принятия решений
#

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier

iris = sns.load_dataset('iris')
species_int = []
for r in iris.values:
    match r[4]:
        case 'setosa':
            species_int.append(1)
        case 'versicolor':
            species_int.append(2)
        case 'virginica':
            species_int.append(3)

species_int_df = pd.DataFrame(species_int)
print(species_int_df.head())

data = iris[['sepal_length', 'petal_length']]
data['species'] = species_int

print(data.head())
print(data.shape)

data_versicolor = data[data['species'] == 2]
data_virginica = data[data['species'] == 3]

data_versicolor_A = data_versicolor.iloc[:25, :]
data_versicolor_B = data_versicolor.iloc[25:, :]


data_virginica_A = data_virginica.iloc[:25, :]
data_virginica_B = data_virginica.iloc[25:, :]

data_df_A = pd.concat([data_virginica_A, data_versicolor_A], ignore_index=True)
data_df_B = pd.concat([data_virginica_B, data_versicolor_B], ignore_index=True)

x1_p = np.linspace(min(data['sepal_length']), max(data['sepal_length']))
x2_p = np.linspace(min(data['petal_length']), max(data['petal_length']))

X1_p, X2_p = np.meshgrid(x1_p, x2_p)

X_p = pd.DataFrame(
    np.vstack([X1_p.ravel(), X2_p.ravel()]).T, columns=['sepal_length', 'petal_length']
)

fix, ax = plt.subplots(2, 4, sharex='col', sharey='row')

max_depth = [1, 3, 5, 7]

X = data_df_A[['sepal_length', 'petal_length']]
y = data_df_A['species']
j = 0
for md in max_depth:

    model = DecisionTreeClassifier(max_depth=md)
    model.fit(X, y)
    ax[0, j].scatter(data_virginica_A['sepal_length'], data_virginica_A['petal_length'])
    ax[0, j].scatter(data_versicolor_A['sepal_length'], data_versicolor_A['petal_length'])

    y_p = model.predict(X_p)
    ax[0, j].contourf(
        X1_p,
        X2_p,
        y_p.reshape(X1_p.shape),
        alpha=0.2,
        levels=2,
        cmap='rainbow',
        zorder=1,
    )
    j += 1

X = data_df_B[['sepal_length', 'petal_length']]
y = data_df_B['species']
j = 0

for md in max_depth:

    model = DecisionTreeClassifier(max_depth=md)
    model.fit(X, y)
    ax[1, j].scatter(data_virginica_B['sepal_length'], data_virginica_B['petal_length'])
    ax[1, j].scatter(data_versicolor_B['sepal_length'], data_versicolor_B['petal_length'])

    y_p = model.predict(X_p)
    ax[1, j].contourf(
        X1_p,
        X2_p,
        y_p.reshape(X1_p.shape),
        alpha=0.2,
        levels=2,
        cmap='rainbow',
        zorder=1,
    )
    j += 1
plt.show()