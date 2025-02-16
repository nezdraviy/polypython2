import pandas as pd
import numpy as np


# # 1. Разобраться как использовать мультииндексные ключи в данном примере
index = [
    ('city_1', 2010),
    ('city_1', 2020),
    ('city_2', 2010),
    ('city_2', 2020),
    ('city_3', 2010),
    ('city_3', 2020),
]

population = [
    101,
    201,
    102,
    202,
    103,
    203,
]

index = pd.MultiIndex.from_tuples(index)

# pop = pd.Series(population, index = index)
pop_df = pd.DataFrame(
    {
        'total': population,
        'something': [
            10,
            11,
            12,
            13,
            14,
            15,
        ]
    }, index = index
)

pop_df_1 = pop_df.loc['city_1', 'something']
pop_df_2 = pop_df.loc[['city_1', 'city_3'], ['total', 'something']]
pop_df_3 = pop_df.loc[['city_1', 'city_3'], 'something']

index = pd.MultiIndex.from_product(
    [
        ['city_1', 'city_2'],
        [2010, 2020]
    ],
    names=['city', 'year']
)
columns = pd.MultiIndex.from_product(
    [
        ['person_1', 'person_2', 'person_3'],
        ['job_1', 'job_2']
    ],
    names=['worker', 'job']
)

data = np.random.sample((4, 6))

wrk_df = pd.DataFrame(data, columns = columns, index = index)


# 2. Из получившихся данных выбрать данные по 
# - 2020 году (для всех столбцов)
wrk_df_1 = wrk_df.loc[pd.IndexSlice[:, 2020], :]
print(wrk_df_1)
# - job_1 (для всех строк)
wrk_df_2 = wrk_df.loc[:, pd.IndexSlice[:, 'job_1']]
print(wrk_df_2)
# - для city_1 и job_2 
wrk_df_3 = wrk_df.loc[pd.IndexSlice['city_1', :], pd.IndexSlice[:, 'job_2']]
print(wrk_df_3)


# 3. Взять за основу DataFrame со следующей структурой
index = pd.MultiIndex.from_product(
    [
        ['city_1', 'city_2'],
        [2010, 2020]
    ],
    names=['city', 'year']
)
columns = pd.MultiIndex.from_product(
    [
        ['person_1', 'person_2', 'person_3'],
        ['job_1', 'job_2']
    ],
    names=['worker', 'job']
)

data = np.random.sample((4, 6))

wrk_df = pd.DataFrame(data, columns = columns, index = index)


# Выполнить запрос на получение следующих данных
# - все данные по person_1 и person_3
wrk_df_4 = wrk_df.loc[:, ['person_1', 'person_3']]
print(wrk_df_4)
# - все данные по первому городу и первым двум person-ам (с использование срезов)
wrk_df_5 = wrk_df.loc['city_1', ['person_1', 'person_2']]
print(wrk_df_5)
# Приведите пример (самостоятельно) с использованием pd.IndexSlice
wrk_df_6 = wrk_df.loc[pd.IndexSlice[:, 2010], pd.IndexSlice['person_1', :]]
print(wrk_df_6)


#4. Привести пример использования inner и outer джойнов для Series (данные примера скорее всего нужно изменить)
ser1 = pd.DataFrame(['a', 'b', 'c'], index = [3, 4, 5])
ser2 = pd.DataFrame(['b', 'c', 'f'], index = [4, 5, 6])

print(pd.concat([ser1, ser2], axis = 1, join = 'outer'))
print(pd.concat([ser1, ser2], axis = 1, join = 'inner'))