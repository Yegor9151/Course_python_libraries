#   Тема “Обучение с учителем”

#   Задание 1
#   Импортируйте библиотеки pandas и numpy.
import pandas as pd
import numpy as np
#   Загрузите "Boston House Prices dataset" из встроенных наборов данных библиотеки sklearn.
from sklearn.datasets import load_boston

boston = load_boston()
data = boston.data
target = boston.target
feature_names = boston.feature_names
#   Создайте датафреймы X и y из этих данных.
X = pd.DataFrame(data, columns=feature_names)
y = pd.DataFrame(target, columns=['price'])
#   Разбейте эти датафреймы на тренировочные (X_train, y_train) и тестовые (X_test, y_test) с помощью функции
#   train_test_split так, чтобы размер тестовой выборки составлял 30% от всех данных, при этом аргумент random_state
#   должен быть равен 42.
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
#   Создайте модель линейной регрессии под названием lr с помощью класса LinearRegression из модуля
#   sklearn.linear_model.
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
#   Обучите модель на тренировочных данных (используйте все признаки) и сделайте предсказание на тестовых.
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
check_test = pd.DataFrame({'y_test': y_test['price'],
                           'y_pred': y_pred.flatten()},
                          columns=['y_test', 'y_pred'])
print(f'check_test:\n{check_test.head(10)}\n')
#   Вычислите R2 полученных предказаний с помощью r2_score из модуля sklearn.metrics.
from sklearn.metrics import r2_score

R2 = r2_score(y_test, y_pred)
print(f'R2 = {R2}\n')

#   Задание 2
#   Создайте модель под названием model с помощью RandomForestRegressor из модуля sklearn.ensemble.
from sklearn.ensemble import RandomForestRegressor

#   Сделайте агрумент n_estimators равным 1000,
model = RandomForestRegressor(
    n_estimators=1000,  # Сделайте агрумент n_estimators равным 1000,
    max_depth=12,  # max_depth должен быть равен 12
    random_state=42  # и random_state сделайте равным 42.
)
#   Обучите модель на тренировочных данных аналогично тому, как вы обучали модель LinearRegression, но при этом в метод
#   fit вместо датафрейма y_train поставьте y_train.values[:, 0], чтобы получить из датафрейма одномерный массив Numpy,
#   так как для класса RandomForestRegressor в данном методе для аргумента y предпочтительно применение массивов вместо
#   датафрейма.
model.fit(X_train, y_train.values[:, 0])
# Сделайте предсказание на тестовых данных и посчитайте R2. Сравните с результатом из предыдущего задания.
y_pred2 = model.predict(X_test)
check_test2 = pd.DataFrame({'y_test': y_test['price'],
                            'y_pred': y_pred2.flatten()},
                           columns=['y_test', 'y_pred'])
print(f'check_test2:\n{check_test2.head(10)}\n')

R22 = r2_score(y_test, y_pred2)
print(f'R22 = {R22}\n')
#   Напишите в комментариях к коду, какая модель в данном случае работает лучше.
print(f'Разнича в точности моделей: model - lr = {R22 - R2}\n')
# ОТВЕТ: в данном случае модель model созданная при помощи RandomForestRegressor работает на 0.16 лучше,
# чем модель lr созданноая при помощи LinearRegression - это можно увидеть по одной из метрик оценивающих качество
# моделей - r2_score

#   * Задание 3
#   Вызовите документацию для класса RandomForestRegressor,
# ?RandomForestRegressor
#   найдите информацию об атрибуте feature_importances.
# feature_importances_ : array of shape = [n_features]
# The feature importances (the higher, the more important the feature).
#   С помощью этого атрибута найдите сумму всех показателей важности,
print(f'Сумма всех показателей важности: {model.feature_importances_.sum()}')
#   установите, какие два признака показывают наибольшую важность.
sorted_values = np.sort(model.feature_importances_)[::-1]
sorted_index = np.argsort(model.feature_importances_)[::-1]
print(
    f'Первый два наиболее важных признака:\n'
    f'{sorted_index[0]} = {sorted_values[0]}\n'
    f'{sorted_index[1]} = {sorted_values[1]}\n'
)
