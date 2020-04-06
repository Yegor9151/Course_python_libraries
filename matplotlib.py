#   *Задание 4
#   В этом задании мы будем работать с датасетом, с которым мы уже знакомы по домашнему заданию по библиотеке
#   Matplotlib, это датасет Credit Card Fraud Detection.
#   Для этого датасета мы будем решать задачу классификации - будем определять, какие из транзакциции по кредитной карте
#   являются мошенническими.
#   Данный датасет сильно несбалансирован (так как случаи мошенничества относительно редки), так что применение метрики
#   accuracy не принесет пользы и не поможет выбрать лучшую модель.
#   Мы будем вычислять AUC, то есть площадь под кривой ROC.
#   Импортируйте из соответствующих модулей RandomForestClassifier, GridSearchCV и train_test_split.
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split

#   Загрузите датасет creditcard.csv
data_set = '../lesson_2/creditcard.csv'


def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.
    """
    start_mem = df.memory_usage().sum() / 1024 ** 2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024 ** 2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df


#   и создайте датафрейм df.
df = pd.read_csv(data_set)
reduce_mem_usage(df)
#   С помощью метода value_counts с аргументом normalize=True убедитесь в том, что выборка несбалансирована.
print(f"\n{df['Class'].value_counts(normalize=True)}\n")
#   Используя метод info, проверьте, все ли столбцы содержат числовые данные и нет ли в них пропусков.
df.info()
print()
#   Примените следующую настройку, чтобы можно было просматривать все столбцы датафрейма:
#   pd.options.display.max_columns = 100.
pd.options.display.max_columns = 100
#   Просмотрите первые 10 строк датафрейма df.
print(
    f'Первыйе 10 строк df:\n'
    f'{df.head(10)}\n'
)
#   Создайте датафрейм X из датафрейма df, исключив столбец Class.
X = df.drop('Class', axis=1)
X.info()
print()
#   Создайте объект Series под названием y из столбца Class.
y = pd.Series(df['Class'])
#   Разбейте X и y на тренировочный и тестовый наборы данных при помощи функции train_test_split, используя аргументы:
#   test_size=0.3, random_state=100, stratify=y.
#   У вас должны получиться объекты X_train, X_test, y_train и y_test.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100, stratify=y)
#   Просмотрите информацию о их форме.
print(
    f'X_train:\n{X_train}\n'
    f'\nX_test:\n{X_test}\n'
    f'\ny_train:\n{y_train}\n'
    f'\ny_test:\n{y_test}\n'
)
#   Для поиска по сетке параметров задайте такие параметры:
#   parameters = [{'n_estimators': [10, 15],
#   'max_features': np.arange(3, 5),
#   'max_depth': np.arange(4, 7)}]
parameters = [{
    'n_estimators': [10, 15],
    'max_features': np.arange(3, 5),
    'max_depth': np.arange(4, 7)
}]
#   Создайте модель GridSearchCV со следующими аргументами:
#   estimator=RandomForestClassifier(random_state=100),
#   param_grid=parameters,
#   scoring='roc_auc',
#   cv=3.
modelGSCV = GridSearchCV(
    estimator=RandomForestClassifier(random_state=100),
    param_grid=parameters,
    scoring='roc_auc',
    cv=3
)
#   Обучите модель на тренировочном наборе данных (может занять несколько минут).
modelGSCV.fit(X_train, y_train)
#   Просмотрите параметры лучшей модели с помощью атрибута best_params_.
print(
    f'Параметры лучшей модели:\n'
    f'{modelGSCV.best_params_}\n'
)
#   Предскажите вероятности классов с помощью полученнной модели и метода predict_proba.
y_pred_proba = modelGSCV.predict_proba(X_test)
print(
    f'Вероятности классов y_pred_proba:\n'
    f'{y_pred_proba}\n'
)
#   Из полученного результата (массив Numpy) выберите столбец с индексом 1 (вероятность класса 1) и запишите в массив
#   y_pred_proba.
y_pred_proba = y_pred_proba[:, 1]
#   Из модуля sklearn.metrics импортируйте метрику roc_auc_score.
from sklearn.metrics import roc_auc_score

#   Вычислите AUC на тестовых данных и сравните с результатом, полученным на тренировочных данных, используя в качестве
#   аргументовмассивы y_test и y_pred_proba.
print(f'Сравнение AUC с результатами: {roc_auc_score(y_test, y_pred_proba)}')
