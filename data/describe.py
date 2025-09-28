#!/usr/bin/env python

import pandas as pd

# Загрузка данных
df = pd.read_csv('ecg_data_clean.csv')

# 1. Общая информация
print("Общая информация о датафрейме:")
df.info()
print("\n")

# 2. Статистическое описание числовых признаков
print("Статистическое описание числовых столбцов:")
print(df.describe())
print("\n")

# 3. Статистическое описание некатегориальных (object) столбцов
print("Уникальные значения и пропуски в нечисловых столбцах:")
for col in df.select_dtypes(include=['object']).columns:
    print(f"– {col}: уникальных={df[col].nunique()}, пропусков={df[col].isna().sum()}")
print("\n")

# 4. Пропуски во всём наборе
print("Пропуски по всем столбцам:")
print(df.isna().sum())
print("\n")

# 5. Строки с любыми «битими» или нечисловыми данными в числовых столбцах
bad_rows = df[df.select_dtypes('number').isnull().any(axis=1)]
print(f"Строк с некорректными (NaN) в числовых столбцах: {len(bad_rows)}")
print(bad_rows)
print("\n")

# 6. Потенциальные выбросы: значения за пределами 1.5×IQR
print("Потенциальные выбросы по каждому числовому столбцу:")
Q1 = df.select_dtypes('number').quantile(0.25)
Q3 = df.select_dtypes('number').quantile(0.75)
IQR = Q3 - Q1
for col in df.select_dtypes('number').columns:
    lower, upper = Q1[col] - 1.5 * IQR[col], Q3[col] + 1.5 * IQR[col]
    mask = (df[col] < lower) | (df[col] > upper)
    count = mask.sum()
    print(f"– {col}: {count} выброс(ов) (поза диапазоном [{lower:.2f}, {upper:.2f}])")
print("\n")

# 7. Дублированные строки
dupes = df.duplicated().sum()
print(f"Дублированных строк: {dupes}")
if dupes:
    print(df[df.duplicated(keep=False)])
