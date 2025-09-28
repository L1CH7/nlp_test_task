#!/usr/bin/env python

import pandas as pd

df = pd.read_csv('ecg_data_raw.csv')
# Оставить строки, где все числовые значения ≤ 29999
df_clean = df[df.select_dtypes('number').lt(29999).all(axis=1)]
df_clean.to_csv('ecg_data_clean.csv', index=False)
print(f"Удалено строк: {len(df) - len(df_clean)}")
