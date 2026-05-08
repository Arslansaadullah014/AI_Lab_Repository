import pandas as pd
import numpy as np

df = pd.read_csv('laptop_scrap_data.csv')

print('We have {} rows.'.format(df.shape[0]))
print('We have {} columns'.format(df.shape[1]))

print(np.sum(pd.isnull(df)))

print(df['Company'].unique())
print(df['OpSys'].unique())
print(df['Storage_Type'].unique())

num = df['Inches'].mode()[0]
df['Inches'] = df['Inches'].fillna(num)

num = df['Price'].mode()[0]
df['Price'] = df['Price'].fillna(num)

num = df['Ram_GB'].mode()[0]
df['Ram_GB'] = df['Ram_GB'].fillna(num)

num = df['SSD'].mode()[0]
df['SSD'] = df['SSD'].fillna(num)

num = df['Company'].mode()[0]
df['Company'] = df['Company'].fillna(num)

num = df['OpSys'].mode()[0]
df['OpSys'] = df['OpSys'].fillna(num)

df = df.dropna()

print(df.isnull().sum())

print(df.dtypes)

df['Inches'] = df['Inches'].astype(np.int64)
df['TouchScreen'] = df['TouchScreen'].astype(np.int64)
df['Ips'] = df['Ips'].astype(np.int64)
df['X_res'] = df['X_res'].astype(np.int64)
df['Y_res'] = df['Y_res'].astype(np.int64)
df['Dedicated_Gpu'] = df['Dedicated_Gpu'].astype(np.int64)
df['Ram_GB'] = df['Ram_GB'].astype(np.int64)
df['SSD'] = df['SSD'].astype(np.int64)
df['HHD'] = df['HHD'].astype(np.int64)
df['Total_Storage_GB'] = df['Total_Storage_GB'].astype(np.int64)
df['Price'] = df['Price'].astype(np.int64)

print(df.dtypes)