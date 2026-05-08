import pandas as pd
df = pd.read_csv('laptop_scrap_data.csv')

print("Top 5 Rows:")
print(df.head())
print()

print("Bottom 5 Rows:")
print(df.tail())
print()

print("Shape:")
print('Rows: ', df.shape[0])
print('Columns: ', df.shape[1])
print()

print("Null Values:")
print(df.isnull().sum())
print()

print("Filling Null Values:")

df['Inches'] = df['Inches'].fillna(df['Inches'].mean())
df['Price'] = df['Price'].fillna(df['Price'].mean())
df['Ram_GB'] = df['Ram_GB'].fillna(df['Ram_GB'].mean())
df['Weight_kg'] = df['Weight_kg'].fillna(df['Weight_kg'].mean())

df['Company'] = df['Company'].fillna(df['Company'].mode()[0])
df['OpSys'] = df['OpSys'].fillna(df['OpSys'].mode()[0])
df['Cpu'] = df['Cpu'].fillna(df['Cpu'].mode()[0])

print("Null values after filling:")
print(df.isnull().sum())
print()

print("Data Types:")
print(df.dtypes)
print()

print("Statistical Summary:")
print(df.describe())
print()

print("Dataset Info:")
print(df.info())
print()

print("Column Names:")
print(df.columns.tolist())
print()

print("Unique Values:")
print("Unique Companies: ", df['Company'].nunique())
print("Unique OS: ", df['OpSys'].nunique())
print("Unique Storage Types: ", df['Storage_Type'].nunique())
print()

print("Company wise Count:")
print(df['Company'].value_counts())
print()

print("Operating System Count:")
print(df['OpSys'].value_counts())
print()

print("Storage Type Count:")
print(df['Storage_Type'].value_counts())
print()

print("Correlation Matrix:")
df[['Price', 'Ram_GB', 'Weight_kg']].corr()
print()

print("Price Statistics:")
print("Mean Price: ", df['Price'].mean())
print("Median Price: ", df['Price'].median())
print("Min Price: ", df['Price'].min())
print("Max Price: ", df['Price'].max())
print("Std Dev: ", df['Price'].std())