import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('Cars Datasets 2025.csv', encoding='latin1')

print('\nПервые 5 строк датасета:\n\n', df.head())

print('\nРазмер датасета:\n\n', df.shape)
print('\nКоличество строк:\n\n', df.shape[0])
print('\nКоличество столбцов:\n\n', df.shape[1])
print('\nСписок столбцов:\n\n', df.columns.tolist())
print('\nТипы данных:\n\n', df.dtypes)
print('\nКоличество пропусков:\n\n', df.isnull().sum().sum())

# HorsePower → убрать ' hp'
df['HorsePower'] = df['HorsePower'].str.replace(' hp', '', regex=True)

# Total Speed → убрать ' km/h'
df['Total Speed'] = df['Total Speed'].str.replace(' km/h', '', regex=True)

# Performance(0 - 100 )KM/H → убрать ' sec'
df['Performance(0 - 100 )KM/H'] = df['Performance(0 - 100 )KM/H'].str.replace(' sec', '', regex=True)

# Cars Prices → убрать '$', ',', пробелы
df['Cars Prices'] = df['Cars Prices'].str.replace(r'[^\d.-]', '', regex=True)
df['Cars Prices'] = df['Cars Prices'].str.split('-').str[0]

# CC/Battery Capacity → убрать ',' и ' cc'
df['CC/Battery Capacity'] = df['CC/Battery Capacity'].str.replace(',', '', regex=True)
df['CC/Battery Capacity'] = df['CC/Battery Capacity'].str.replace(' cc', '', regex=True)

# Torque → убрать ' Nm'
df['Torque'] = df['Torque'].str.replace(' Nm', '', regex=True)

print('\nПосле очистки текстовых единиц:\n\n')
print(df[['HorsePower', 'Total Speed', 'Performance(0 - 100 )KM/H', 
          'Cars Prices', 'CC/Battery Capacity', 'Torque', 'Seats']].head())

# Приведение к числовым типам с errors='coerce'
df['HorsePower'] = pd.to_numeric(df['HorsePower'], errors='coerce')
df['Total Speed'] = pd.to_numeric(df['Total Speed'], errors='coerce')
df['Performance(0 - 100 )KM/H'] = pd.to_numeric(df['Performance(0 - 100 )KM/H'], errors='coerce')
df['Cars Prices'] = pd.to_numeric(df['Cars Prices'], errors='coerce')
df['CC/Battery Capacity'] = pd.to_numeric(df['CC/Battery Capacity'], errors='coerce')
df['Torque'] = pd.to_numeric(df['Torque'], errors='coerce')
df['Seats'] = pd.to_numeric(df['Seats'], errors='coerce')

print('\nТипы данных после преобразования:\n\n')
print(df.dtypes)

# Удаление дубликатов
initial_rows = len(df)
df.drop_duplicates(inplace=True)
final_rows = len(df)
print('\nУдалено дубликатов:\n\n', (initial_rows - final_rows))

# Поиск min/max/mean/median для числовых столбцов
numeric_columns = ['CC/Battery Capacity', 'HorsePower', 'Total Speed', 
                  'Performance(0 - 100 )KM/H', 'Cars Prices', 'Seats', 'Torque']
print('\nСтатистика для числовых столбцов:\n\n')
stats = df[numeric_columns].agg(['min', 'max', 'mean', 'median'])
stats_formatted = stats.copy()
stats_formatted['Cars Prices'] = stats_formatted['Cars Prices'].apply(
    lambda x: f"${x:,.2f}" if pd.notna(x) else "NaN"
)
print(stats_formatted)

# Заполнение пропусков
numeric_medians = df[numeric_columns].median()
df.fillna(numeric_medians, inplace=True)
df[['Company Names', 'Cars Names', 'Engines', 'Fuel Types']] = df[['Company Names', 'Cars Names', 'Engines', 'Fuel Types']].fillna('Unknown')
print('\nПропуски после заполнения:\n\n')
print(df.isnull().sum())

# Самая дорогая модель
if df['Cars Prices'].isna().all():
    print('\nНет данных о ценах автомобилей')
else:
    most_expensive_idx = df['Cars Prices'].idxmax()
    most_expensive = df.loc[most_expensive_idx]
    print('\nСамая дорогая модель:\n\n', most_expensive['Company Names'], most_expensive['Cars Names'])
    print('\nЦена:\n\n', most_expensive['Cars Prices'])

# Самая быстрая модель
fastest_idx = df['Total Speed'].idxmax()
fastest = df.loc[fastest_idx]
print(f"\nСамая быстрая модель: {fastest['Company Names']} {fastest['Cars Names']}")
print(f"Максимальная скорость: {fastest['Total Speed']} км/ч")

# Самая мощная модель
most_powerful_idx = df['HorsePower'].idxmax()
most_powerful = df.loc[most_powerful_idx]
print(f"\nСамая мощная модель: {most_powerful['Company Names']} {most_powerful['Cars Names']}")
print(f"Мощность: {most_powerful['HorsePower']} л.с.")

plt.figure(figsize=(13, 5))
# Гистограмма цен
plt.subplot(1, 3, 1)
plt.hist(df['Cars Prices'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
plt.title('Распределение цен автомобилей')
plt.xlabel('Цена ($)')
plt.ylabel('Количество')

# Гистограмма мощности
plt.subplot(1, 3, 2)
plt.hist(df['HorsePower'], bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
plt.title('Распределение мощности')
plt.xlabel('Мощность (л.с.)')
plt.ylabel('Количество')

# Гистограмма скорости
plt.subplot(1, 3, 3)
plt.hist(df['Total Speed'], bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
plt.title('Распределение максимальной скорости')
plt.xlabel('Скорость (км/ч)')
plt.ylabel('Количество')

plt.tight_layout()
plt.show()

brand_counts = df['Company Names'].value_counts()

# Строим круговую диаграмму с выносками
top_10_brands = brand_counts.head(10)
plt.figure(figsize=(10, 6))
wedges, texts, autotexts = plt.pie(top_10_brands.values, 
                                  labels=top_10_brands.index, 
                                  autopct='%1.1f%%', 
                                  startangle=90,
                                  colors=plt.cm.Set3(np.linspace(0, 1, 10)))
plt.title('Топ-10 брендов по количеству моделей', fontsize=16, fontweight='bold')
plt.axis('equal')
for autotext in autotexts:
    autotext.set_color('black')
    autotext.set_fontweight('bold')
plt.tight_layout()
plt.show()

# Диаграммы размаха
top_10_brands = df['Company Names'].value_counts().head(10).index
top_brands_df = df[df['Company Names'].isin(top_10_brands)]
fig, axes = plt.subplots(1, 3, figsize=(12, 6))

axes[0].boxplot([top_brands_df[top_brands_df['Company Names'] == brand]['Cars Prices'].dropna() 
                for brand in top_10_brands], labels=top_10_brands)
axes[0].set_title('Распределение цен по брендам')
axes[0].set_ylabel('Цена ($)')
axes[0].tick_params(axis='x', rotation=45)

axes[1].boxplot([top_brands_df[top_brands_df['Company Names'] == brand]['HorsePower'].dropna() 
                for brand in top_10_brands], labels=top_10_brands)
axes[1].set_title('Распределение мощности по брендам')
axes[1].set_ylabel('Мощность (л.с.)')
axes[1].tick_params(axis='x', rotation=45)

axes[2].boxplot([top_brands_df[top_brands_df['Company Names'] == brand]['Total Speed'].dropna() 
                for brand in top_10_brands], labels=top_10_brands)
axes[2].set_title('Распределение скорости по брендам')
axes[2].set_ylabel('Скорость (км/ч)')
axes[2].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

# Дополнительный анализ: вычисляем медианы и стандартные отклонения
print('\nМедианные значения и стандартные отклонения для топ-10 брендов:')

for brand in top_10_brands:
    brand_data = top_brands_df[top_brands_df['Company Names'] == brand]
    
    price_median = brand_data['Cars Prices'].median()
    price_std = brand_data['Cars Prices'].std()
    
    hp_median = brand_data['HorsePower'].median()
    hp_std = brand_data['HorsePower'].std()
    
    speed_median = brand_data['Total Speed'].median()
    speed_std = brand_data['Total Speed'].std()
    
    print(f"\n{brand}:")
    print(f"  Цены: медиана = ${price_median:,.0f}, ст. отклонение = ${price_std:,.0f}")
    print(f"  Мощность: медиана = {hp_median:.0f} л.с., ст. отклонение = {hp_std:.0f} л.с.")
    print(f"  Скорость: медиана = {speed_median:.0f} км/ч, ст. отклонение = {speed_std:.0f} км/ч")

# Сводная статистика по всем топ-10 брендам
print(f"Общая медиана цен: ${top_brands_df['Cars Prices'].median():,.0f}")
print(f"Общая медиана мощности: {top_brands_df['HorsePower'].median():.0f} л.с.")
print(f"Общая медиана скорости: {top_brands_df['Total Speed'].median():.0f} км/ч")

# Вычисляем матрицу корреляций только для числовых столбцов
correlation_matrix = df.corr(numeric_only=True)

# Создаем heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, 
            annot=True, 
            cmap='coolwarm', 
            center=0,
            square=True,
            fmt='.2f',
            linewidths=0.5)

plt.title('\nМатрица корреляций числовых признаков\n\n', fontsize=16, pad=20)
plt.tight_layout()
plt.show()

# Находим 3 самые сильные связи с ценой (исключая корреляцию цены с самой собой)
price_correlations = correlation_matrix['Cars Prices'].drop('Cars Prices')

# Сортируем по абсолютному значению корреляции (силе связи)
top_3_strongest = price_correlations.abs().sort_values(ascending=False).head(3)

print('\nТоп-3 самые сильные связи с ценой:\n\n')
for feature, abs_corr in top_3_strongest.items():
    actual_corr = price_correlations[feature]
    direction = 'положительная' if actual_corr > 0 else 'отрицательная'
    print(f"{feature}: {actual_corr:.3f} ({direction} связь)")
    
    # Интерпретация силы связи
    if abs_corr > 0.7:
        strength = 'очень сильная'
    elif abs_corr > 0.5:
        strength = 'сильная'
    elif abs_corr > 0.3:
        strength = 'умеренная'
    else:
        strength = 'слабая'
    print(f"  → {strength} связь с ценой")

# Дополнительно: визуализация топ-3 связей с ценой
top_3_features = top_3_strongest.index

fig, axes = plt.subplots(1, 3, figsize=(12, 5))

for i, feature in enumerate(top_3_features):
    axes[i].scatter(df[feature], df['Cars Prices'], alpha=0.6)
    axes[i].set_xlabel(feature)
    axes[i].set_ylabel('Цена ($)')
    
    # Добавляем линию тренда
    z = np.polyfit(df[feature].dropna(), df['Cars Prices'].dropna(), 1)
    p = np.poly1d(z)
    axes[i].plot(df[feature], p(df[feature]), "r--", alpha=0.8)
    
    corr_value = correlation_matrix.loc['Cars Prices', feature]
    axes[i].set_title(f'{feature}\nкорреляция: {corr_value:.3f}')

plt.tight_layout()
plt.show()
