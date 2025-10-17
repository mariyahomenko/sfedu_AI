import pandas as pd
import numpy
import matplotlib.pyplot as plt

df = pd.read_csv("wikidata_universities_sample.csv") 
head = df.head(5)
tail = df.tail(5) 
print('\nПервые и последние 5 строк:\n\n', head, tail)

print('\n\nТипы данных:\n\n')
size = df.shape 
type = df.info()
print('\nРазмер таблицы, наличие пропусков:\n\n', size, '\n\n', type)

# Форматирование даты
df['inception'] = pd.to_datetime(df['inception'], errors='coerce', utc=True)
# Удалим дубликаты
df.drop_duplicates() 
# Заполним пропуски
df.fillna("Unknown") 

coutries = df['countryLabel'].nunique()
print('\nКоличество уникальных стран:\n\n', coutries)

min = df['inception'].dt.year.min()
max = df['inception'].dt.year.max()
print('\nМинимальный и максимальный годы основания университетов:\n\n', min.astype(int), 'и', max.astype(int))

mean = df['inception'].dt.year.mean()
median = df['inception'].dt.year.median()
print('\nСредний и медианный год основания университетов:\n\n', mean.astype(int), 'и', median.astype(int))

# Отсортируем таблицу по году основания университетов
df.sort_values('inception') 

#Создадим столбец с годом основания
df['year'] = df['inception'].dt.year

# Посчитаем возраст университетов на 2025 год
df['age'] = 2025 - df['year']

#Создадим категориальный столбец, где Ancient — старше 300 лет, Old — 100–300 лет, Modern — младше 100 лет
df['age_group'] = df['age'].apply( lambda x: 'Ancient' if x > 300 else ('Old' if x >= 100 else 'Modern') ) 

min_age = df['age'].min()
max_age = df['age'].max()
print('\nМинимальный и максимальный возраст университетов:\n\n', min_age.astype(int), 'и', max_age.astype(int))

mean_age = df['age'].mean()
median_age = df['age'].median()
print('\nСредний и медианный возраст университетов:\n\n', mean_age.astype(int), 'и', median_age.astype(int))

top_10_countries = df['countryLabel'].value_counts().head(10)
print('\n10 стран с наибольшим количеством университетов:\n\n', top_10_countries)

mean_age_by_country = df[df['countryLabel'].isin(top_10_countries.index)].groupby('countryLabel')['age'].mean()
print('\nСредний возраст по странам:\n\n', mean_age_by_country.astype(int))


top_10_country_names = top_10_countries.index
top_10_df = df[df['countryLabel'].isin(top_10_country_names)]
oldest = top_10_df.loc[top_10_df['age'].idxmax()]
country_with_oldest = oldest['countryLabel']
print('\nСтрана с самым старым университетом:\n\n', country_with_oldest)
youngest = top_10_df.loc[top_10_df['age'].idxmin()]
country_with_youngest = youngest['countryLabel']
print('\nСтрана с самым молодым университетом:\n\n', country_with_youngest)

print('\nКоличество университетов по возрастным категориям:\n\n', df['age_group'].value_counts())

print('\nПостроим гистограмму распределения годов основания.\n\n')
df['inception'].hist(bins=20)
plt.title('Распределение годов основания университетов')
plt.xlabel('Год основания')
plt.ylabel('Количество университетов')
plt.show()

print('\nПостроим столбчатую диаграмму топ-10 стран.\n\n')
df['countryLabel'].value_counts().head(10).plot(kind='bar')
plt.title('Топ-10 стран по количеству университетов')
plt.xlabel('Страна')
plt.ylabel('Количество университетов')
plt.show()

print('\nПостроим круговую диаграмму распределения категорий age_group.\n\n')
df['age_group'].value_counts().plot(kind='pie', autopct='%1.1f%%') 
plt.title('Распределение университетов по возрастным категориям')
plt.ylabel('')
plt.show()

print('\n' \
'Выводы по анализу университетов:\n\n' \
'Лидеры по количеству университетов: США, Япония, Китай, Корея и Россия возглавляют рейтинг, ' \
'что отражает их инвестиции в высшее образование.\n' \
'Средний возраст вузов составляет около 75 лет, что указывает на богатые академические традиции ' \
'и долгую историю развития высшего образования в большинстве анализируемых стран.\n' \
'Наиболее активные годы основания приходятся на XX века, особенно периоды индустриализации ' \
'и после Второй мировой войны, когда наблюдался бум создания новых учебных заведений.\n' \
'Преобладают современные университеты: категория "Modern" составляет более 70% от общего числа, ' \
'что свидетельствует о активном развитии высшего образования в последние столетия и создании новых вузов.\n' \
'Историческое наследие сохраняется: несмотря на преобладание современных вузов, значительная доля "Ancient" и "Old" ' \
'университетов (около 20-30%) демонстрирует сохранение академических традиций и непрерывность образовательных институтов.\n\n' \
'Выводы по работе с pandas:\n\n' \
'Эффективная обработка данных: pandas позволила легко работать с большим объемом данных об университетах, ' \
'выполняя фильтрацию, группировку и агрегацию без написания сложного кода.\n' \
'Простой анализ распределений: с помощью методов value_counts() и hist() быстро проанализировали распределения ' \
'по странам и годам основания, выявив закономерности в данных.\n' \
'Интуитивная визуализация: встроенные методы построения графиков (plot(), hist()) ' \
'беспечили быстрое создание информативных диаграмм прямо из DataFrame без дополнительных библиотек.\n' \
'Удобство работы с категориальными данными: создание возрастных категорий через apply() с lambda-функцией ' \
'показало гибкость pandas в преобразовании числовых данных в категориальные для более глубокого анализа.')