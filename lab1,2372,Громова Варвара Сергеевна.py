import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Загрузка данных
df = pd.read_csv('C:/Users/4873634/Documents/input/Crude_Oil_Data.csv')  # Укажите путь к вашему CSV файлу
# 1. Описание датасета
print("Описание датасета:")
print("\nПредметная область: Энергетика и экономика, анализ/прогнозирование цен на нефть.")
print("Источник данных: https://www.kaggle.com/datasets/mhassansaboor/crude-oil-stock-dataset-2000-2024")
print("Тип данных: Реальные данные о ценах на нефтянные акции.")
print("\nАтрибуты и их типы:")
print(df.dtypes)
print("\nПример данных:")
print(df.head())


# Получение статистики по числовым данным
print(df.describe())

# Преобразование столбца 'Date' в формат datetime, если это необходимо
df['Date'] = pd.to_datetime(df['Date'])

# Проверка пропущенных значений
print("\nПроверка пропущенных значений:")
print(df.isnull().sum())

# Замена пропущенных значений на среднее для числовых столбцов
df.fillna(df.mean(), inplace=True)

# Проверка данных после обработки
print("\nПосле обработки пропущенных значений:")
print(df.isnull().sum())

# 1. Гистограмма цен закрытия
plt.figure(figsize=(10,6))
sns.histplot(df['Close'], kde=True)
plt.title('Гистограмма цен закрытия акций на нефть')
plt.show()

# 2. Анализ выбросов с использованием IQR (межквартильный размах)
Q1 = df['Close'].quantile(0.25)
Q3 = df['Close'].quantile(0.75)
IQR = Q3 - Q1

# Выбросы
outliers = (df['Close'] < (Q1 - 1.5 * IQR)) | (df['Close'] > (Q3 + 1.5 * IQR))
print(f"Количество выбросов: {outliers.sum()}")

# 3. Среднее значение и стандартное отклонение для столбцов 'Open', 'High', 'Low', 'Close', 'Volume'
print("\nСреднее значение:")
print(df[['Open', 'High', 'Low', 'Close', 'Volume']].mean())  # Среднее значение
print("\nСтандартное отклонение:")
print(df[['Open', 'High', 'Low', 'Close', 'Volume']].std())   # Стандартное отклонение

# 4. Корреляция между числовыми столбцами
corr = df.corr()

# Тепловая карта корреляции
plt.figure(figsize=(8,6))
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Корреляция между параметрами')
plt.show()

# 5. Построение графиков рассеяния
sns.pairplot(df[['Open', 'High', 'Low', 'Close']])
plt.show()
