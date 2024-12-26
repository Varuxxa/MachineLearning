import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Загрузка данных
df = pd.read_csv('C:/Users/4873634/Documents/input/Crude_Oil_Data.csv')  # Замените на ваш путь к файлу

# 1. Преобразование столбца 'Date' в формат datetime
df['Date'] = pd.to_datetime(df['Date'])

# 2. Добавление нового признака - 7-дневное скользящее среднее
df['7_day_avg'] = df['Close'].rolling(window=7).mean()

# 3. Обработка пропущенных значений
# Заполнение пропущенных значений средними значениями по столбцам
df.fillna(df.mean(), inplace=True)

# 4. Удаление выбросов с помощью IQR
Q1 = df['Close'].quantile(0.25)
Q3 = df['Close'].quantile(0.75)
IQR = Q3 - Q1
df = df[(df['Close'] >= (Q1 - 1.5 * IQR)) & (df['Close'] <= (Q3 + 1.5 * IQR))]

# 5. Стандартизация данных
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df[['Open', 'High', 'Low', 'Close', 'Volume', '7_day_avg']])

# 6. Применение метода KMeans
kmeans = KMeans(n_clusters=3, random_state=42)  # Пример с 3 кластерами
df['Cluster'] = kmeans.fit_predict(scaled_data)

# 7. Визуализация кластеров
plt.figure(figsize=(10,6))
plt.scatter(df['Close'], df['Volume'], c=df['Cluster'], cmap='viridis')
plt.title('Кластеры нефти по цене закрытия и объёму')
plt.xlabel('Цена закрытия')
plt.ylabel('Объём')
plt.show()

# 8. Построение графиков рассеяния для разных признаков
sns.pairplot(df[['Open', 'High', 'Low', 'Close', 'Volume', '7_day_avg', 'Cluster']], hue='Cluster', palette='viridis')
plt.show()

# 9. Оценка оптимального числа кластеров с использованием метода локтя
inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_data)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8, 6))
plt.plot(range(1, 11), inertia, marker='o')
plt.title('Метод локтя для определения оптимального числа кластеров')
plt.xlabel('Количество кластеров')
plt.ylabel('Инерция')
plt.show()

# 10. Выводы
print(f"Количество кластеров: {len(df['Cluster'].unique())}")
