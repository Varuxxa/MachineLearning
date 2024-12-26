import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# 1. Загрузка данных
df = pd.read_csv('C:/Users/4873634/Documents/input/Mall_Customers.csv')

# 2. Добавление нового аттрибута
# Создадим новый аттрибут, который будет сочетанием "дохода" и "оценки трат"
df['Income_Spending_Score'] = df['Annual Income (k$)'] * df['Spending Score (1-100)']

# 3. Причесывание данных
# Удаление дубликатов
df = df.drop_duplicates()

# Обработка пропущенных значений
# Заполняем пропущенные значения только в числовых столбцах средними значениями
numerical_columns = df.select_dtypes(include=['number']).columns
df[numerical_columns] = df[numerical_columns].fillna(df[numerical_columns].mean())

# 4. Построение графиков
# 4.1 График зависимости возраста от дохода
plt.figure(figsize=(10, 6))
plt.scatter(df['Age'], df['Annual Income (k$)'], c='blue', label='Возраст vs Доход')
plt.title('Зависимость возраста от дохода')
plt.xlabel('Возраст')
plt.ylabel('Доход (k$)')
plt.show()

# 4.2 График зависимости возраста от Spending Score
plt.figure(figsize=(10, 6))
plt.scatter(df['Age'], df['Spending Score (1-100)'], c='green', label='Возраст vs Spending Score')
plt.title('Зависимость возраста от Spending Score')
plt.xlabel('Возраст')
plt.ylabel('Spending Score (1-100)')
plt.show()

# 4.3 График зависимости дохода от Spending Score
plt.figure(figsize=(10, 6))
plt.scatter(df['Annual Income (k$)'], df['Spending Score (1-100)'], c='red', label='Доход vs Spending Score')
plt.title('Зависимость дохода от Spending Score')
plt.xlabel('Доход (k$)')
plt.ylabel('Spending Score (1-100)')
plt.show()

# 5. Стандартизация данных (нужно для корректной работы KMeans)
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']])

# 6. Применение KMeans
kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(scaled_data)

# 7. Визуализация кластеров
plt.figure(figsize=(10, 6))
plt.scatter(df['Age'], df['Annual Income (k$)'], c=df['Cluster'], cmap='viridis')
plt.title('Кластеры клиентов по возрасту и доходу')
plt.xlabel('Возраст')
plt.ylabel('Доход (k$)')
plt.show()

# 8. Оценка оптимального числа кластеров с использованием метода локтя
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

# 9. Выводы
print(f"Количество кластеров: {len(df['Cluster'].unique())}")
