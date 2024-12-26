import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

df = pd.read_csv('C:/Users/4873634/Documents/input/Mall_Customers.csv')

# Создаем бинарную метку: если Spending Score > 50, то класс 1 (высокие траты), иначе класс 0 (низкие траты)
df['SpendingClass'] = (df['Spending Score (1-100)'] > 50).astype(int)

# Определяем признаки (X) и целевой атрибут (y)
X = df.drop(['CustomerID', 'Gender', 'Spending Score (1-100)', 'SpendingClass'], axis=1)
y = df['SpendingClass']

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Стандартизация данных
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Инициализация моделей
knn = KNeighborsClassifier(n_neighbors=5)
tree = DecisionTreeClassifier(random_state=42)

# Обучение моделей
knn.fit(X_train_scaled, y_train)
tree.fit(X_train_scaled, y_train)

# Предсказания
y_pred_knn = knn.predict(X_test_scaled)
y_pred_tree = tree.predict(X_test_scaled)

# Вычисление метрик
metrics = {
    "Accuracy": [accuracy_score(y_test, y_pred_knn), accuracy_score(y_test, y_pred_tree)],
    "Precision": [precision_score(y_test, y_pred_knn), precision_score(y_test, y_pred_tree)],
    "Recall": [recall_score(y_test, y_pred_knn), recall_score(y_test, y_pred_tree)],
    "F1-Score": [f1_score(y_test, y_pred_knn), f1_score(y_test, y_pred_tree)],
    "ROC AUC": [roc_auc_score(y_test, knn.predict_proba(X_test_scaled)[:,1]),
                roc_auc_score(y_test, tree.predict_proba(X_test_scaled)[:,1])]
}

# Создание DataFrame для метрик
metrics_df = pd.DataFrame(metrics, index=["kNN", "Decision Tree"])

# Вывод результатов в табличном виде
print(metrics_df)

# Визуализация ROC-кривых
fpr_knn, tpr_knn, _ = roc_curve(y_test, knn.predict_proba(X_test_scaled)[:,1])
fpr_tree, tpr_tree, _ = roc_curve(y_test, tree.predict_proba(X_test_scaled)[:,1])

plt.figure(figsize=(8, 6))
plt.plot(fpr_knn, tpr_knn, color='blue', label='kNN ROC curve')
plt.plot(fpr_tree, tpr_tree, color='green', label='Decision Tree ROC curve')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves')
plt.legend()
plt.show()
