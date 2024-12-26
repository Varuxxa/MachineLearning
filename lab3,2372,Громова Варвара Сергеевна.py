import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, roc_curve
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt

wine_data = pd.read_csv('C:/Users/4873634/Documents/input/WineQT.csv')
# 1. Описание датасета
print("Описание датасета:")
print("\nПредметная область: Анализ качества вина")
print("Источник данных: https://www.kaggle.com/datasets/yasserh/wine-quality-dataset?resource=download")
print("Тип данных: Реальные данные")
print("\nАтрибуты и их типы:")
print(wine_data.dtypes)
print("\nПример данных:")
print(wine_data.head())

# Преобразование столбца quality в категории
def categorize_quality(quality):
    if quality <= 5:
        return 0  # Низкое качество
    elif quality == 6:
        return 1  # Среднее качество
    else:
        return 2  # Высокое качество

wine_data['quality_category'] = wine_data['quality'].apply(categorize_quality)

# Разделение данных на признаки (X) и целевую переменную (y)
X = wine_data.drop(columns=['quality', 'Id', 'quality_category'])
y = wine_data['quality_category']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# kNN модель
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
knn_predictions = knn.predict(X_test)
knn_probabilities = knn.predict_proba(X_test)

# Decision Tree модель
tree = DecisionTreeClassifier(random_state=42)
tree.fit(X_train, y_train)
tree_predictions = tree.predict(X_test)
tree_probabilities = tree.predict_proba(X_test)

# Оценка моделей (Точность, Полнота, F-мера)
print("\nОтчет классификации для модели kNN:")
print(classification_report(y_test, knn_predictions, target_names=['Низкое качество', 'Среднее качество', 'Высокое качество']))
print(f"Точность модели kNN: {accuracy_score(y_test, knn_predictions):.2f}")

print("\nОтчет классификации для модели дерева решений:")
print(classification_report(y_test, tree_predictions, target_names=['Низкое качество', 'Среднее качество', 'Высокое качество']))
print(f"Точность модели дерева решений: {accuracy_score(y_test, tree_predictions):.2f}")

# Построение ROC-кривых и расчет AUC
# Бинаризация классов для ROC-кривых
y_test_binarized = label_binarize(y_test, classes=[0, 1, 2])

# ROC и AUC для kNN
knn_roc_auc = roc_auc_score(y_test_binarized, knn_probabilities, multi_class='ovr')
# ROC и AUC для дерева решений
tree_roc_auc = roc_auc_score(y_test_binarized, tree_probabilities, multi_class='ovr')

# Построение ROC-кривых
plt.figure(figsize=(12, 8))
for i in range(3):  # Для каждого класса
    # ROC-кривые для kNN
    fpr_knn, tpr_knn, _ = roc_curve(y_test_binarized[:, i], knn_probabilities[:, i])
    plt.plot(fpr_knn, tpr_knn, label=f'kNN Класс {i} (AUC = {roc_auc_score(y_test_binarized[:, i], knn_probabilities[:, i]):.2f})')

    # ROC-кривые для дерева решений
    fpr_tree, tpr_tree, _ = roc_curve(y_test_binarized[:, i], tree_probabilities[:, i])
    plt.plot(fpr_tree, tpr_tree, linestyle='--', label=f'Дерево решений Класс {i} (AUC = {roc_auc_score(y_test_binarized[:, i], tree_probabilities[:, i]):.2f})')

# Настройки графика
plt.title('ROC-кривые для моделей kNN и дерева решений')
plt.xlabel('Доля ложных срабатываний (FPR)')
plt.ylabel('Доля истинных срабатываний (TPR)')
plt.legend(loc='best')
plt.grid()
plt.show()

# Вывод общего AUC
print(f"\nОбщая площадь под кривой (AUC) для kNN: {knn_roc_auc:.2f}")
print(f"Общая площадь под кривой (AUC) для дерева решений: {tree_roc_auc:.2f}")
