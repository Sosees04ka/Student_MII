import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Загрузка данных из файла Excel
dataset = pd.read_excel('Yamana_Gold.xlsx')

# Извлечение цен открытия и закрытия
opening_prices = dataset['Low']
closing_prices = dataset['Close']

# Преобразование данных в двумерный массив
X = np.array(list(zip(opening_prices, closing_prices)))


def kmeans(X, k, max_iters=100):
    # Инициализация центроидов случайными точками из набора данных
    np.random.seed(42)
    centroids = X[np.random.choice(range(X.shape[0]), size=k, replace=False)]

    for _ in range(max_iters):
        # Расчет расстояний между точками и центроидами
        distances = np.sqrt(np.sum((X[:, np.newaxis] - centroids) ** 2, axis=2))

        # Поиск ближайшего центроида для каждой точки
        labels = np.argmin(distances, axis=1)

        # Обновление центроидов
        new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])

        # Проверка на сходимость
        if np.all(centroids == new_centroids):
            break

        centroids = new_centroids

    return labels, centroids

k = 3
labels, centroids = kmeans(X, k)

# Визуализация результатов кластеризации
plt.scatter(opening_prices, closing_prices, c=labels, cmap='viridis', s=1)

# Настройка внешнего вида графика
plt.xlabel('Opening Price')
plt.ylabel('Closing Price')
plt.title('Clustering of Opening and Closing Prices')

# Отображение графика
plt.show()

# Вывод результатов
print("Метки кластеров:", labels)
print("Центроиды:", centroids)
