import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree, DecisionTreeClassifier


class DecisionTreeNode:
    def __init__(self, feature_index=None, threshold=None, value=None, left=None, right=None):
        self.feature_index = feature_index  # Индекс признака, по которому происходит разделение
        self.threshold = threshold  # Значение для разделения
        self.value = value  # Прогнозируемое значение в листе (среднее значение y)
        self.left = left  # Левое поддерево
        self.right = right  # Правое поддерево


class DecisionTreeRegressor:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.root = None

    def _mse(self, y):
        return np.mean((y - np.mean(y)) ** 2)

    def _split_criterion(self, X, y, feature_index, threshold):
        left_mask = X[:, feature_index] <= threshold
        right_mask = X[:, feature_index] > threshold
        y_left = y[left_mask]
        y_right = y[right_mask]
        return self._mse(y_left) + self._mse(y_right)

    def _build_tree(self, X, y, depth):
        n_samples, n_features = X.shape
        if depth == self.max_depth or len(np.unique(y)) == 1:
            return DecisionTreeNode(value=np.mean(y))
        best_criterion = np.inf
        best_feature_index = None
        best_threshold = None
        for feature_index in range(n_features):
            unique_values = np.unique(X[:, feature_index])
            for threshold in unique_values:
                criterion = self._split_criterion(X, y, feature_index, threshold)
                if criterion < best_criterion:
                    best_criterion = criterion
                    best_feature_index = feature_index
                    best_threshold = threshold

        if best_feature_index is None:
            return DecisionTreeNode(value=np.mean(y))
        left_mask = X[:, best_feature_index] <= best_threshold
        right_mask = X[:, best_feature_index] > best_threshold
        X_left, y_left = X[left_mask], y[left_mask]
        X_right, y_right = X[right_mask], y[right_mask]
        left = self._build_tree(X_left, y_left, depth + 1)
        right = self._build_tree(X_right, y_right, depth + 1)

        return DecisionTreeNode(feature_index=best_feature_index, threshold=best_threshold, left=left, right=right)

    def fit(self, X, y):
        self.root = self._build_tree(X, y, depth=0)

    def predict(self, X):
        return np.array([self._predict_sample(x) for x in X])

    def _predict_sample(self, x, node=None):
        if node is None:
            node = self.root
        if node.value is not None:
            return node.value
        if x[node.feature_index] <= node.threshold:
            return self._predict_sample(x, node.left)
        else:
            return self._predict_sample(x, node.right)


dataset = pd.read_excel('Yamana_Gold.xlsx')
X = dataset['Date'].dt.month.values.reshape(-1, 1)
y = dataset['Low'].values
dectree = DecisionTreeRegressor()
dectree.fit(X.reshape(-1, 1), y)

plt.figure(figsize=(12, 6))
# Переименование атрибутов
plot_tree(dectree, feature_names=['Месяц'], filled=True, class_names=['high'], rounded=True, precision=5)
plt.show()