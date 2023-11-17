import pandas as pd
from sklearn.tree import DecisionTreeRegressor, plot_tree
import matplotlib.pyplot as plt

# Загрузка данных
dataset = pd.read_excel('Yamana_Gold.xlsx')
X = dataset['Date'].dt.month
y = dataset['Low']

# Создание и обучение модели дерева решений
dectree = DecisionTreeRegressor(max_depth=2)
dectree.fit(X.values.reshape(-1, 1), y)

# Визуализация дерева решений
plt.figure(figsize=(12, 6))
# Переименование атрибутов
plot_tree(dectree, feature_names=['Месяц'], filled=True,
          class_names=['Low'], rounded=True, precision=5)
plt.show()
