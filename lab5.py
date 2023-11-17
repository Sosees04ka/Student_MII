import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Загрузка данных
dataset = pd.read_excel('Yamana_Gold.xlsx')
n = int(len(dataset) * 0.99)  # 99% информации
x = dataset['Low'].values[:n].reshape(-1, 1)
y = dataset['Close'].values[:n]

# Создание модели линейной регрессии
regression = LinearRegression()

# Обучение модели
regression.fit(x, y)

# Предсказание y для всех значений x
y_pred = regression.predict(x)

# Рисование графика
plt.scatter(x, y, color='blue', label='Реальные значения', s=1)
plt.plot(x, y_pred, color='red', label='Предсказанные значения')
plt.xlabel('Цена открытия')
plt.ylabel('Цена закрытия')
plt.title('')

# Вывод коэффициентов регрессии на графике
plt.text(0, 23, f"Коэффициент наклона: {regression.coef_[0]}", fontsize=10)
plt.text(0, 22, f"Коэффициент смещения: {regression.intercept_}", fontsize=10)

plt.legend()
plt.show()
