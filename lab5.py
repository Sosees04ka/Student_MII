import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Загрузка данных
dataset = pd.read_excel('Yamana_Gold.xlsx')
n = int(len(dataset) * 0.99)  # 99% информации
x = dataset['Low'].values[:n]
y = dataset['Close'].values[:n]


# Реализация линейной регрессии
def linear_regression(x, y):
    # Вычисление суммы x, y, x^2, xy
    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_x2 = np.sum(x ** 2)
    sum_xy = np.sum(x * y)

    # Вычисление коэффициентов регрессии
    n = len(x)
    #Коэффициент наклона
    slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
    # Коэффициент смещения
    intercept = (sum_y - slope * sum_x) / n

    return slope, intercept

slope, intercept = linear_regression(x, y)

# Предсказание y для всех значений x
y_pred = slope * x + intercept

# Рисование графика
plt.scatter(x, y, color='blue', label='Реальные значения', s=1)
plt.plot(x, y_pred, color='red', label='Предсказанные значения')
plt.xlabel('Цена открытия')
plt.ylabel('Цена закрытия')
plt.title('')

# Вывод коэффициентов регрессии на графике
plt.text(0, 23, f"Коэффициент наклона: {slope}", fontsize=10)
plt.text(0, 22, f"Коэффициент смещения: {intercept}", fontsize=10)

plt.legend()
plt.show()
