import numpy as np
from flask import Flask, render_template, request
import pandas as pd

app = Flask(__name__)

dataset = pd.read_excel('Yamana_Gold.xlsx')

#Задание 1
summer_data = dataset[(dataset['Date'].dt.month >= 6) & (dataset['Date'].dt.month <= 8)]
price_stats = summer_data['Open'].agg(['min', 'max', 'mean']).rename(index={'min': 'Минимальная цена открытия', 'max': 'Максимальная цена открытия', 'mean': 'Средняя цена открытия'})

#Задание 2
winter_data = dataset[(dataset['Date'].dt.month >= 1) & (dataset['Date'].dt.month <= 3)]
winter_price_stats = winter_data.groupby(winter_data['Date'].dt.year, as_index=False)['Open'].agg({'min', 'mean', 'max'}).rename(columns={'min': 'Минимальная цена открытия', 'mean': 'Средняя цена открытия', 'max': 'Максимальная цена открытия'})

#Задание 4
year_price_stats = dataset.groupby(dataset['Date'].dt.year, as_index=False)['Open'].agg({'min', 'mean', 'max'}).rename(columns={'min': 'Минимальная цена открытия', 'mean': 'Средняя цена открытия', 'max': 'Максимальная цена открытия'})


names = np.array([
    'Date',
    'Open',
    'High',
    'Low',
    'Close',
    'Adj Close',
    'Volume'
])
descriptions = np.array([
    'Торговый день',
    'Самая высокая цена торгов',
    'Самая низкая цена торгов',
    'Цена открытия',
    'Цена закрытия',
    '«Отрегулированная» цена закрытия',
    'Количество акций, с которыми совершались сделки в торговый день'
])

@app.route("/")
def home():
    return render_template('home.html')

@app.route("/info")
def info():
    return render_template('info.html')

@app.route("/table", methods=['GET'])
def table():
    setting_data = request.args
    sliced_data = dataset.iloc[int(setting_data['x_start']):int(setting_data['x_end']), int(setting_data['y_start']):int(setting_data['y_end'])]
    sliced_data_description = pd.DataFrame(np.column_stack((names, descriptions))).iloc[
                              int(setting_data['y_start']):int(setting_data['y_end'])]

    # Задание 3
    year_touch_data = dataset[(dataset['Date'].dt.year == int(setting_data['year']))]
    year_touch_stats = year_touch_data.groupby(year_touch_data['Date'].dt.year, as_index=False)['Open'].agg(
        {'min', 'mean', 'max'}).rename(columns={'min': 'Минимальная цена открытия', 'mean': 'Средняя цена открытия',
                                                'max': 'Максимальная цена открытия'})

    return render_template('table.html', data=sliced_data.to_html(classes='table'), data_type = pd.DataFrame(sliced_data.dtypes).to_html(classes='table', header=False),
                           data_count = pd.DataFrame(sliced_data.isnull().sum()).to_html(classes='table', header=False),
                           data_count_false = pd.DataFrame(sliced_data.isnull().apply(lambda x: x.value_counts().get(False, 0))).to_html(classes='table', header=False),
                           sliced_data_description = sliced_data_description.to_html(classes='table', header=False, index=False),
                           summer_prices = pd.DataFrame(price_stats).to_html(classes='table', header=False),
                           winter_price_stats = pd.DataFrame(winter_price_stats).to_html(classes='table', index=False),
                           year_price_stats = pd.DataFrame(year_price_stats).to_html(classes='table', index=False),
                           year_touch_stats = pd.DataFrame(year_touch_stats).to_html(classes='table', index=False))

if __name__ == "__main__":
    app.run(debug=False)