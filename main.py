import matplotlib

from bloomFilter import BloomFilter
from bloomFilterAnn import BloomFilterAnn
from bloomFilterIrina import BloomFilterIrina

matplotlib.use('Agg')  # Используем Agg бэкенд вместо Tkinter
from matplotlib import pyplot as plt
import numpy as np
from flask import Flask, render_template, request, redirect
import pandas as pd
import random

import math
from bitarray import bitarray

app = Flask(__name__)

dataset = pd.read_excel('Yamana_Gold.xlsx')

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

ArrKeyWordAnn = ["salary", "company", "employee" ]

ArrKeyWordIrina = ["vegetables", "fruits"]

bloom_filter = BloomFilter(1000, 100)
bloom_filterAnn = BloomFilterAnn(1000, 100)
bloom_filterIrina = BloomFilterIrina(1000, 100)

for i in range(len(names)):
    bloom_filter.add_to_filter(names[i])

for i in range(len(descriptions)):
    bloom_filter.add_to_filter(descriptions[i])
bloom_filter.add_to_filter("Yamana Gold")

for i in range(len(ArrKeyWordAnn)):
    bloom_filterAnn.add_to_filter(ArrKeyWordAnn[i])

for i in range(len(ArrKeyWordIrina)):
    bloom_filterIrina.add_to_filter(ArrKeyWordIrina[i])

def expand_dataset():
    dataset = pd.read_excel('Yamana_Gold.xlsx')
    random_index = random.randint(0, len(dataset) - 1)
    num_new_rows = int(len(dataset) * 0.1)
    dataset_2 = dataset.iloc[random_index: random_index + num_new_rows].copy()

    for column in dataset_2.select_dtypes(include=[np.number]):
        mean_value = dataset_2[column].mean()
        std_dev = dataset_2[column].std()
        random_value = np.random.normal(mean_value, std_dev)
        dataset_2[column] += random_value

    expanded_dataset = pd.concat([dataset, dataset_2], ignore_index=True)
    return expanded_dataset

@app.route("/")
def home():
    return render_template('home.html')

@app.route("/info")
def info():
    return render_template('info.html')

@app.route("/filter", methods=['GET'])
def filter():
    setting_data = request.args
    if not bloom_filter.check_is_not_in_filter(setting_data['theme']):
        return redirect('/')
    if not bloom_filterAnn.check_is_not_in_filter(setting_data['theme']):
        return redirect('http://127.0.0.1:5001/bloom_filter?keyWord=salary')
    if not bloom_filterIrina.check_is_not_in_filter(setting_data['theme']):
        return redirect('http://127.0.0.1:5002')
    else:
        return render_template('notFoundTheme.html')

@app.route("/table", methods=['GET'])
def table():
    setting_data = request.args

    if  setting_data['checkset'] == 'Да':
        dataset = expand_dataset()
    else:
        dataset = pd.read_excel('Yamana_Gold.xlsx')

    print(pd.read_excel('Yamana_Gold.xlsx'))
    print(expand_dataset())

    if setting_data['checkgraph'] == 'Да':
        plt.plot(dataset['Date'], dataset['Open'], color='orange', label='Самая низкая цена торгов')
        plt.plot(dataset['Date'], dataset['High'], color='blue', label='Самая высокая цена торгов')
        plt.xlabel('Год')
        plt.ylabel('Цена')
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2)
        plt.savefig('static/images/OpenHigh.jpg', format='jpeg')
        plt.clf()

        plt.plot(dataset['Date'], dataset['Low'], color='orange', label='Цена открытия')
        plt.plot(dataset['Date'], dataset['Close'], color='blue', label='Цена закрытия')
        plt.xlabel('Год')
        plt.ylabel('Цена')
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2)
        plt.savefig('static/images/LowClose.jpg', format='jpeg')
        plt.clf()

        plt.plot(dataset['Date'], dataset['Adj Close'], color='blue', label='«Отрегулированная» цена закрытия')
        plt.xlabel('Год')
        plt.ylabel('Цена')
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2)
        plt.savefig('static/images/AdjClose.jpg', format='jpeg')
        plt.clf()

        plt.plot(dataset['Date'], dataset['Volume'], color='blue',
                 label='Количество акций, с которыми совершались сделки в торговый день')
        plt.xlabel('Год')
        plt.ylabel('Акций')
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2)
        plt.savefig('static/images/Volume.jpg', format='jpeg')
        plt.clf()

    sliced_data = dataset.iloc[int(setting_data['x_start']):int(setting_data['x_end']), int(setting_data['y_start']):int(setting_data['y_end'])]
    sliced_data_description = pd.DataFrame(np.column_stack((names, descriptions))).iloc[
                              int(setting_data['y_start']):int(setting_data['y_end'])]

    # Задание 1
    summer_data = dataset[(dataset['Date'].dt.month >= 6) & (dataset['Date'].dt.month <= 8)]
    price_stats = summer_data['Open'].agg(['min', 'max', 'mean']).rename(
        index={'min': 'Минимальная цена открытия', 'max': 'Максимальная цена открытия',
               'mean': 'Средняя цена открытия'})

    # Задание 2
    winter_data = dataset[(dataset['Date'].dt.month >= 1) & (dataset['Date'].dt.month <= 3)]
    winter_price_stats = winter_data.groupby(winter_data['Date'].dt.year, as_index=False)['Open'].agg(
        {'min', 'mean', 'max'}).rename(columns={'min': 'Минимальная цена открытия', 'mean': 'Средняя цена открытия',
                                                'max': 'Максимальная цена открытия'})

    # Задание 4
    year_price_stats = dataset.groupby(dataset['Date'].dt.year, as_index=False)['Open'].agg(
        {'min', 'mean', 'max'}).rename(columns={'min': 'Минимальная цена открытия', 'mean': 'Средняя цена открытия',
                                                'max': 'Максимальная цена открытия'})

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
                           year_touch_stats = pd.DataFrame(year_touch_stats).to_html(classes='table', index=False),
                           setting_data = setting_data)

if __name__ == "__main__":
    app.run(debug=False)