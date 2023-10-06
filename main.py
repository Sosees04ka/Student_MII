import numpy as np
from flask import Flask, render_template, request
import pandas as pd

app = Flask(__name__)

dataset = pd.read_excel('Yamana_Gold.xlsx')

@app.route("/")
def home():
    return render_template('home.html')

@app.route("/info")
def info():
    return render_template('info.html')

@app.route("/table", methods=['GET'])
def table():
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

    setting_data = request.args
    sliced_data = dataset.iloc[int(setting_data['x_start']):int(setting_data['x_end']), int(setting_data['y_start']):int(setting_data['y_end'])]
    sliced_data_description = pd.DataFrame(np.column_stack((names, descriptions))).iloc[
                              int(setting_data['y_start']):int(setting_data['y_end'])]
    return render_template('table.html', data=sliced_data.to_html(classes='table'), data_type = pd.DataFrame(sliced_data.dtypes).to_html(classes='table', header=False),
                           data_count = pd.DataFrame(sliced_data.isnull().sum()).to_html(classes='table', header=False),
                           data_count_false = pd.DataFrame(sliced_data.isnull().apply(lambda x: x.value_counts().get(False, 0))).to_html(classes='table', header=False),
                           sliced_data_description = sliced_data_description.to_html(classes='table', header=False, index=False))

if __name__ == "__main__":
    app.run(debug=False)