from flask import Flask, render_template
import pandas as pd

app = Flask(__name__)

dataset = pd.read_excel('Yamana_Gold.xlsx')

@app.route("/")
def home():
    return render_template('index.html', data=dataset.to_html(classes='mystyle'))

if __name__ == "__main__":
    app.run(debug=False)