from flask import Flask, redirect, url_for,request
from num2words import num2words
app=Flask(__name__)
@app.route("/")
def home():
    return "<html><form Action='http://127.0.0.1:5000/numtext'Method=get>" \
    "<input type=text size=20 name=name>" \
    "<input type=submit value='Кнопка'>" \
    "</form></html>"
@app.route("/numtext",methods=['GET','POST'])
def numtext():
    data = request.args
    return num2words(data['name'],lang='ru')
if __name__=="__main__":
    app.run(debug=False)