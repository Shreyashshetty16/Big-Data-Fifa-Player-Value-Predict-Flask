from flask import Flask,request, url_for, redirect, render_template
import pickle
import numpy as np

app = Flask(__name__)

model=pickle.load(open('model.pkl','rb'))


@app.route('/')
def hello_world():
    return render_template("index.html")


@app.route('/predict',methods=['POST','GET'])
def predict():
    int_features=[int(x) for x in request.form.values()]
    final=[np.array(int_features)]
    print(int_features)
    print(final)
    prediction = model.predict(final)
    rounded= [np.round(x) for x in prediction]
    return render_template('index.html', pred='Predicated Player Value is {}'.format(rounded[0]), x="player rating is ")

if __name__ == '__main__':
    app.run(debug=True)