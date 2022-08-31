import sklearn
import numpy as np
import pickle
filename="model.pkl"
model=pickle.load(open(filename, 'rb'))
from flask import Flask,render_template,request
app = Flask(__name__,template_folder='templates',static_folder='static')
@app.route('/')
def home():
    return render_template('app.html')
@app.route('/predict_Weather', methods=['POST','GET'])
def predict_Weather():
    a = np.zeros(7)
    a[0] = float(request.form['precipitation'])
    a[1] = float(request.form['temp_max'])
    a[2] = float(request.form['temp_min'])
    a[3] = float(request.form['wind'])
    date=request.form['date']
    a[4]=int(date.split('-')[0])
    a[5]=int(date.split('-')[1])
    a[6]=int(date.split('-')[2])
    print(date)
    a=a.reshape(1,7) 
    result=model.predict(a)[0]
    return render_template("app.html",result=result)
if __name__ == "__main__":
    print("Starting Python Flask Server For Heart Disease Prediction...")
    app.run(debug=True)