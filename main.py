from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
from sklearn.svm import SVR

app = Flask(__name__)

df_KZ = pd.read_csv('eco-print-KZ.csv')
df_KZ = df_KZ[['Year', 'Built-up Land', 'Carbon', 'Cropland', 'Fishing Grounds', 'Forest Products', 'Grazing Land', 'Total']]

years = df_KZ['Year']
parameters = df_KZ.columns[1:]

predictions = {}
preset = []

for parameter in parameters:
    X_train = years.values.reshape(-1, 1)
    y_train = df_KZ[parameter].values
    
    svr_rbf = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
    
    svr_rbf.fit(X_train, y_train)
    
    predicted_value = svr_rbf.predict([[2023]])[0]
    
    predictions[parameter] = predicted_value
    preset.append(round(predicted_value, 2))

@app.route("/get-data/<int:year>")
def get_data(year):
  print(year)
  if year < 1992 or year > 2030:
      return jsonify({"error": "Недопустимый год"})
  
  predictions = {}
  for parameter in parameters:
      X_train = years.values.reshape(-1, 1)
      y_train = df_KZ[parameter].values

      svr_rbf = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)

      svr_rbf.fit(X_train, y_train)

      predicted_value = svr_rbf.predict([[year]])[0]

      predictions[parameter] = round(predicted_value, 2)
  return jsonify(predictions)

@app.route('/')
def index():
  

  buildup = preset[0]
  carbon = preset[1]
  cropland = preset[2]
  fishing = preset[3]
  forest = preset[4]
  grazing = preset[5]
  total = preset[6]
  
  
  return render_template('index.html', buildup=buildup, carbon=carbon, cropland=cropland, fishing=fishing, forest=forest, grazing=grazing, total=total)

@app.route('/acc')
def acc():
  return render_template('acc.html')

@app.route('/popup')
def popup():
  return render_template('popup.html')




app.run(host='0.0.0.0', port=81)
