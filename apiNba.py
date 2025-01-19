from flask import Flask, request, jsonify , render_template
from joblib import load
import pandas as pd
import json


app = Flask(__name__)

# Predictoras:    ['field_goal_%_x', 'blocks_x', 'blocks_y', 'field_goal_%_y', 'recieved_field_goal_%_x', 'turnovers_y', 'recieved_points_x', 'score_x', 'recieved_field_goal_%_y', 'turnovers_x', 'recieved_points_y', 'score_y', 'record_y', 'home_x', 'record_x']

# Cargar el modelo
model_rf = load('rfNbaModel.joblib')
model_mlp = load('mlpNbaModel.joblib')

# Ruta para servir el HTML
@app.route('/')
def home():
    return render_template('index.html')

# Ruta para realizar predicciones
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['data']
    model_selected = request.json['model']
    data_df = pd.DataFrame(data)  # Convertir a DataFrame
    if(model_selected == 0):
        predicciones = model_rf.predict(data_df).tolist()
    else:
        predicciones = model_mlp.predict(data_df).tolist()
    return jsonify(predicciones)

if __name__ == '__main__':
    app.run()
