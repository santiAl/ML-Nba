from flask import Flask, request, jsonify , render_template
from joblib import load
import pandas as pd
import json


app = Flask(__name__)

# Predictoras:    ['assists_x', 'blocks_x', 'offensive_rebounds_x', 'three_pointers_made_x', 'turnovers_x', 'field_goal_%_x', 'home_x', 'blocks_y', 'turnovers_y', 'record_y']

# Cargar el modelo
model = load('rrNbaModel.joblib')


data = {
    'assists_x': [20],
    'blocks_x': [4],
    'offensive_rebounds_x': [8],
    'three_pointers_made_x': [13],
    'turnovers_x': [3],
    'field_goal_%_x': [0.25],
    'home_x': [1],
    'blocks_y': [3],
    'turnovers_y': [2],
    'record_y': [7]
}

# Crear un DataFrame
input_data = pd.DataFrame(data)

result = model.predict(input_data)
print(result)




# Ruta para servir el HTML
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['data']
    print(data)
    data_df = pd.DataFrame(data)  # Convertir a DataFrame  
    predicciones = model.predict(data_df).tolist()
    print(predicciones)
    return jsonify(predicciones)

if __name__ == '__main__':
    app.run()
