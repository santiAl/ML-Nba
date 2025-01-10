import mysql.connector
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import RidgeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from joblib import dump

# Configuración de conexión
config = {
    'host': 'localhost',         # Ejemplo: 'localhost' o la IP del servidor
    'user': 'root',      # Usuario de la base de datos
    'database': 'nbadb' # Nombre de la base de datos
}

try:
    # Crear conexión
    connection = mysql.connector.connect(**config)
    
    if connection.is_connected():
        print("Conexión exitosa a la base de datos")
        
        # Crear un cursor para realizar consultas
        cursor = connection.cursor()
        
        # Ejemplo de consulta
        cursor.execute("SELECT DATABASE();")
        result = cursor.fetchone()
        print("Base de datos actual:", result)

except mysql.connector.Error as e:
    print(f"Error al conectar a MySQL: {e}")


cursor.callproc('get_table1')
for result in cursor.stored_results():
        df = pd.DataFrame(result.fetchall(), columns=result.column_names)

df = df.iloc[:, ~df.columns.duplicated()]
df = df.drop(columns=['away_steals', 'status','home_steals'])
df['start_time_str'] = pd.to_datetime(df['start_time_str'],  format='%Y-%m-%d')
df['away_field_goal_%'] = df['away_field_goals_made'] / df['away_field_goal_attempts']
df['home_field_goal_%'] = df['home_field_goals_made'] / df['home_field_goal_attempts']
df['away_free_throw_%'] = df['away_free_throws_made'] / df['away_free_throw_attempts']
df['home_free_throw_%'] = df['home_free_throws_made'] / df['home_free_throw_attempts']
df['away_three_pointer_%'] = df['away_three_pointers_made'] / df['away_three_pointer_attempts']
df['home_three_pointer_%'] = df['home_three_pointers_made'] / df['home_three_pointer_attempts']


# Eliminacion datos atipicos.
condition = (df['home_score'] == 0) | (df['away_score'] == 0)
df = df[~condition]
df = df.drop(columns=["home_personal_fouls","away_personal_fouls"])


def test(data,model,predictors,size):
    # Porcentaje de datos para entrenamiento
    train_size = size

    # Calcular el índice de corte entrenamiento
    train_index = int(len(data) * train_size)

    train = data[:train_index]
    test =  data[train_index:]

    model.fit(train[predictors],train['won_x'])

    preds = model.predict(test[predictors])
    preds = pd.Series(preds,index=test.index)

    combined = pd.concat([test['won_x'],preds], axis = 1)
    combined.columns = ["actual","prediction"]   #Renombremos las columnas

    return combined



def team_games(team):      # Hace una fila de cada partido segun el equipo
    filtered_df = df[(df['away_team_name'] == team) | (df['home_team_name'] == team)]
    filtered_df.sort_values(by='start_time_str', ascending=False, inplace=True)
    result = filtered_df.apply(new_df,axis=1,args={team,})
    result_df = pd.DataFrame(result.tolist())

    return result_df


def new_df(row,team): # Cambia nombre de columnas (las ordena)
    if row['away_team_name'] == team:
        columns = df.filter(regex='^away').columns
        new_column_names = [col.replace('away_', '') for col in columns]
    else:
        columns = df.filter(regex='^home').columns
        new_column_names = [col.replace('home_', '') for col in columns]
    
    new_df_r = {}
    for index , col in enumerate(columns):
        new_df_r[new_column_names[index]] = row[col]
    
    if(row["home_team_name"] == team):
        if(row['home_score'] > row['away_score']):
            new_df_r['won'] = 1
        else:
            new_df_r['won'] = 0
    else:
        if(row['home_score'] > row['away_score']):
            new_df_r['won'] = 0
        else:
            new_df_r['won'] = 1

    if(row["away_team_name"] == team):
        new_df_r['oponent_id'] = row['home_team_id']
    else:
        new_df_r['oponent_id'] = row['away_team_id']

    new_df_r['start_time_str'] = row['start_time_str']
    new_df_r['game_id'] = row['game_id']
    new_df_r['home'] = False if row['away_team_name'] == team else True
    return new_df_r
    


def find_team_averages(team):
    shifted_team = team.shift(1)       # Para sacar las estadisticas del equipo antes de jugar ese partido
    rolling = shifted_team.rolling(10).mean()    # Aqui cambiar este numero para encontar las estadisticas de los ultimos n partidos
    return rolling

def find_record(team):
    count = team["won"].rolling(2).sum()
    return count

def find_best_features_rf(data,n=None):
    full_two = data.copy()
    rf_two = RandomForestClassifier(n_estimators=200, random_state=43,max_depth=15,min_samples_leaf=40)
    train_index = int(len(full_two) * 0.7)
    train = full_two[:train_index]
    x_train = train.drop(columns=['won_x','start_time_str',"game_id_x"])
    y_train = train['won_x']
    
    rf_two.fit(x_train, y_train)

    # Obtener las importancias
    importances = rf_two.feature_importances_

    #Seleccionar las n características más importantes
    if n == None:
        n = data.shape[1] 
    
    top_features = np.argsort(importances)[-n:]

    selected_column_names = x_train.columns[top_features].tolist()

    return selected_column_names


df_combined = pd.DataFrame()

for team in df['home_team_name'].unique():

    removed_columns = ['period','team_name']
    current_team = team_games(team)
    selected_columns = current_team.columns[~current_team.columns.isin(removed_columns)]           

    rolling_df = current_team[selected_columns]

    
    rolling_df['cumulative_made'] = rolling_df['field_goals_made'].cumsum() - rolling_df['field_goals_made']
    rolling_df['cumulative_attempted'] = rolling_df['field_goal_attempts'].cumsum() - rolling_df['field_goal_attempts']

    # Calcular el porcentaje acumulado hasta el partido anterior
    rolling_df['cumulative_percentage'] = (rolling_df['cumulative_made'] / rolling_df['cumulative_attempted'])

    
    # Extraer la columnas (que no quiero que se les saque el porcentaje) antes del cálculo
    game_id = rolling_df['game_id']
    home = rolling_df['home']
    oponent = rolling_df['oponent_id']
    won = rolling_df['won']
    start_time = rolling_df['start_time_str']
    percentage = rolling_df['cumulative_percentage']

    # Eliminar columnas temporalmente
    rolling_df = rolling_df.drop(columns=['game_id',"home","oponent_id","start_time_str","won",'cumulative_made','cumulative_attempted','cumulative_percentage'])

    rolling_df = rolling_df.groupby(['team_id'] , group_keys = False).apply(find_team_averages)   # Calculo del promedio de estadisticas de los ultimos n partidos
    
    # Reinsertar columnas después del cálculo
    rolling_df['won'] = won
    rolling_df['game_id'] = game_id
    rolling_df['home'] = home
    rolling_df['oponent_id'] = oponent
    rolling_df['start_time_str'] = start_time
    #print(rolling_df)
    rolling_df['matches_win'] = rolling_df['won'].cumsum()
    rolling_df['matches_win'] = rolling_df.apply(lambda row: row['matches_win'] - 1 if row['won'] else row['matches_win'], axis=1)
    rolling_df['matches_defeat'] = (rolling_df['won'] == 0).cumsum()
    rolling_df['matches_defeat'] = rolling_df.apply(lambda row: row['matches_defeat'] if row['won'] else row['matches_defeat'] - 1, axis=1)
    rolling_df['record'] = rolling_df['matches_win'] - rolling_df['matches_defeat']
    rolling_df['cumulative_percentage'] = percentage
    # Para las estadisticas del otro equipo podria sacar estas estadisticas para todos los equipos y luego hacer un merge.

    df_combined = pd.concat([df_combined, rolling_df], ignore_index=True)


#Para tener las estadisticas de ambos equipos en la misma fila.
full = df_combined.merge(df_combined, left_on=["team_id","start_time_str"], right_on=["oponent_id","start_time_str"])

#print(full[(full['team_id_x']==42)][['true_count_last_10_x','true_count_last_10_y']])

full = full.sort_values(by="start_time_str")

# Aca arriba saque varias columnas que estaban correlacionadas entre si y empreoraban el rendimiento de rf
full = full.drop(columns=["oponent_id_y","game_id_y","home_y","won_y","team_id_y","free_throw_attempts_x","free_throw_attempts_y","three_pointer_attempts_x","three_pointer_attempts_y","field_goals_made_x","field_goals_made_y",'score_x','score_y','matches_defeat_y','matches_defeat_x','matches_win_x','matches_win_y'])   # Columnas que no me interesan



removed_columns = list(full.columns[full.dtypes == 'object']) + ['won_x','team_id_x','oponent_id_x','start_time_str','game_id_x']
selected_columns = full.columns[~full.columns.isin(removed_columns)]

#-------------------------------- Definicion de los diferentes modelos -------------------------------------

xgb_model = XGBClassifier(
    n_estimators=100,  # Número de árboles
    learning_rate=0.1,  # Tasa de aprendizaje
    max_depth=3,       # Profundidad máxima de los árboles
    random_state=42
)
rf = RandomForestClassifier(n_estimators=200, random_state=43,max_depth=10,min_samples_leaf=40)
rr = RidgeClassifier(alpha=1)
split = TimeSeriesSplit(n_splits=3)

sfs = SequentialFeatureSelector(rr, n_features_to_select = 10, direction="backward", cv=split)   # Slecciona las mejores features (columnas)


#-------------------------------- Definicion de los diferentes modelos -----------------------------------


# Ridge funciona mejor estandatizando.
scaler = MinMaxScaler()
full[selected_columns] = scaler.fit_transform(full[selected_columns])   # Escalado de los datos menos los que estan en removed_columns
full = full.dropna()
sfs.fit(full[selected_columns],full["won_x"])

# Obtener los predictores seleccionados para rr
predictors = list(selected_columns[sfs.get_support()])

# Buscar los mejores predictores para random forest
predictors_rf = find_best_features_rf(full,5)

predictions = test(full,rr,predictors,0.7)

print(predictors)
acc = accuracy_score(predictions['actual'],predictions['prediction'])
print(acc)

#dump(rr, 'rrNbaModel.joblib')

# Para hacer graficos de dispersion.
"""
#df_subset = full.iloc[:, 15:22]

df_subset = full[["won_x","record_x","record_y"]]

# Comprobar si 'target' está entre las primeras 10 columnas
if 'won_x' not in df_subset.columns:
    # Si 'target' no está, asegurarse de que se agregue
    df_subset = full.iloc[:, 16:22]  # Selecciona las primeras 9 columnas
    df_subset['won_x'] = full['won_x']  # Añadir la columna 'target'

sns.pairplot(df_subset, hue='won_x', diag_kind='kde')
plt.show()
"""
# Para hacer una matriz de correlación
"""
correlation_matrix = full.corr()
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Matriz de Correlación")
plt.show()
"""

# Cerrar la conexión
if connection.is_connected():
    cursor.close()
    connection.close()
    print("Conexión cerrada")
