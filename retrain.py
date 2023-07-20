"""
1. Procesar los datos de la carpeta "data/real_time_data/"
2. Juntarlos en un solo dataset
3. Unirlos a los últimos datos de la carpeta "data/clean/"
4. Reentrenar

"""

import os
import joblib
import pickle

import pandas as pd
import mlflow

from functions.prepare_data_utils import DataPreparator
from functions.test_utils import get_last_model
from functions.model_utils import * 


# GENERAMOS ÚNICO DATAFRAME A PARTIR DE TODOS LOS REGISTROS QUE SE HAN UTILIZADO PARA HACER LA INFERENCIA EN TIEMPO REAL
att_path = "data/real_time_data/attributes"
target_path = "data/real_time_data/targets"
attributes_df = pd.DataFrame()
target_df = pd.DataFrame()

# Concatenamos los atributos
for file in os.listdir(att_path):        
    f = joblib.load(os.path.join(att_path, file))
    attributes_df = pd.concat([attributes_df,f])
    # Cambiamos los nombres de las columnas
    attributes_df.rename(columns={o:n for o,n in zip(list(attributes_df.columns),attributes_df.iloc[0].values)})

# Concatenamos los targets
for file in os.listdir(target_path):        
    f = joblib.load(os.path.join(target_path, file))
    target_df = pd.concat([target_df,f])
    # Cambiamos los nombres de las columnas
    target_df.rename(columns={o:n for o,n in zip(list(attributes_df.columns),attributes_df.iloc[0].values)})

df = pd.merge(attributes_df, target_df, left_index =True, right_index = True).rename(columns = {0:"target"}).drop_duplicates()


# Extraemos el último dataframe limpio
dp = DataPreparator()
df_last_clean = dp.get_last_modified_file()
# print(df_last_clean)

last_model = dp.get_last_modified_file(dp.get_last_modified_file("MODELOS"))
print(f"Last Model: {last_model}")


with mlflow.start_run():

    mt = ModelTrainer(df_last_clean, target="target")
    model = mt.load_model(last_model)

    retrained_model = mt.retrain_model(model,X = attributes_df, y = target_df)
    model_name, model_path, artifact_path, timestamp_today, experiment_name = mt.set_directories()
    mt.save_model(retrained_model, filename=)