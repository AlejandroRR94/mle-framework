from functions.model_utils import *
from functions.prepare_data_utils import *
from functions.test_utils import *

from etl_pipeline import *

import pandas as pd
import numpy as np
from prefect import task, flow
import mlflow
import xgboost as xgb
import lightgbm as lgb

import os
import pathlib
import sys
import joblib
import datetime
import argparse


class real_time_inference():

    """
    Cargamos los datos extraído de la API
    """

    def __init__(self, n_rows:int=100):
        self.n_rows = n_rows


    def data_preparation(self):
    # Extraemos los datos de la url
        
        if len(sys.argv) > 1:
            self.n_rows = sys.argv[1]
        else:
            self.n_rows = self.n_rows

        print(f"Realizando inferencia sobre {self.n_rows} registros")
        extraction = extract_data.fn(
            url= 'https://api.energidataservice.dk/dataset/CO2Emis?limit=5',
            n_rows = self.n_rows
        )
    
        # Realizamos las transformaciones
        transformation_0 = rename_columns.fn(df = extraction)
        transformation_1 = cast_columns.fn(transformation_0)
        df = encode_area_columns.fn(transformation_1)

        dp = DataPreparator() # Instanciamso DataPreparator

        # Obtenemos los datos transformadose
        df_1, _ = dp.get_data(data=df)

        return df_1

    def load_model(self):

        logged_model = 'runs:/a7f7d8c905944d45a076e477153cfd53/MODELOS/modelo_2023-07-19'

        # Load model as a PyFuncModel.
        loaded_model = mlflow.pyfunc.load_model(logged_model)

        return loaded_model
    
    def inference(self,model):
        
        df_1 = self.data_preparation()
        
        if model == None: # Si no disponemos de modelo en MLFlow, cargamos el último entrenado
            model =  joblib.load(get_last_model("models", "pkl"))

        X, y = df_1.drop(["co2_emissions", "area"], axis = 1), df_1.co2_emissions

        # Guardamos los archivos que simulan el escenario de real time para guardar histórico
        date = datetime.datetime.now()
        joblib.dump(X, f"data/real_time_data/attributes/realtime_attributes_{date}.pkl")
        joblib.dump(y, f"data/real_time_data/targets/realtime_targets_{date}.pkl")

        predictions = model.predict(X)
        joblib.dump(predictions, f"data/real_time_data/predictions/predictions_ {date}.pkl")     

        # Instanciamos la clase ModelTrainer para hacer uso de sus métodos
        mt = ModelTrainer(df=df_1, target="co2_emissions")

        # Creamos diccionario de métricas
        # metrics = mt.create_metrics_dict(model=model, y_true = y, y_pred = predictions)

        # Actualizamos el json de métricas
        mt.save_inference_metrics(filename= "logs/inference_logs.json",
                                    model=model, y_true = y, y_pred = predictions
                                    )
        
if __name__ == "__main__":


    # Crea el analizador de argumentos
    parser = argparse.ArgumentParser(description='Extracción de pocos registros de la API')

    # Agrega argumentos
    parser.add_argument('n_rows', help='Número de registros a extraer de la API')

    # Analiza los argumentos de la línea de comandos
    args = parser.parse_args()

    rti = real_time_inference(n_rows=args.n_rows)
    
    mlflow_model = rti.load_model()

    rti.inference(model=mlflow_model)