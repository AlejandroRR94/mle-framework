# Librerías de utilidad general
import pandas as pd
from typing import Tuple
import os
from datetime import datetime, timedelta

# Librerías de visualización
import matplotlib.pyplot as plt
import seaborn as sns

# Álgebra lineal
import numpy as np

# Machine Learning
from sktime.forecasting.model_selection import temporal_train_test_split
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
import xgboost as xgb


# Librería para la creación de DB
import sqlite3


class DataPreparator():
    """
    Clase que encapsula todas las funciones de preparación de datos para
    el modelo de forecasting

    Uso:
        1. Construye la clase DataPreparator()
        2. Utiliza el método get_data()

    """


    def __init__(self, path:str="data/clean/"):
         
         self.path = path

         
    def get_last_modified_file(self, alternative_path:str=None):
        """
        Ordena los archivos según su fecha de modificación en un diccionario
        para obtener el último
        
        Retorna: directorio del archivo modificado en último lugar
        """

        files = [os.path.join(self.path, f) for f in os.listdir(self.path)]
        # file_dict = {os.path.getmtime(f):f for f in files}
        file_dict = {os.path.getmtime(f)+i:f for f,i in zip(files, range(len(files)))}
        
        return file_dict[min(file_dict.keys())]



    
    def load_data(self)->pd.DataFrame:
        """
        Por defecto, carga el archivo más reciente en el directorio.

        Args:
            - path: string con el directorio de la carpeta que contiene los archivos
                    o el directorio del propio archivo
            - newest: booleano que indica si se desea el archivo más receinte.

        Returns:
            - DataFrame con los datos para entrenar
        """
        if os.path.isfile(self.path):
            suffix = self.path.split(".")[-1]
            print(suffix)
            if suffix == "csv":
                self.df = pd.read_csv(self.path)
            else:
                self.df = pd.read_excel(self.path)

        else:
            newest_file = self.get_last_modified_file()
            suffix = newest_file.split(".")[-1]
            
            if suffix == "csv":
                self.df = pd.read_csv(newest_file)
            else:
                self.df = pd.read_excel(newest_file)
        
        return self.df


    
    def separate_by_area(self, data:pd.DataFrame)->Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Separa los datos según el area de consumo
        """

        df_1 = data[data.area==1].set_index("datetime_utc").sort_index()
        df_2 = data[data.area==2].set_index("datetime_utc").sort_index()

        return df_1, df_2


    
    def split_data(self,
        data: pd.DataFrame, target: str = "co2_emissions", fh: int = 24) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Estructura los datos para ser pasados al modelo:
        - Especifica el índice como lo exige sktime.
        - Prepara variables exógenas.
        - Prepara las series temporales para ser pronosticadas.
        - Divide los datos entre conjuntos de entrenamiento y evaluación.
        """

        # Especifica el índice como lo espera sktime.
        data["datetime_utc"] = pd.PeriodIndex(data["datetime_utc"], freq="m")
        data = data.set_index(["area", "datetime_utc"]).sort_index()

        # Preparar variables exógenas
        X = data.drop(columns=[target])
        # Preparar serie temporal a ser pronosticada
        y = data[[target]]

        y_train, y_test, X_train, X_test = temporal_train_test_split(y, X, test_size=fh)

        return y_train, y_test, X_train, X_test



    
    def create_features(self, df:pd.DataFrame) -> pd.DataFrame:
        """
        Crea atributos de series temporales basándose en el índice en formato datetime.

        Retorna:
            - pd.DataFrame con los campos añadidos
        """
        df = df.copy()
        df.set_index(pd.to_datetime(df.index), inplace = True)
        df['hour'] = df.index.hour
        df["minute"] = df.index.minute
        # df["second"] = df.index.second
        df['dayofweek'] = df.index.dayofweek
        # df['quarter'] = df.index.quarter
        # df['month'] = df.index.month
        # df['year'] = df.index.year
        # df['dayofyear'] = df.index.dayofyear
        # df['dayofmonth'] = df.index.day
        # df['weekofyear'] = df.index.isocalendar().week
        
        return df

    
    def add_lags(self, df:pd.DataFrame) -> pd.DataFrame:
        """
        Añade variables para saber cuál era el valor de la serie temporal 5, 10 y 15 minutos
        previos al registro correspondiente.
        """
        target_map = df['co2_emissions'].to_dict()
        df['lag1'] = (df.index - pd.Timedelta('5 minutes')).map(target_map)
        df['lag2'] = (df.index - pd.Timedelta('10 minutes')).map(target_map)
        df['lag3'] = (df.index - pd.Timedelta('15 minutes')).map(target_map)
        return df
    


    def get_data(self)->Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Pipeline para obtener los datos y prepararlos
        """
        
        data = self.load_data()

        df_1, df_2 = self.separate_by_area(data)

        df_1 = self.create_features(df_1)
        df_1 = self.add_lags(df_1)

        df_2 = self.create_features(df_2)
        df_2 = self.add_lags(df_2)

        return df_1, df_2



     