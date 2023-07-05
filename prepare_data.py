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

# Librearía Orquestación
from prefect import task, flow


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

         
    def get_last_modified_file(self):
        """
        Ordena los archivos según su fecha de modificación en un diccionario
        para obtener el último
        
        Retorna: directorio del archivo modificado en último lugar
        """
        files = [os.path.join(self.path, f) for f in os.listdir(self.path)]
        # file_dict = {os.path.getmtime(f):f for f in files}
        file_dict = {os.path.getmtime(f)+i:f for f,i in zip(files, range(len(files)))}
        
        return file_dict[min(file_dict.keys())]



    # @task
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
                df = pd.read_csv(self.path)
            else:
                df = pd.read_excel(self.path)

        else:
            newest_file = self.get_last_modified_file()
            suffix = newest_file.split(".")[-1]
            
            if suffix == "csv":
                df = pd.read_csv(newest_file)
            else:
                df = pd.read_excel(newest_file)
        
        return df


    # @task
    def separate_by_area(self, data:pd.DataFrame)->Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Separa los datos según el area de consumo
        """

        df_1 = data[data.area==1].set_index("datetime_utc").sort_index()
        df_2 = data[data.area==2].set_index("datetime_utc").sort_index()

        return df_1, df_2


    # @task
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



    # @task
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

    # @task
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
    

# @task(name = "Pipeline de carga y preparación de datos")
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


 
class ModelTrainer():
    def __init__(self, df:pd.DataFrame, target:str):
        self.df = df # Dataframe para utilizar
        self.target = target # Nombre de la columna objetivo

    # @task
    def train_k_fold(self, df:pd.DataFrame, k:int=4, test_size:int = 100, gap:int=0)->Tuple[list, list]:

            """
            Entrena un modelo XGBoost con K-Fold Validation
            
            Args:
                    - df: pd.DataFrame
                    - k: número de folds
                    - test_size: Número de registros reservados para evaluar el modelo


            Retorna:
                    - Tupla con las predicciones y las métricas
            """
            
            tss = TimeSeriesSplit(n_splits=k, test_size=test_size, gap=gap)
            df = self.df.sort_index()

            preds = [] # Lista vacía para almacenar las predicciones
            scores = [] # Lista vacía para almacenar las métricas

            for train_idx, val_idx in tss.split(df):
                    train = df.iloc[train_idx]
                    test = df.iloc[val_idx]

                    train = DataPreparator.create_features(train)
                    test = DataPreparator.create_features(test)

                    FEATURES = ['hour', "minute", 'dayofweek']

                    TARGET = "co2_emissions"

                    X_train = train[FEATURES]
                    y_train = train[TARGET]

                    X_test = test[FEATURES]
                    y_test = test[TARGET]

                    reg = xgb.XGBRegressor(base_score=0.5, booster = "gbtree",
                                            n_estimators=1000,
                                            early_stopping_rounds=50,
                                            objective = "reg:squarederror",
                                            max_depth = 4,
                                            learning_rate = 0.01)
                    
                    reg.fit(X_train, y_train,
                            eval_set = [(X_train, y_train), (X_test, y_test)],
                            verbose = 100)
                    
                    y_pred = reg.predict(X_test)
                    preds.append(y_pred)
                    score = np.sqrt(mean_squared_error(y_test, y_pred))
                    scores.append(score)
            
            print(f'Score across folds {np.mean(scores):0.4f}')
            print(f'Fold scores:{scores}')

            return reg, preds, scores


    def plot_k_folds(self, df, k:int = 4, test_size:int = 100, gap:int = 0):

        """
        Grafica las diferentes iteraciones que se llevarían a cabo con un
        k-fold en las series temporales

        Args:
            - df: pandas dataframe
            - k: número de folds
            - test_size: número de registros
            - gap: número de registros de espacio entre secuencias
        """

        tss = TimeSeriesSplit(n_splits=k, test_size=test_size, gap=gap)

        df = self.df.sort_index()

        fig, axs = plt.subplots(k, 1, figsize=(10, 10), sharex=True)
        fold = 0
        for train_idx, val_idx in tss.split(df):
            train = df.iloc[train_idx]['co2_emissions']
            test = df.iloc[val_idx]['co2_emissions']
            # plt.title(f'Data Train/Test Split Fold {fold}')
            axs[fold].set_title(f'Data Train/Test Split Fold {fold}')
            axs[fold].plot(train,
                                label='Training Set'
                                )
            axs[fold].plot(test,
                                label='Test Set')
            axs[fold].axvline(train.index.max(), color='black', ls='--')
            axs[fold].legend()
            fold += 1
        
        plt.show()

    def train_model_normal(self):




     