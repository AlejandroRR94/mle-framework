"""
Classes used to train the model
"""

import json
from sktime.forecasting.model_selection import temporal_train_test_split
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
import xgboost as xgb
from lightgbm import LGBMRegressor
import pickle
import joblib
import datetime
import pandas as pd
from typing import Tuple

from functions.prepare_data_utils import *


class ModelTrainer():
    """
    Clase que engloba los métodos para entrenar un modelo.

    Args:

    - df: Pandas DataFrame a utilizar en los diferentes métodos
    - target: *string* que indica el nombre de la variable objetivo

    """
    def __init__(self, df:pd.DataFrame, target:str):
        self.df = df # Dataframe para utilizar
        self.target = target # Nombre de la columna objetivo

    # @task
    def train_k_fold(self, k:int=4, test_size:int = 100, gap:int=0)->Tuple[list, list]:

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

                    dp = DataPreparator()
                    train = dp.create_features(df=train)
                    test = dp.create_features(df=test)

                    FEATURES = ['hour', "minute", 'dayofweek', "lag1", "lag2", "lag3"]

                    TARGET = self.target

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


    def plot_k_folds(self, k:int = 4, test_size:int = 100, gap:int = 0):

        """
        Grafica las diferentes iteraciones que se llevarían a cabo con un
        k-fold en las series temporales

        Args:

            - df: pandas dataframe
            - k: número de folds
            - test_size: número de registros
            - gap: número de registros de espacio entre secuencias

        Returns:

        - Gráfico con los diferentes k-folds
        """

        tss = TimeSeriesSplit(n_splits=k, test_size=test_size, gap=gap)

        df = self.df.sort_index()

        fig, axs = plt.subplots(k, 1, figsize=(10, 10), sharex=True)
        fold = 0
        for train_idx, val_idx in tss.split(df):
            train = df.iloc[train_idx][self.target]
            test = df.iloc[val_idx][self.target]
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

    def training_split(self, test_percentage:float=0.2):
        """
        Realiza la separación entre los conjuntos de entrenamiento y
        evaluación según el porcentaje indicado en el argumento *test_percentage*

        Args:

        - test_percentage: *float* que indica el porcentaje de los datos destinado a
        evaluar el modelo

        """

        df = self.df.sort_index()
         
        test_size = int(len(self.df)*test_percentage)
        val_size = int(test_size/2)
        
        train_set = self.df[:-(test_size)]
        # val_set = self.df[-(test_size + val_size): - test_size]
            
        test_set = self.df[-test_size:]

        return train_set, test_set
    
    def make_x_y(self, data_set:pd.DataFrame)->Tuple[pd.DataFrame, pd.DataFrame]:
        """
        A partir de un dataset, genera las variables X e y para realizar el
        entrenamiento

        Args:

        - data_set: pandas dataframe del qeu extraer X e y

        Returns:

        - X, y: Tupla de variables atributo y variable target
        """
         
        dp = DataPreparator()
        SET = dp.create_features(data_set)
         
        FEATURES = ['hour', "minute", 'dayofweek', "lag1", "lag2", "lag3"]
        TARGET = self.target

        X = SET[FEATURES]
        y = SET[TARGET]

        return X, y




    def train_model_normal(self, test_percentage:float=0.2,
                           params:dict=None, lightgbm:bool=False
                           )->Tuple[object:tuple]:
        """
        Entrena un modelo con el split clásico de 2 conjuntos


        Args:
        
            - test_percentage
            - params: diccionario de parámetros para pasar al XGBoost o al 
            LightGBM
            - lightgbm: booleano que indica si se va a utilizar un LightGBM


        Returns:

            - model: objeto del modelo
            - X_train: Variables atributo de entrenamiento
            - 

        """

        train_set, test_set = self.training_split(test_percentage=test_percentage)

        X_train, y_train = self.make_x_y(train_set)
        # X_val, y_val = self.make_x_y(val_set)
        X_test, y_test = self.make_x_y(test_set)

        if params==None:
            reg = xgb.XGBRegressor(base_score=0.5, booster = "gbtree",
                                                n_estimators=1000,
                                                early_stopping_rounds=50,
                                                objective = "reg:squarederror",
                                                max_depth = 3,
                                                learning_rate = 0.01)
            
            ligth_reg = LGBMRegressor(base_score=0.5, booster = "gbtree",
                                                n_estimators=1000,
                                                early_stopping_rounds=50,
                                                objective = "reg:squarederror",
                                                max_depth = 3,
                                                learning_rate = 0.01)
        else:
            reg = xgb.XGBRegressor(**params)
            light_reg = LGBMRegressor(**params)
                    
        
        if lightgbm:
             model = light_reg
             model.fit(X_train, y_train)
        else:
             model = reg
             
             model.fit(X_train, 
                y_train,
                eval_set = [(X_train, y_train), (X_test, y_test)],
                )
        
        y_pred = reg.predict(X_test)
        
        train_metrics = self.create_metrics_dict(model = model, y_true = y_test, y_pred = y_pred)
        
        return model, X_train, (X_test, y_test), y_pred, train_metrics 
    

    def create_metrics_dict(self, model, y_true, y_pred,  date:datetime=datetime.now()):
         """
         Crea log con las métricas de interés del entrenamiento. Si no existe, 
         lo crea

         Retorna:

         - diccionario con los datos.
         """
        #  date = datetime.now()

         if "gboost" in str(type(model)):
            model = "xgboost"
         else:
            model="lightgbm"
            
        #  if os.path.isfile(filename) == False:
        #     models = [model]
        #     dates = [date]
        #     R2_score = [r2_score(y_true = y_true, y_pred = y_pred)]
        #     MAPE = [mean_absolute_percentage_error(y_true = y_true, y_pred = y_pred)]
        #     MAE = [mean_absolute_error(y_true = y_true, y_pred = y_pred)]
        #     RMSE = [np.sqrt(mean_squared_error(y_true = y_true, y_pred = y_pred)
        #                                  )]
            
        #     metrics = {
        #          "model":models,
        #          "training_date":dates,
        #          "r2_score":R2_score,
        #          "MAPE":MAPE,
        #          "MAE":MAE,
        #          "RMSE":RMSE
        #     }

        #     # out_file = open(filename, "w")
        #     # json.dump(metrics, out_file, indent=0, default=str)
        #     # out_file.close()
            
        #  else:
            
         metrics = {
        "model": model,
        "training_date": date,
        "r2_score":r2_score(y_true = y_true, y_pred = y_pred),
        "MAPE":mean_absolute_percentage_error(y_true = y_true, y_pred = y_pred),
        "MAE": mean_absolute_error(y_true = y_true, y_pred = y_pred),
        "RMSE":np.sqrt(mean_squared_error(y_true = y_true, y_pred = y_pred)),
        }
                
                
            
         return metrics
    
    def save_inference_metrics(self, filename:str, model, y_true, y_pred):
        """
        Recibe las métricas calculadas en la función create_metrics_dict()
        y las añade al json existente
        """
        
        metrics = self.create_metrics_dict(model, y_true, y_pred, filename)

        if os.path.isfile(filename): 
            with open(filename) as il:
                    inference_json = json.load(il)
            
            for k, v in zip(inference_json.keys(), metrics.values()):
                    inference_json[k].append(v)

            out_file = open(filename, "w")
            json.dump(inference_json, out_file, indent = 0, default = str)
            out_file.close()

            return inference_json
        
        else:
            with open(filename, "w") as out_file:
                  json.dump(metrics, out_file, indent=0, default=str)

            return metrics

    
    def save_data_logs(self, X:pd.DataFrame, key:str, training_logs:dict, inference_logs:dict):
         """
         Guardar log con los principales estadísticos de los datos

         Argumentos:

         - X: Atributos de los datos

         Retorna:

         - json con la información
         """

         logs = {
              "training": training_logs,
              "inference": inference_logs
              
         }

         return logs


    def train_all_data(self):
        """
        Entrena con todos los datos

        Retorna:

        - Objeto con el modelo entrenado
        """
        
        df = self.df.sort_index()
        df.drop(["area"], axis = 1, inplace = True)

        dp = DataPreparator()
        df_features = dp.create_features(df)
        df_lags = dp.add_lags(df_features)

        FEATURES = ['hour', "minute", 'dayofweek', "lag1", "lag2", "lag3"]
        
        X = df_lags[FEATURES]
        y = df_lags[self.target]

        reg = xgb.XGBRegressor(base_score=0.5, booster = "gbtree",
                                                n_estimators=1000,
                                                objective = "reg:squarederror",
                                                max_depth = 3,
                                                learning_rate = 0.01)
        
        reg.fit(X, y, verbose = 1)

        return reg
    
    def save_model(self, model:object, filename:str):
              
        """
        Guardamos el modelo tanto en formato pickle como en formato
        txt para una mejor representación y más facilidad en debugging
        """            
        import datetime

        date = datetime.date.today()
        joblib.dump(model, f"{filename}_xgboost_{date}.pkl")
        
        # model.save_model(f"{filename}_xgboost_{date}.pkl")
        model.save_model(f"{filename}_xgboost_{date}.json")
        # Guardamos la representación del modelo para futuros debuggings
        # model.dump_model(f"{filename}_xgboost_{date}.txt")
    
         
        
    def load_model(self, model:object, filename:str):
         
         """
         Cargamos el modelo
         """
         
         xgb_model = joblib.load(filename)

         return xgb_model