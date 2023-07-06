
from sktime.forecasting.model_selection import temporal_train_test_split
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
import xgboost as xgb

from prepare_data import * 


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
