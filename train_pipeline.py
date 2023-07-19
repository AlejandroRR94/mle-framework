"""
Training pipeline
"""


import mlflow
from mlflow.models import Model, infer_signature
import mlflow.sklearn
import logging
import joblib
import sys
import pathlib
import os
from datetime import date

# print(sys.path)
from functions.prepare_data_utils import *
from functions.model_utils import *
from functions.test_utils import * 

# ESTABLECER TRACKING URI

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

timestamp_today = datetime.now()
date_today = date.today()
experiment_name = f"experiment_{date_today}"
mlflow.set_experiment(experiment_name)

# mlflow.xgboost.autolog()

os.chdir(os.getcwd())

with mlflow.start_run():
    
    # Preparación de datos
    dp = DataPreparator(path = "data/clean/")
    df_1, df_2 = dp.get_data()

    mt = ModelTrainer(df=df_1, target = "co2_emissions")
    
    # Guardamos los artefactos de modelo
    print(date_today)
    artifact_path =f"MODELOS/modelo_{date_today}"
    print(artifact_path)
    model_name = f"modelo_xgboost_{timestamp_today}.pkl"
    model_path = os.path.join(artifact_path, model_name)
    print(model_path)
    if not os.path.isdir(artifact_path): 
        os.makedirs(artifact_path)
        print("carpeta de modelos creada")
        # os.chmod(model_path, 0o777)
    
    params = {
    'n_estimators': 1000,
    'max_depth': 3,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0,
    'reg_lambda': 1,
    'objective': 'reg:squarederror',
    'random_state': 42,
    'early_stopping_rounds':50,
    'random_state':42,
    'booster':'gbtree'
}


    model,X_train, (X_test, y_test), y_pred, scores = mt.train_model_normal(test_percentage=0.2,
                                                                            params = params)

    # Guardamos el modelo
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    
    # Añadimos el objeto al log
    mlflow.log_artifact(model_path)
    
    # Cargamos las métricas a MLFlow
    current_metrics, metrics_logs_json = mt.save_inference_metrics(model=model, filename="logs/inference_logs.json", y_true=y_test, y_pred=y_pred)
    # for k,v in metrics.items():
    #     mlflow.log_metric(k, v)
    

    joblib.dump(X_test, "data/test/test_attributes.pkl")
    joblib.dump(y_test, "data/test/test_target.pkl")


    mt.save_model(model, "models/model")

    signature = infer_signature(X_train, y_pred)
    # mlflow.log_metrics(metrics)
    mlflow.log_metrics(metrics={k:current_metrics[k] for k in list(current_metrics.keys()) if k not in ["model", "training_date"]})
    mlflow.xgboost.log_model(model, signature=signature, artifact_path=artifact_path)



    # print(type(model))

    # mlflow.xgboost.log_model(model,
    #                          artifact_path = "/mlruns/0/0e25f2e88bb44228841e04d3fe901532/artifacts/model/model.xgb",
    #                          signatures = signature # Guarda formato de inputs y outputs
    #                          )

mlflow.end_run()