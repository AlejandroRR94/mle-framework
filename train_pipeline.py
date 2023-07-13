"""
Training pipeline
"""


import mlflow
from mlflow.models import Model, infer_signature
import mlflow.sklearn
import logging
import datetime
import joblib
import sys
import pathlib
import os

# print(sys.path)
from functions.prepare_data_utils import *
from functions.model_utils import *

# ESTABLECER TRACKING URI

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


mlflow.xgboost.autolog()

os.chdir(os.getcwd())

with mlflow.start_run():
    
    # Preparación de datos
    dp = DataPreparator(path = "data/clean/")
    df_1, df_2 = dp.get_data()

    mt = ModelTrainer(df=df_1, target = "co2_emissions")

    model,X_train, (X_test, y_test), y_pred, scores = mt.train_model_normal(test_percentage=0.2)
    
    mt.save_inference_metrics(model=model, filename="logs/inference_logs.json", y_true=y_test, y_pred=y_pred)

    joblib.dump(X_test, "data/test/test_attributes.pkl")
    joblib.dump(y_test, "data/test/test_target.pkl")


    mt.save_model(model, "models/model")

    signature = infer_signature(X_train, y_pred)
    
    # print(type(model))

    # mlflow.xgboost.log_model(model,
    #                          artifact_path = "/mlruns/0/0e25f2e88bb44228841e04d3fe901532/artifacts/model/model.xgb",
    #                          signatures = signature # Guarda formato de inputs y outputs
    #                          )

