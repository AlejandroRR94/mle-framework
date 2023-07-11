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

# import importlib.util
# import sys
# spec = importlib.util.spec_from_file_location("module.name", "/home/arr/Documents/workspace/BS/productivizacion/MLE_FW/mle-framework/my_utils/prepare_data_utils.py")
# foo = importlib.util.module_from_spec(spec)
# sys.modules["prepare_data_utils.DataPreparator"] = foo
# spec.loader.exec_module(foo)
# dp = foo.DataPreparator()

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append("/home/arr/Documents/workspace/BS/productivizacion/MLE_FW/mle-framework/my_utils")
# print(sys.path)

from prepare_data_utils import *
from model_utils import *

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

    model,X_train, (X_test, y_test), y_pred, score = mt.train_model_normal(test_percentage=0.2)

    joblib.dump(X_test, "data/test_attributes.pkl")
    joblib.dump(y_test, "data/test_target.pkl")


    mt.save_model(model, "models/model")

    signature = infer_signature(X_train, y_pred)
    
    # print(type(model))

    # mlflow.xgboost.log_model(model,
    #                          artifact_path = "/mlruns/0/0e25f2e88bb44228841e04d3fe901532/artifacts/model/model.xgb",
    #                          signatures = signature # Guarda formato de inputs y outputs
    #                          )

