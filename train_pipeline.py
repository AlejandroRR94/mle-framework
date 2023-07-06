from prepare_data import *
from model_utils import * 

import mlflow
from mlflow.models import Model, infer_signature
import mlflow.sklearn
import logging





# ESTABLECER TRACKING URI
logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


mlflow.xgboost.autolog()

with mlflow.start_run():
    
    # Preparación de datos
    dp = DataPreparator(path = "data/clean/")
    df_1, df_2 = dp.get_data()

    mt = ModelTrainer(df=df_1, target = "co2_emissions")

    # MODIFICAR SCRIPT PARA QUE RETORNE EL OBJETO DEL MODELO
    model,X_train, (X_test, y_test), y_pred, score = mt.train_model_normal(df_1,
                                              test_percentage=0.2)

    signature = infer_signature(X_train, y_pred)
    
    print(type(model))

    mlflow.xgboost.log_model(model,
                             artifact_path = "/mlruns/0/0e25f2e88bb44228841e04d3fe901532/artifacts/model/model.xgb",
                             signatures = signature # Guarda formato de inputs y outputs
                             )

# ENTRENAR CON TODA LA MUESTRA  