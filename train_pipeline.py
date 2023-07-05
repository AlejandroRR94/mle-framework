from prepare_data import *

import mlflow
from mlflow.models import Model, infer_signature
import mlflow.sklearn
import logging





# ESTABLECER TRACKING URI
logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


mlflow.xgboost.autolog()

with mlflow.start_run():
    df_1, df_2 = get_data("data/clean/")

    # MODIFICAR SCRIPT PARA QUE RETORNE EL OBJETO DEL MODELO
    model, preds, scores = train_k_fold(df_1)

    print(type(model))

    # mlflow.xgboost.log_model(model,
    #                          artifact_path = "path/to/weights",
    #                          signatures = signature # Guarda formato de inputs y outputs
                            #  )

# ENTRENAR CON TODA LA MUESTRA