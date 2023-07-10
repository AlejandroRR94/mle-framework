import pickle
from test_utils import *
from model_utils import ModelTrainer
import os
import joblib
import json
import xgboost as xgb
from sklearn.metrics import mean_absolute_percentage_error, r2_score



models_path = "models/"


newest_model = get_last_model("models", "pkl")

# print(newest_model)

if newest_model.endswith("json"):
    with open(newest_model, "rb") as f:
        model_json = json.load(f)

        model = xgb.Booster()
        model.load_model(model_json)

else:
    model = load_model(newest_model)


X_test = joblib.load("data/test/test_attributes.pkl")
y_test = joblib.load("data/test/test_target.pkl")

test_predictions = model.predict(X_test)

print(mean_absolute_percentage_error(y_test, y_pred = test_predictions))
print(r2_score(y_true = y_test, y_pred = test_predictions))

check_r2_score(y_real = y_test, y_predicted = test_predictions, thresh = 0.9)

check_mean_percentage_error(y_real = y_test, y_predicted = test_predictions, thresh = 0.15)

# print(type(model))
