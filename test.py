import sys
sys.path.append("/home/arr/Documents/workspace/BS/productivizacion/MLE_FW/mle-framework/my_utils")

import pytest
import joblib
import pickle
import xgboost as xgb
import numpy as np
from test_utils import get_last_model
from sklearn.metrics import r2_score, mean_absolute_percentage_error

class TestClass:

    @pytest.fixture
    def model(self):
        return joblib.load(get_last_model("models", "pkl"))

    @pytest.fixture
    def X_test(self):
        return joblib.load("data/test/test_attributes.pkl")


    @pytest.fixture
    def y_test(self):
        return joblib.load("data/test/test_target.pkl")

    @pytest.fixture
    def y_pred(self, model, X_test):
        return model.predict(X_test)

    def test_r2_score(self, y_test, y_pred):
        try: 
            assert r2_score(y_true = y_test, y_pred = y_pred) >= 0.9
            
        except AssertionError as msg:
            print("R2 Score < 0.9!\n")
            print(msg)
            

    def test_mean_absolute_percentage_error(self, y_test, y_pred):
        try:
            assert mean_absolute_percentage_error(y_true = y_test, y_pred =y_pred) <= 0.15
        except AssertionError as msg:
            print("MAPE > 0.15!\n")
            print(msg)

        

