import pandas as pd
from typing import Tuple
from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.forecasting.naive import NaiveForecaster
from sktime.utils.plotting import plot_series
import seaborn as sns

def prepare_data(
    data: pd.DataFrame, target: str = "energy_consumption", fh: int = 24) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Structure the data for training:
    - Set the index as is required by sktime.
    - Prepare exogenous variables.
    - Prepare the time series to be forecasted.
    - Split the data into train and test sets.
    """

    # Set the index as is required by sktime.
    data["datetime_utc"] = pd.PeriodIndex(data["datetime_utc"], freq="H")
    data = data.set_index(["area", "consumer_type", "datetime_utc"]).sort_index()

    # Prepare exogenous variables.
    X = data.drop(columns=[target])
    # Prepare the time series to be forecasted.
    y = data[[target]]

    y_train, y_test, X_train, X_test = temporal_train_test_split(y, X, test_size=fh)

    return y_train, y_test, X_train, X_test

def train_forecaster(X_train, y_train):
    forecaster = NaiveForecaster()
    forecaster.fit(y)
    y_pred = forecaster.predict(fh=1)
