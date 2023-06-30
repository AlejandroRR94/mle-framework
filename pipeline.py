import json
import requests
import pandas as pd
from datetime import date
from prefect import task, flow

### REQUEST TO API ###

@task
def extract_data(url:str, n_rows:int=150)->json:
    """
    Makes the request to the API.
    
    args:
        -url (string): url to request to
        -n_rows (int): number of rows of data we want the API to return

    Returns:
        -Pandas DataFrame with the information returned from the API
    """
    split_url = url.split("=")[0]
    full_url  =  split_url + "=" +  str(n_rows)
    print(full_url)

    response = requests.get(
                            full_url
                            )
    if not response:
        raise Exception("No data fetched!")
    result = response.json()
    
    
    records = result.get("records", [])

    fecha = date.today()

    dataframe = pd.DataFrame(records)
    dataframe.to_csv(f"data/raw/data_{fecha}.csv", index=False)


    return dataframe

@task
def rename_columns(
    df:pd.DataFrame
    ):
    """
    Changes column names
    """
    
    data = df.copy()

    data.drop(columns = ["HourDK"], inplace = True)

    data.rename(
        columns = {
            "HourUTC": "datetime_utc",
            "PriceArea": "area",
            "ConsumerType_DE35": "consumer_type",
            "TotalCon": "energy_consumption",
        },
        inplace=True
    )

    return data

@task
def cast_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Changes the column dtype 
    """
    data = df.copy()

    data["datetime_utc"] = pd.to_datetime(data["datetime_utc"])
    data["area"] = data["area"].astype("string")
    data["consumer_type"] = data["consumer_type"].astype("int32")
    data["energy_consumption"] = data["energy_consumption"].astype("float64")

    return data

@task
def encode_area_columns(df:pd.DataFrame) -> pd.DataFrame:
    """
    Applies label encoding to Area column
    """

    data = df.copy()

    area_mappings = {"DK": 0 , "DK1": 1, "DK2":2}

    data["area"] = data["area"].map(lambda string_area: area_mappings.get(string_area))
    data["area"] = data["area"].astype("int8")

    return data

@task
def load(data:pd.DataFrame, path:str) -> None:
    """
    Saves the file in the specified path
    """
    fecha = date.today()
    data.to_csv(path+f"clean_data_{str(fecha)}.csv",index=False)

@flow
def ETL():

    # Extracts the data from the url
    extraction = extract_data(
        url = 'https://api.energidataservice.dk/dataset/ConsumptionDE35Hour?limit=150',
        n_rows = 150
                            )
    transformation_0 = rename_columns(df = extraction)
    transformation_1 = cast_columns(transformation_0)
    transformation_2 = encode_area_columns(transformation_1)

    # Loads the data
    load(data = transformation_2, path = "data/clean/")

if __name__ == "__main__":
    ETL()