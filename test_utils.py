
import os
import pickle
import xgboost as xgb
import joblib
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_percentage_error

def get_last_model(path:str=None, format:str="pkl"):
    """
    Ordena los archivos según su fecha de modificación en un diccionario
    para obtener el último
    
    Retorna: directorio del archivo modificado en último lugar
    """
    format = "."+format
            
    files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(format)]
    # file_dict = {os.path.getmtime(f):f for f in files}
    file_dict = {os.path.getmtime(f)+i:f for f,i in zip(files, range(len(files)))}
    
    return file_dict[min(file_dict.keys())]

    
def load_model(filename:str):
        
    """
    Cargamos el modelo
    """
    
    xgb_model = joblib.load(filename)

    return xgb_model


def check_r2_score( y_real:np.array, y_predicted:np.array, thresh:float=0.9):   
    
    """
    Comprueba que el coeficiente de regresión
    sea igual o superior a 0.9
    """

    r2 = r2_score(y_true = y_real, y_pred = y_predicted)

    try:
        assert r2 >= thresh
    except AssertionError as msg:
     print(msg)

def check_mean_percentage_error(y_real:np.array, y_predicted:np.array,thresh:float=0.15):
    """
    Comprueba que el error porcentual medio absoluto
    sea igual o menor al 15%
    """

    mpe = mean_absolute_percentage_error(y_true = y_real, y_pred = y_predicted)

    try:
        assert mpe<=thresh
    except AssertionError as msg:
        print(msg)
