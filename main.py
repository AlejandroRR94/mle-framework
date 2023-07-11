

import sys
import os

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append("/home/arr/Documents/workspace/BS/productivizacion/MLE_FW/mle-framework/my_utils")
# print(sys.path)

from model_utils import *
from prepare_data_utils import *
import subprocess
from prefect import task, flow



if __name__ == "__main__":
    
    @task
    def ETL_PIPELINE():
        """
        Ejecuta el script "etl_pìpeline.py"
        """
        comando = "python pipelines/etl_pipeline.py"
        proceso = subprocess.Popen(comando, shell = True,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE)
        
        salida, error = proceso.communicate()

        if proceso.returncode == 0:
            print("La salida del comando es:")
            print(salida.decode("utf-8"))

        else:
            print("Se produjo un error al ejecutar el comando")
            print(error.decode("utf-8"))

    @task(name="")
    def TRAINING_PIPELINE():
        """
        Ejejcuta el script "train_pipeline.py"
        """
        comando = "python pipelines/train_pipeline.py"
        proceso = subprocess.Popen(comando, shell = True,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE)
        
        salida, error = proceso.communicate()

        if proceso.returncode == 0:
            print("La salida del comando es:")
            print(salida.decode("utf-8"))

        else:
            print("Se produjo un error al ejecutar el comando")
            print(error.decode("utf-8"))

    @flow
    def FULL_PIPELINE():
        
        print("\n####Starting the ETL...####")
        ETL_PIPELINE()
        print("ETL finished!\n\n####Starting training...####")

        TRAINING_PIPELINE()
        print("Training Finished!")

    if __name__ == "__main__":
        FULL_PIPELINE()