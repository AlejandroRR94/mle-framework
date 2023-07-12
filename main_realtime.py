import subprocess
from prefect import task, flow




@task(name="Activate Environment")
def activate_conda_environment():
    """
    Activa el entorno de producción
    """
    comando = "conda activate prod"
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

@task(name="ejecutar script")
def run_realtime():
    """
    Ejecuta el script de inferencia real time
    """

    comando = "python real_time_inference.py"
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
def run():
    activate_conda_environment()
    run_realtime()

if __name__ == "__main__":
    run()   
    
    